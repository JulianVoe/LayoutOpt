// layout_opt.cpp
// Build: g++ -O3 -std=c++20 layout_opt.cpp -o layout_opt
//
// This file is intentionally self-contained and designed to be extended.
// It implements:
//  - reading a keycode stream (whitespace-separated uint32s)
//  - a static hardware model (positions, fingers, effort, coordinates, permutable slots)
//  - a layout mapping from keycodes -> key positions (with fixed vs permutable symbols)
//  - corpus n-gram precomputation (uni/bi/tri) for fast evaluation
//  - a metric registry with default weights (some implemented, many stubs)
//  - a weighted score (with optional baseline normalization scaffolding)
//  - a simple simulated annealing local search over permutations of permutable symbols

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using Keycode = std::uint32_t;
using KeyId   = std::uint16_t;

enum class Hand : std::uint8_t { L, R, None };
enum class Finger : std::uint8_t { Thumb, Index, Middle, Ring, Pinky, None };

static inline const char* to_string(Finger f) {
  switch (f) {
    case Finger::Thumb:  return "thumb";
    case Finger::Index:  return "index";
    case Finger::Middle: return "middle";
    case Finger::Ring:   return "ring";
    case Finger::Pinky:  return "pinky";
    default:             return "none";
  }
}
static inline const char* to_string(Hand h) {
  switch (h) {
    case Hand::L: return "L";
    case Hand::R: return "R";
    default:      return "-";
  }
}

struct Vec2 {
  double x = 0.0;
  double y = 0.0;
};

static inline double dist(const Vec2& a, const Vec2& b) {
  const double dx = a.x - b.x;
  const double dy = a.y - b.y;
  return std::sqrt(dx*dx + dy*dy);
}

struct KeyPhysical {
  int row = 0;   // 0=top, 1=home, 2=bottom, 3+=thumb cluster etc.
  int col = 0;   // for your own modeling; used by some heuristics
  Vec2 pos{};    // coordinates in arbitrary units (consistent across keys)
  Hand hand = Hand::None;
  Finger finger = Finger::None;
  double effort = 0.0;  // per-press effort baseline
  bool is_thumb = false;
  bool is_permutable_slot = false; // whether this physical key is in the "letters" permutation pool
};

struct Hardware {
  std::vector<KeyPhysical> keys;
  std::vector<KeyId> permutable_slots;                  // physical slots to be permuted
  std::unordered_map<Keycode, KeyId> fixed_symbol_slot; // fixed symbols (shift/space/etc.)

  const KeyPhysical& at(KeyId k) const { return keys.at(static_cast<std::size_t>(k)); }

  void validate() const {
    for (KeyId k : permutable_slots) {
      if (k >= keys.size()) throw std::runtime_error("permutable slot out of range");
    }
  }
};

// --- Example hardware: "Voyager-like" 3x5 per hand + 3 thumbs per hand (36 keys total).
// Edit freely: what matters is that keys[] encodes coordinates, finger assignments, effort, and permutable slots.
static Hardware make_voyager_like_example() {
  Hardware hw;

  // Helper to add a key.
  auto add = [&](int row, int col, double x, double y, Hand hand, Finger finger,
                 double effort, bool thumb, bool perm) -> KeyId {
    KeyPhysical kp;
    kp.row = row; kp.col = col; kp.pos = {x,y};
    kp.hand = hand; kp.finger = finger;
    kp.effort = effort;
    kp.is_thumb = thumb;
    kp.is_permutable_slot = perm;
    hw.keys.push_back(kp);
    const KeyId id = static_cast<KeyId>(hw.keys.size() - 1);
    if (perm) hw.permutable_slots.push_back(id);
    return id;
  };

  // You should tune these to your board + your comfort model.
  // Effort baseline: lower for home/index/middle, higher for pinky/top/bottom.
  auto eff = [&](Finger f, int row, bool thumb)->double{
    double e = 1.0;
    if (thumb) e *= 0.6;
    switch (f) {
      case Finger::Index:  e *= 0.85; break;
      case Finger::Middle: e *= 0.80; break;
      case Finger::Ring:   e *= 1.05; break;
      case Finger::Pinky:  e *= 1.25; break;
      default: break;
    }
    if (row == 0) e *= 1.08;      // top row
    else if (row == 2) e *= 1.12; // bottom row
    return e;
  };

  // Layout grid: left hand cols 0..4, right hand cols 0..4; we space hands apart by +7 in x.
  const double gap = 7.0;

  auto finger_for_col = [&](int col)->Finger{
    if (col == 0) return Finger::Pinky;
    if (col == 1) return Finger::Ring;
    if (col == 2) return Finger::Middle;
    return Finger::Index; // cols 3,4 on index in this simplified model
  };

  // Left 3 rows x 5 cols, permutable slots = true for these 15.
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 5; ++c) {
      const Finger f = finger_for_col(c);
      add(r, c, /*x*/(double)c, /*y*/(double)r, Hand::L, f, eff(f,r,false), false, true);
    }
  }
  // Right
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 5; ++c) {
      const Finger f = finger_for_col(c);
      add(r, c, /*x*/gap + (double)c, /*y*/(double)r, Hand::R, f, eff(f,r,false), false, true);
    }
  }

  // Thumbs: 3 per hand, not permutable by default.
  // (You can decide to allow permutation here, but doing so changes the problem substantially.)
  for (int t = 0; t < 3; ++t) {
    add(3, t, 1.5 + (double)t, 3.2, Hand::L, Finger::Thumb, eff(Finger::Thumb,3,true), true, false);
    add(3, t, gap + 1.5 + (double)t, 3.2, Hand::R, Finger::Thumb, eff(Finger::Thumb,3,true), true, false);
  }

  // Fixed "hardware reality" symbol placements (edit to your keycode convention).
  // These fixed mappings mean: regardless of letter permutation, these symbols map to these physical keys.
  // Here we pick:
  //  - SPACE on left thumb middle,
  //  - SHIFT on left thumb left,
  //  - ENTER on right thumb middle,
  //  - BACKSPACE on right thumb right,
  //  - ESC on left thumb right.
  //
  // If your corpus stream uses different keycodes, change them here.
  constexpr Keycode KC_SPACE = 32;  // ' '
  constexpr Keycode KC_SHIFT = 1000;
  constexpr Keycode KC_ENTER = 13;
  constexpr Keycode KC_BSPC  = 8;
  constexpr Keycode KC_ESC   = 27;

  const KeyId L_THUMB_0 = static_cast<KeyId>(30); // left thumb cluster begins after 30 letter keys (15+15)
  const KeyId R_THUMB_0 = static_cast<KeyId>(33); // right thumbs after left thumbs (3 keys)
  hw.fixed_symbol_slot[KC_SHIFT] = L_THUMB_0 + 0;
  hw.fixed_symbol_slot[KC_SPACE] = L_THUMB_0 + 1;
  hw.fixed_symbol_slot[KC_ESC]   = L_THUMB_0 + 2;

  hw.fixed_symbol_slot[KC_ENTER] = R_THUMB_0 + 1;
  hw.fixed_symbol_slot[KC_BSPC]  = R_THUMB_0 + 2;

  hw.validate();
  return hw;
}

// ---------- Corpus (keycode stream + ngram counts) ----------

struct Pair {
  Keycode a, b;
  bool operator==(const Pair& o) const { return a==o.a && b==o.b; }
};
struct Triple {
  Keycode a, b, c;
  bool operator==(const Triple& o) const { return a==o.a && b==o.b && c==o.c; }
};
struct PairHash {
  std::size_t operator()(const Pair& p) const noexcept {
    return (std::size_t(p.a) * 1315423911u) ^ (std::size_t(p.b) + 0x9e3779b97f4a7c15ULL);
  }
};
struct TripleHash {
  std::size_t operator()(const Triple& t) const noexcept {
    std::size_t h = std::size_t(t.a);
    h = h * 1315423911u ^ std::size_t(t.b + 0x9e3779b9u);
    h = h * 1315423911u ^ std::size_t(t.c + 0x85ebca6bu);
    return h;
  }
};

struct Corpus {
  std::vector<Keycode> stream_all;
  std::vector<Keycode> stream_nomod;

  std::unordered_map<Keycode, std::uint64_t> uni_all;
  std::unordered_map<Pair, std::uint64_t, PairHash> bi_all;
  std::unordered_map<Triple, std::uint64_t, TripleHash> tri_all;

  std::unordered_map<Keycode, std::uint64_t> uni_nomod;
  std::unordered_map<Pair, std::uint64_t, PairHash> bi_nomod;
  std::unordered_map<Triple, std::uint64_t, TripleHash> tri_nomod;

  std::uint64_t n1_all = 0, n2_all = 0, n3_all = 0;
  std::uint64_t n1_nomod = 0, n2_nomod = 0, n3_nomod = 0;

  std::unordered_set<Keycode> modifier_keycodes;

  static Corpus read_keycodes(const std::string& path,
                              std::unordered_set<Keycode> modifier_set) {
    Corpus corp;
    corp.modifier_keycodes = std::move(modifier_set);

    std::ifstream in(path);
    if (!in) throw std::runtime_error("failed to open corpus file: " + path);

    Keycode kc;
    while (in >> kc) corp.stream_all.push_back(kc);
    if (corp.stream_all.size() < 2) throw std::runtime_error("corpus too small");

    // filtered stream (for "typing flow" metrics)
    corp.stream_nomod.reserve(corp.stream_all.size());
    for (Keycode x : corp.stream_all) {
      if (corp.modifier_keycodes.count(x) == 0) corp.stream_nomod.push_back(x);
    }

    auto build = [](const std::vector<Keycode>& s,
                    auto& uni, auto& bi, auto& tri,
                    std::uint64_t& n1, std::uint64_t& n2, std::uint64_t& n3) {
      n1 = n2 = n3 = 0;
      for (Keycode a : s) { uni[a]++; n1++; }
      for (std::size_t i=0; i+1<s.size(); ++i) { bi[{s[i], s[i+1]}]++; n2++; }
      for (std::size_t i=0; i+2<s.size(); ++i) { tri[{s[i], s[i+1], s[i+2]}]++; n3++; }
    };

    build(corp.stream_all,   corp.uni_all,   corp.bi_all,   corp.tri_all,   corp.n1_all,   corp.n2_all,   corp.n3_all);
    build(corp.stream_nomod, corp.uni_nomod, corp.bi_nomod, corp.tri_nomod, corp.n1_nomod, corp.n2_nomod, corp.n3_nomod);

    return corp;
  }
};

// ---------- Layout ----------

struct Layout {
  // permutable symbols are placed into hw.permutable_slots one-to-one.
  std::vector<Keycode> perm_symbols;       // size = P
  std::vector<KeyId>   perm_slot_of_sym;   // size = P (sym i -> slot)
  std::unordered_map<Keycode, std::size_t> perm_index; // sym -> i

  // fixed symbols map to fixed slots (from hardware); may also include additional fixed mappings
  std::unordered_map<Keycode, KeyId> fixed;

  // Lookup: keycode -> physical key id. Throws if unknown.
  KeyId slot_for(Keycode kc) const {
    if (auto it = fixed.find(kc); it != fixed.end()) return it->second;
    if (auto it = perm_index.find(kc); it != perm_index.end()) return perm_slot_of_sym[it->second];
    throw std::runtime_error("layout: unmapped keycode " + std::to_string(kc));
  }

  // Swap two permutable symbols' slots (move operator on permutations).
  void swap_symbols(std::size_t i, std::size_t j) {
    std::swap(perm_slot_of_sym[i], perm_slot_of_sym[j]);
  }

  static Layout make_initial(const Hardware& hw,
                             const std::vector<Keycode>& perm_syms_in_order,
                             std::unordered_map<Keycode, KeyId> fixed_extra = {}) {
    Layout L;
    L.fixed = hw.fixed_symbol_slot;
    for (auto& [k,v] : fixed_extra) L.fixed[k] = v;

    L.perm_symbols = perm_syms_in_order;
    L.perm_slot_of_sym.resize(L.perm_symbols.size());
    if (L.perm_symbols.size() != hw.permutable_slots.size()) {
      throw std::runtime_error("perm_symbols size (" + std::to_string(L.perm_symbols.size()) +
                               ") must equal permutable_slots size (" + std::to_string(hw.permutable_slots.size()) + ")");
    }
    for (std::size_t i=0; i<L.perm_symbols.size(); ++i) {
      L.perm_index[L.perm_symbols[i]] = i;
      L.perm_slot_of_sym[i] = hw.permutable_slots[i];
    }
    return L;
  }
};

// ---------- Evaluation Context ----------

struct EvalCtx {
  const Hardware& hw;
  const Corpus& corp;

  // which corpus to use for "flow" metrics
  bool ignore_modifiers_for_flow = true;

  // Accessors
  const auto& uni() const { return ignore_modifiers_for_flow ? corp.uni_nomod : corp.uni_all; }
  const auto& bi()  const { return ignore_modifiers_for_flow ? corp.bi_nomod  : corp.bi_all; }
  const auto& tri() const { return ignore_modifiers_for_flow ? corp.tri_nomod : corp.tri_all; }
  std::uint64_t n1() const { return ignore_modifiers_for_flow ? corp.n1_nomod : corp.n1_all; }
  std::uint64_t n2() const { return ignore_modifiers_for_flow ? corp.n2_nomod : corp.n2_all; }
  std::uint64_t n3() const { return ignore_modifiers_for_flow ? corp.n3_nomod : corp.n3_all; }
};

// ---------- Geometry / classification helpers ----------

static inline int finger_rank_out_to_in(Finger f) {
  // outward -> inward (pinky=0 ... index=3). Thumb is excluded from rolls by default.
  switch (f) {
    case Finger::Pinky:  return 0;
    case Finger::Ring:   return 1;
    case Finger::Middle: return 2;
    case Finger::Index:  return 3;
    default:             return -1;
  }
}

static inline bool adjacent_fingers(Finger a, Finger b) {
  const int ra = finger_rank_out_to_in(a);
  const int rb = finger_rank_out_to_in(b);
  if (ra < 0 || rb < 0) return false;
  return std::abs(ra - rb) == 1;
}

static inline bool is_scissor_full(const KeyPhysical& k1, const KeyPhysical& k2) {
  if (k1.hand != k2.hand) return false;
  if (!adjacent_fingers(k1.finger, k2.finger)) return false;
  // full scissor: one in top row, other in bottom row (relative to home row=1)
  return (k1.row == 0 && k2.row == 2) || (k1.row == 2 && k2.row == 0);
}

static inline bool is_scissor_half(const KeyPhysical& k1, const KeyPhysical& k2) {
  if (k1.hand != k2.hand) return false;
  if (!adjacent_fingers(k1.finger, k2.finger)) return false;
  // half scissor: one on home row and the other off-home (top or bottom)
  return (k1.row == 1 && (k2.row == 0 || k2.row == 2)) || (k2.row == 1 && (k1.row == 0 || k1.row == 2));
}

static inline bool is_lateral_stretch(const KeyPhysical& k1, const KeyPhysical& k2) {
  // Stubby but usable rule: same hand, and big horizontal gap in columns (>=2).
  // You will want to refine this based on actual Voyager geometry and your definition.
  if (k1.hand != k2.hand) return false;
  return std::abs(k1.col - k2.col) >= 2;
}

static inline bool is_same_finger(const KeyPhysical& k1, const KeyPhysical& k2) {
  return k1.finger == k2.finger && k1.finger != Finger::None;
}

static inline bool is_roll_in(const KeyPhysical& k1, const KeyPhysical& k2) {
  if (k1.hand != k2.hand) return false;
  const int r1 = finger_rank_out_to_in(k1.finger);
  const int r2 = finger_rank_out_to_in(k2.finger);
  if (r1 < 0 || r2 < 0 || r1 == r2) return false;
  return r2 > r1;
}

static inline bool is_roll_out(const KeyPhysical& k1, const KeyPhysical& k2) {
  if (k1.hand != k2.hand) return false;
  const int r1 = finger_rank_out_to_in(k1.finger);
  const int r2 = finger_rank_out_to_in(k2.finger);
  if (r1 < 0 || r2 < 0 || r1 == r2) return false;
  return r2 < r1;
}

static inline bool is_redirect_trigram(const KeyPhysical& k1, const KeyPhysical& k2, const KeyPhysical& k3) {
  if (k1.hand != k2.hand || k2.hand != k3.hand) return false;
  const int a = finger_rank_out_to_in(k1.finger);
  const int b = finger_rank_out_to_in(k2.finger);
  const int c = finger_rank_out_to_in(k3.finger);
  if (a < 0 || b < 0 || c < 0) return false;
  // direction reversal
  return (a < b && b > c) || (a > b && b < c);
}

static inline bool is_weak_redirect(const KeyPhysical& k1, const KeyPhysical& k2, const KeyPhysical& k3) {
  // "weak redirect": redirect where center finger is ring/pinky (common convention).
  if (!is_redirect_trigram(k1,k2,k3)) return false;
  return (k2.finger == Finger::Ring || k2.finger == Finger::Pinky);
}

static inline bool is_weakish_redirect(const KeyPhysical& k1, const KeyPhysical& k2, const KeyPhysical& k3) {
  // "weak-ish redirect": redirect where center finger is not index (or not thumb/index).
  if (!is_redirect_trigram(k1,k2,k3)) return false;
  return (k2.finger == Finger::Middle || k2.finger == Finger::Ring || k2.finger == Finger::Pinky);
}

static inline bool is_rakeable_same_finger(const KeyPhysical& k1, const KeyPhysical& k2) {
  // same finger, adjacent rows, same column
  if (!is_same_finger(k1,k2)) return false;
  return (k1.col == k2.col) && (std::abs(k1.row - k2.row) == 1);
}
static inline bool is_slideable_same_finger(const KeyPhysical& k1, const KeyPhysical& k2) {
  // same finger, adjacent columns, same row
  if (!is_same_finger(k1,k2)) return false;
  return (k1.row == k2.row) && (std::abs(k1.col - k2.col) == 1);
}

// ---------- Metrics ----------

enum class Direction { LowerBetter, HigherBetter };

struct MetricSpec {
  std::string name;
  double weight = 1.0;
  Direction dir = Direction::LowerBetter;
  double ideal_min = 0.0; // best-case lower bound (used for normalization)
  double ideal_max = 1.0; // best-case upper bound (used for normalization if HigherBetter and bounded)
  std::function<double(const EvalCtx&, const Layout&)> compute;
};

// Implemented metrics (core). Everything else is provided as stubs.
static double metric_effort(const EvalCtx& ctx, const Layout& L) {
  // Sum unigram frequency * per-key effort. (Uses flow corpus choice; you may want "all" here.)
  long double acc = 0.0L;
  const long double denom = (long double)ctx.n1();
  for (const auto& [kc, cnt] : ctx.uni()) {
    const KeyId kid = L.slot_for(kc);
    acc += (long double)cnt * ctx.hw.at(kid).effort;
  }
  return (denom > 0) ? (double)(acc / denom) : 0.0;
}

static double metric_travel(const EvalCtx& ctx, const Layout& L) {
  // Bigram Euclidean travel distance between key centers.
  long double acc = 0.0L;
  const long double denom = (long double)ctx.n2();
  for (const auto& [p, cnt] : ctx.bi()) {
    const KeyId a = L.slot_for(p.a);
    const KeyId b = L.slot_for(p.b);
    acc += (long double)cnt * dist(ctx.hw.at(a).pos, ctx.hw.at(b).pos);
  }
  return (denom > 0) ? (double)(acc / denom) : 0.0;
}

static double metric_sfb(const EvalCtx& ctx, const Layout& L) {
  // Same-finger bigrams rate (fraction of bigrams that are same-finger and different keys).
  long double bad = 0.0L;
  const long double denom = (long double)ctx.n2();
  for (const auto& [p, cnt] : ctx.bi()) {
    const KeyId a = L.slot_for(p.a);
    const KeyId b = L.slot_for(p.b);
    const auto& ka = ctx.hw.at(a);
    const auto& kb = ctx.hw.at(b);
    if (a != b && is_same_finger(ka, kb)) bad += (long double)cnt;
  }
  return (denom > 0) ? (double)(bad / denom) : 0.0;
}

static double metric_sfb_effective(const EvalCtx& ctx, const Layout& L,
                                   double rake_discount = 1.0, double slide_discount = 1.0) {
  // SFB - discounts for rakeable and slideable same-finger bigrams.
  long double sfb = 0.0L, rake = 0.0L, slide = 0.0L;
  const long double denom = (long double)ctx.n2();
  for (const auto& [p, cnt] : ctx.bi()) {
    const KeyId a = L.slot_for(p.a);
    const KeyId b = L.slot_for(p.b);
    if (a == b) continue;
    const auto& ka = ctx.hw.at(a);
    const auto& kb = ctx.hw.at(b);
    if (!is_same_finger(ka,kb)) continue;
    sfb += (long double)cnt;
    if (is_rakeable_same_finger(ka,kb))  rake  += (long double)cnt;
    if (is_slideable_same_finger(ka,kb)) slide += (long double)cnt;
  }
  const long double eff = sfb - rake_discount*rake - slide_discount*slide;
  return (denom > 0) ? (double)(eff / denom) : 0.0;
}

static double metric_skip_sfb(const EvalCtx& ctx, const Layout& L) {
  // Same-finger skipgrams (distance-1): for each trigram (a,b,c), consider (a,c).
  long double bad = 0.0L;
  const long double denom = (long double)ctx.n3();
  for (const auto& [t, cnt] : ctx.tri()) {
    const KeyId a = L.slot_for(t.a);
    const KeyId c = L.slot_for(t.c);
    if (a == c) continue;
    const auto& ka = ctx.hw.at(a);
    const auto& kc = ctx.hw.at(c);
    if (is_same_finger(ka,kc)) bad += (long double)cnt;
  }
  return (denom > 0) ? (double)(bad / denom) : 0.0;
}

static double metric_alternation(const EvalCtx& ctx, const Layout& L) {
  // Hand alternation rate (fraction of bigrams switching hands).
  long double alt = 0.0L;
  const long double denom = (long double)ctx.n2();
  for (const auto& [p, cnt] : ctx.bi()) {
    const KeyId a = L.slot_for(p.a);
    const KeyId b = L.slot_for(p.b);
    if (ctx.hw.at(a).hand != Hand::None && ctx.hw.at(b).hand != Hand::None &&
        ctx.hw.at(a).hand != ctx.hw.at(b).hand) {
      alt += (long double)cnt;
    }
  }
  return (denom > 0) ? (double)(alt / denom) : 0.0;
}

static double metric_roll_in(const EvalCtx& ctx, const Layout& L) {
  long double r = 0.0L;
  const long double denom = (long double)ctx.n2();
  for (const auto& [p, cnt] : ctx.bi()) {
    const auto& k1 = ctx.hw.at(L.slot_for(p.a));
    const auto& k2 = ctx.hw.at(L.slot_for(p.b));
    if (is_roll_in(k1,k2)) r += (long double)cnt;
  }
  return (denom > 0) ? (double)(r / denom) : 0.0;
}
static double metric_roll_out(const EvalCtx& ctx, const Layout& L) {
  long double r = 0.0L;
  const long double denom = (long double)ctx.n2();
  for (const auto& [p, cnt] : ctx.bi()) {
    const auto& k1 = ctx.hw.at(L.slot_for(p.a));
    const auto& k2 = ctx.hw.at(L.slot_for(p.b));
    if (is_roll_out(k1,k2)) r += (long double)cnt;
  }
  return (denom > 0) ? (double)(r / denom) : 0.0;
}

static double metric_redirects(const EvalCtx& ctx, const Layout& L) {
  long double bad = 0.0L;
  const long double denom = (long double)ctx.n3();
  for (const auto& [t, cnt] : ctx.tri()) {
    const auto& k1 = ctx.hw.at(L.slot_for(t.a));
    const auto& k2 = ctx.hw.at(L.slot_for(t.b));
    const auto& k3 = ctx.hw.at(L.slot_for(t.c));
    if (is_redirect_trigram(k1,k2,k3)) bad += (long double)cnt;
  }
  return (denom > 0) ? (double)(bad / denom) : 0.0;
}

static double metric_weak_redirects(const EvalCtx& ctx, const Layout& L) {
  long double bad = 0.0L;
  const long double denom = (long double)ctx.n3();
  for (const auto& [t, cnt] : ctx.tri()) {
    const auto& k1 = ctx.hw.at(L.slot_for(t.a));
    const auto& k2 = ctx.hw.at(L.slot_for(t.b));
    const auto& k3 = ctx.hw.at(L.slot_for(t.c));
    if (is_weak_redirect(k1,k2,k3)) bad += (long double)cnt;
  }
  return (denom > 0) ? (double)(bad / denom) : 0.0;
}

static double metric_weakish_redirects(const EvalCtx& ctx, const Layout& L) {
  long double bad = 0.0L;
  const long double denom = (long double)ctx.n3();
  for (const auto& [t, cnt] : ctx.tri()) {
    const auto& k1 = ctx.hw.at(L.slot_for(t.a));
    const auto& k2 = ctx.hw.at(L.slot_for(t.b));
    const auto& k3 = ctx.hw.at(L.slot_for(t.c));
    if (is_weakish_redirect(k1,k2,k3)) bad += (long double)cnt;
  }
  return (denom > 0) ? (double)(bad / denom) : 0.0;
}

static double metric_scissors_full(const EvalCtx& ctx, const Layout& L) {
  long double bad = 0.0L;
  const long double denom = (long double)ctx.n2();
  for (const auto& [p, cnt] : ctx.bi()) {
    const auto& k1 = ctx.hw.at(L.slot_for(p.a));
    const auto& k2 = ctx.hw.at(L.slot_for(p.b));
    if (is_scissor_full(k1,k2)) bad += (long double)cnt;
  }
  return (denom > 0) ? (double)(bad / denom) : 0.0;
}

static double metric_scissors_half(const EvalCtx& ctx, const Layout& L) {
  long double bad = 0.0L;
  const long double denom = (long double)ctx.n2();
  for (const auto& [p, cnt] : ctx.bi()) {
    const auto& k1 = ctx.hw.at(L.slot_for(p.a));
    const auto& k2 = ctx.hw.at(L.slot_for(p.b));
    if (is_scissor_half(k1,k2)) bad += (long double)cnt;
  }
  return (denom > 0) ? (double)(bad / denom) : 0.0;
}

static double metric_lateral_stretch(const EvalCtx& ctx, const Layout& L) {
  long double bad = 0.0L;
  const long double denom = (long double)ctx.n2();
  for (const auto& [p, cnt] : ctx.bi()) {
    const auto& k1 = ctx.hw.at(L.slot_for(p.a));
    const auto& k2 = ctx.hw.at(L.slot_for(p.b));
    if (is_lateral_stretch(k1,k2)) bad += (long double)cnt;
  }
  return (denom > 0) ? (double)(bad / denom) : 0.0;
}

// --- Modifier / shift metrics: these assume your corpus contains explicit modifier keycodes (e.g. KC_SHIFT).
// They are computed on the full stream (ctx.ignore_modifiers_for_flow should be false for these).
static double metric_modifier_effort(const EvalCtx& ctx, const Layout& L) {
  // Effort contribution from modifiers alone.
  long double acc = 0.0L;
  long double denom = 0.0L;
  for (const auto& [kc, cnt] : ctx.corp.uni_all) {
    if (ctx.corp.modifier_keycodes.count(kc) == 0) continue;
    denom += (long double)cnt;
    acc += (long double)cnt * ctx.hw.at(L.slot_for(kc)).effort;
  }
  return (denom > 0) ? (double)(acc / denom) : 0.0;
}

// ---- Stubs (extend as needed). Return 0.0 for now; you can implement later without touching the optimizer.
static double metric_stub(const EvalCtx&, const Layout&) { return 0.0; }

// ---------- Scoring ----------

struct ScoreBreakdown {
  struct Item { std::string name; double raw=0.0; double norm=0.0; double weight=0.0; };
  std::vector<Item> items;
  double score01 = 0.0;  // 0..1 (as "fraction of max improvement") if normalized; else arbitrary
};

struct Scorer {
  std::vector<MetricSpec> metrics;

  // Baseline values per metric (for normalization). If absent, score uses raw weighted sum.
  std::optional<std::unordered_map<std::string, double>> baseline;

  // Normalization model (QWERTY-fixed style).
  // For LowerBetter:  norm = (base - val)/(base - ideal_min)
  // For HigherBetter: norm = (val - base)/(ideal_max - base)
  // You can clamp norm if you want to limit outliers.
  static double normalize(const MetricSpec& m, double val, double base) {
    if (m.dir == Direction::LowerBetter) {
      const double denom = (base - m.ideal_min);
      if (denom <= 1e-12) return 0.0;
      return (base - val) / denom;
    } else {
      const double denom = (m.ideal_max - base);
      if (denom <= 1e-12) return 0.0;
      return (val - base) / denom;
    }
  }

  ScoreBreakdown evaluate(const EvalCtx& ctx, const Layout& L) const {
    ScoreBreakdown out;
    out.items.reserve(metrics.size());

    double wsum = 0.0;
    double acc  = 0.0;

    for (const auto& m : metrics) {
      const double raw = m.compute(ctx, L);
      double norm = raw;
      if (baseline) {
        auto it = baseline->find(m.name);
        if (it == baseline->end()) throw std::runtime_error("baseline missing metric: " + m.name);
        norm = normalize(m, raw, it->second);
      }
      out.items.push_back({m.name, raw, norm, m.weight});
      acc  += m.weight * norm;
      wsum += m.weight;
    }
    out.score01 = (wsum > 0) ? (acc / wsum) : 0.0;
    return out;
  }
};

// ---------- Local Search (Simulated Annealing over swaps) ----------

struct AnnealParams {
  std::uint64_t iters = 200000;
  double t0 = 0.10;        // initial temperature (in score units)
  double t_end = 1e-4;     // final temperature
  std::uint64_t seed = 1;
};

struct SearchResult {
  Layout best_layout;
  ScoreBreakdown best;
  ScoreBreakdown last;
};

static double temperature(std::uint64_t i, std::uint64_t n, double t0, double t1) {
  // Geometric schedule.
  if (n <= 1) return t1;
  const double frac = double(i) / double(n - 1);
  return t0 * std::pow(t1 / t0, frac);
}

static SearchResult anneal(const EvalCtx& ctx, const Scorer& scorer, Layout init, const AnnealParams& ap) {
  std::mt19937_64 rng(ap.seed);
  std::uniform_int_distribution<std::size_t> pick(0, init.perm_symbols.size() - 1);
  std::uniform_real_distribution<double> U(0.0, 1.0);

  auto cur = init;
  auto cur_bd = scorer.evaluate(ctx, cur);

  auto best = cur;
  auto best_bd = cur_bd;

  for (std::uint64_t it = 0; it < ap.iters; ++it) {
    const double T = temperature(it, ap.iters, ap.t0, ap.t_end);

    std::size_t i = pick(rng), j = pick(rng);
    while (j == i) j = pick(rng);

    cur.swap_symbols(i, j);
    auto cand_bd = scorer.evaluate(ctx, cur);

    const double delta = cand_bd.score01 - cur_bd.score01; // maximize
    bool accept = false;
    if (delta >= 0) accept = true;
    else {
      const double p = std::exp(delta / std::max(1e-12, T));
      accept = (U(rng) < p);
    }

    if (accept) {
      cur_bd = cand_bd;
      if (cur_bd.score01 > best_bd.score01) {
        best = cur;
        best_bd = cur_bd;
      }
    } else {
      // revert
      cur.swap_symbols(i, j);
    }
  }

  return {best, best_bd, cur_bd};
}

// ---------- Utility: choose permutable symbols ----------

static std::vector<Keycode> infer_permutable_symbols_from_corpus(const Corpus& corp,
                                                                 const Hardware& hw,
                                                                 const std::unordered_set<Keycode>& fixed_symbols_extra) {
  // Infer from corpus: all symbols that are not fixed (hardware fixed + extra fixed).
  // This is a convenience. In practice you may want an explicit allowlist (e.g. letters only).
  std::unordered_set<Keycode> fixed = fixed_symbols_extra;
  for (const auto& [kc, kid] : hw.fixed_symbol_slot) fixed.insert(kc);

  std::vector<Keycode> syms;
  syms.reserve(corp.uni_all.size());
  for (const auto& [kc, cnt] : corp.uni_all) {
    if (fixed.count(kc)) continue;
    syms.push_back(kc);
  }
  std::sort(syms.begin(), syms.end());
  // You must have exactly as many perm symbols as perm slots.
  if (syms.size() != hw.permutable_slots.size()) {
    std::ostringstream oss;
    oss << "inferred permutable symbols = " << syms.size()
        << " but permutable slots = " << hw.permutable_slots.size() << ".\n"
        << "You likely need to provide an explicit symbol allowlist (e.g. letters only),\n"
        << "or adjust which physical keys are permutable slots, or fix more symbols.";
    throw std::runtime_error(oss.str());
  }
  return syms;
}

// ---------- Metric registry (core + stubs) ----------

static Scorer make_default_scorer() {
  Scorer sc;

  // Defaults: weights are placeholders. You will tune them.
  // Direction/ideals chosen so QWERTY-normalization works when enabled.
  sc.metrics = {
    {"effort",            2.0, Direction::LowerBetter, 0.0, 1.0, metric_effort},
    {"travel",            1.5, Direction::LowerBetter, 0.0, 1.0, metric_travel},
    {"sfb",               3.0, Direction::LowerBetter, 0.0, 1.0, metric_sfb},
    {"sfb_effective",     2.0, Direction::LowerBetter, 0.0, 1.0,
      [](const EvalCtx& ctx, const Layout& L){ return metric_sfb_effective(ctx,L,1.0,1.0); }},
    {"skip_sfb",          1.0, Direction::LowerBetter, 0.0, 1.0, metric_skip_sfb},
    {"lateral_stretch",   1.0, Direction::LowerBetter, 0.0, 1.0, metric_lateral_stretch},
    {"scissors_full",     1.0, Direction::LowerBetter, 0.0, 1.0, metric_scissors_full},
    {"scissors_half",     0.7, Direction::LowerBetter, 0.0, 1.0, metric_scissors_half},
    {"alternation",       0.8, Direction::HigherBetter, 0.0, 1.0, metric_alternation},
    {"roll_in",           0.6, Direction::HigherBetter, 0.0, 1.0, metric_roll_in},
    {"roll_out",          0.6, Direction::LowerBetter, 0.0, 1.0, metric_roll_out},
    {"redirects",         1.0, Direction::LowerBetter, 0.0, 1.0, metric_redirects},
    {"weak_redirects",    0.7, Direction::LowerBetter, 0.0, 1.0, metric_weak_redirects},
    {"weakish_redirects", 0.7, Direction::LowerBetter, 0.0, 1.0, metric_weakish_redirects},

    // Modifier block (computed on ALL stream; see main() where ctx.ignore_modifiers_for_flow is toggled if desired).
    {"modifier_effort",   0.6, Direction::LowerBetter, 0.0, 1.0, metric_modifier_effort},

    // Stubs for future expansion (names align with the metrics discussed).
    {"pinky_off_home",    0.0, Direction::LowerBetter, 0.0, 1.0, metric_stub},
    {"pinky_dist",        0.0, Direction::LowerBetter, 0.0, 1.0, metric_stub},
    {"thumb_effort",      0.0, Direction::LowerBetter, 0.0, 1.0, metric_stub},
    {"finger_load",       0.0, Direction::LowerBetter, 0.0, 1.0, metric_stub},
    {"home_row_usage",    0.0, Direction::HigherBetter,0.0, 1.0, metric_stub},
    {"other_same_finger", 0.0, Direction::LowerBetter, 0.0, 1.0, metric_stub},
    {"3roll_in",          0.0, Direction::HigherBetter,0.0, 1.0, metric_stub},
    {"3roll_out",         0.0, Direction::LowerBetter, 0.0, 1.0, metric_stub},
    {"trigram_alt",       0.0, Direction::HigherBetter,0.0, 1.0, metric_stub},
    {"same_hand_shift",   0.0, Direction::LowerBetter, 0.0, 1.0, metric_stub},
    {"shift_chord_cost",  0.0, Direction::LowerBetter, 0.0, 1.0, metric_stub},
    {"skip_bigrams2",     0.0, Direction::LowerBetter, 0.0, 1.0, metric_stub},
  };

  return sc;
}

// ---------- Printing ----------

static void print_breakdown(const ScoreBreakdown& bd, bool normalized) {
  std::cout << (normalized ? "Score (normalized): " : "Score (raw-weighted): ")
            << std::fixed << std::setprecision(6) << bd.score01
            << (normalized ? "  (x100 gives % of max improvement)\n" : "\n");

  std::cout << "Metric breakdown:\n";
  for (const auto& it : bd.items) {
    std::cout << "  " << std::setw(18) << it.name
              << "  raw=" << std::setw(10) << std::setprecision(6) << it.raw
              << "  " << (normalized ? "norm=" : "val=")
              << std::setw(10) << std::setprecision(6) << it.norm
              << "  w=" << std::setw(6) << std::setprecision(3) << it.weight
              << "\n";
  }
}

static void usage(const char* argv0) {
  std::cerr << "Usage: " << argv0 << " <keycode_stream.txt> [iters] [seed]\n"
            << "  keycode_stream.txt: whitespace-separated uint32 keycodes\n"
            << "  iters (optional): annealing iterations (default 200000)\n"
            << "  seed  (optional): rng seed (default 1)\n";
}

// ---------- Main ----------

int main(int argc, char** argv) {
  try {
    if (argc < 2) { usage(argv[0]); return 2; }
    const std::string corpus_path = argv[1];
    std::uint64_t iters = (argc >= 3) ? std::stoull(argv[2]) : 200000ULL;
    std::uint64_t seed  = (argc >= 4) ? std::stoull(argv[3]) : 1ULL;

    Hardware hw = make_voyager_like_example();

    // Declare modifier keycodes for your corpus encoding.
    // If you explicitly log Shift/Ctrl/Alt as keycodes, list them here.
    // (These will be excluded from flow metrics when ignore_modifiers_for_flow=true.)
    std::unordered_set<Keycode> modifiers = {1000 /*KC_SHIFT*/};

    Corpus corp = Corpus::read_keycodes(corpus_path, modifiers);

    // Decide which symbols are fixed beyond hardware-fixed ones.
    // Typically you keep: shift/space/enter/backspace/esc fixed.
    std::unordered_set<Keycode> fixed_extra = {}; // add more if your corpus includes them and you want them fixed

    // Infer permutable symbols from corpus by excluding fixed. This is a convenience.
    // For real use, you might replace this by an explicit allowlist: e.g. ASCII letters only.
    std::vector<Keycode> perm_syms = infer_permutable_symbols_from_corpus(corp, hw, fixed_extra);

    Layout init = Layout::make_initial(hw, perm_syms);

    Scorer scorer = make_default_scorer();

    // Baseline normalization: use the initial layout as baseline.
    // For QWERTZ/QWERTY baselines, you would instead build a baseline layout mapping and evaluate that.
    {
      EvalCtx ctx{hw, corp};
      ctx.ignore_modifiers_for_flow = true;
      auto base_bd = scorer.evaluate(ctx, init);

      std::unordered_map<std::string, double> base;
      for (const auto& it : base_bd.items) base[it.name] = it.raw;
      scorer.baseline = std::move(base);
    }

    EvalCtx ctx{hw, corp};
    ctx.ignore_modifiers_for_flow = true;

    // Optional: if you want modifier metrics computed on ALL stream, you can either:
    //  - run a second pass with ctx.ignore_modifiers_for_flow=false and merge,
    //  - or keep modifier metrics implemented against corp.uni_all (as we did).
    // The current metric_modifier_effort ignores ctx.ignore_modifiers_for_flow and reads corp.uni_all directly.

    AnnealParams ap;
    ap.iters = iters;
    ap.seed = seed;
    ap.t0 = 0.10;
    ap.t_end = 1e-4;

    auto res = anneal(ctx, scorer, init, ap);

    std::cout << "Best score * 100: " << std::fixed << std::setprecision(3) << (100.0 * res.best.score01) << "\n";
    print_breakdown(res.best, /*normalized=*/true);

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
