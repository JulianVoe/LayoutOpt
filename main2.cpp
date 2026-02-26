// layout_opt.cpp
// Build: g++ -O3 -std=c++20 layout_opt.cpp -o layout_opt
//
// Data-oriented, single-file optimizer:
//
// 1) Reads your EVTLOG1 binary stream (uint16 keycodes, 0 = <PAUSE> boundary).
// 2) Builds unigram/bigram/trigram counts that DO NOT cross pauses.
// 3) Defines a static hardware model (keys with coords, finger, hand, row/col, effort).
// 4) Defines a multi-layer output model: each logical symbol is assigned to a (layer, key) slot.
//    Producing a symbol on layer>0 costs an extra layer-key press, so 1 logical token -> 1 or 2 physical presses.
// 5) Evaluates a layout via one accumulation pass that computes many metrics at once.
// 6) Runs simulated annealing (cosine-cycle temperature) with frequency-weighted swap proposals.
//
// Important conceptual note (you likely already see it):
// - Your file records physical Linux key presses on the CURRENT board (e.g. SHIFT + KEY_1 for '!').
// - This optimizer treats each recorded keycode as a "logical token you intend to produce on the new board".
//   That means SHIFT in the log remains SHIFT on the new board; digits remain digits, etc.
//   If you want "optimize for produced characters" (so '!' is a single token), you need a decoding stage
//   that turns the keycode stream into a symbol stream using modifier state. This file does NOT do that.
//
// Extension points are explicit:
// - Add more layers (nav layer, fn layer) by increasing layer_keys and slots.
// - Add/implement additional metrics in the accumulator + metric list.
// - Replace annealing with tabu / ILS / delta scoring later (layout state here is built for that).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <unordered_set>
#include <optional>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

using u8  = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

static constexpr u16 PAUSE_CODE = 0; // recorded + internal stream sentinel

// ---------------- EVTLOG reader (your format) ----------------

static constexpr char MAGIC[8] = {'E','V','T','L','O','G','1','\0'};
#pragma pack(push,1)
struct Header {
  char magic[8];
  u16  version;
  u16  record_size;
  u32  map_entries;
};
#pragma pack(pop)

static std::unordered_map<u16,std::string> read_evtlog(
    const std::string& path,
    std::vector<u16>& out_codes_including_pauses)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);

  Header h{};
  in.read(reinterpret_cast<char*>(&h), sizeof(h));
  if (!in) throw std::runtime_error("file too short (no header)");

  if (std::memcmp(h.magic, MAGIC, 8) != 0) throw std::runtime_error("bad magic (not EVTLOG1)");
  if (h.version != 1) throw std::runtime_error("unsupported version");
  if (h.record_size != sizeof(u16)) throw std::runtime_error("record_size mismatch (expected 2)");

  std::unordered_map<u16,std::string> code_to_name;
  code_to_name.reserve(h.map_entries * 2);

  for (u32 i=0; i<h.map_entries; ++i) {
    u16 code = 0; u8 n = 0;
    in.read(reinterpret_cast<char*>(&code), 2);
    in.read(reinterpret_cast<char*>(&n), 1);
    if (!in) throw std::runtime_error("truncated mapping table");
    std::string name(n, '\0');
    in.read(name.data(), n);
    if (!in) throw std::runtime_error("truncated mapping table (name)");
    code_to_name.emplace(code, std::move(name));
  }

  out_codes_including_pauses.clear();
  out_codes_including_pauses.reserve(1<<20);

  while (true) {
    u16 kc = 0;
    in.read(reinterpret_cast<char*>(&kc), 2);
    if (!in) break;
    out_codes_including_pauses.push_back(kc);
  }
  if (out_codes_including_pauses.empty()) throw std::runtime_error("no records");

  return code_to_name;
}

// ---------------- Corpus: compressed symbols + ngrams w/ pause boundaries ----------------

struct Corpus {
  std::unordered_map<u16,std::string> code_to_name;
  std::unordered_map<std::string,u16> name_to_code;
  std::unordered_set<u16> occurring_codes;
  std::vector<u16> stream;         // keycodes with PAUSE_CODE boundaries

  std::unordered_map<u16,u64> uni;
  std::unordered_map<u32,u64> bi;  // key = (a<<16)|b
  std::unordered_map<u64,u64> tri; // key = (a<<32)|(b<<16)|c

  u64 n1=0, n2=0, n3=0;
};

static inline u32 pack2(u16 a, u16 b) { return (u32(a) << 16) | u32(b); }
static inline u64 pack3(u16 a, u16 b, u16 c) { return (u64(a) << 32) | (u64(b) << 16) | u64(c); }

static Corpus build_corpus(
    const std::vector<u16>& codes_with_pauses,
    const std::unordered_map<u16,std::string>& code_to_name)
{
  Corpus C;
  C.code_to_name = code_to_name;
  C.name_to_code.reserve(code_to_name.size());
  for (const auto& [code, name] : code_to_name) C.name_to_code.emplace(name, code);

  C.stream.reserve(codes_with_pauses.size());
  for (u16 kc : codes_with_pauses) {
    C.stream.push_back(kc);
    if (kc != PAUSE_CODE) C.occurring_codes.insert(kc);
  }

  C.bi.reserve(std::min<u64>(C.stream.size(), 1<<20));
  C.tri.reserve(std::min<u64>(C.stream.size(), 1<<20));

  u16 prev1 = PAUSE_CODE;
  u16 prev2 = PAUSE_CODE;

  for (u16 x : C.stream) {
    if (x == PAUSE_CODE) {
      prev1 = prev2 = PAUSE_CODE;
      continue;
    }
    C.uni[x]++; C.n1++;

    if (prev1 != PAUSE_CODE) {
      C.bi[pack2(prev1, x)]++; C.n2++;
    }
    if (prev2 != PAUSE_CODE && prev1 != PAUSE_CODE) {
      C.tri[pack3(prev2, prev1, x)]++; C.n3++;
    }
    prev2 = prev1;
    prev1 = x;
  }

  return C;
}

// ---------------- Hardware model (edit freely) ----------------
//
// Finger encoding chosen so "outer->inner roll direction" is numeric order.
// 0=pinky,1=ring,2=middle,3=index,4=thumb. (Thumb excluded from roll logic by default.)
enum class FingerId : int { Pinky=0, Ring=1, Middle=2, Index=3, Thumb=4 };
enum class HandId : int { L=0, R=1 };

struct Vec2 {
  float x = 0.0f;
  float y = 0.0f;

  float dist(const Vec2& o) const {
    const float dx = x - o.x;
    const float dy = y - o.y;
    return std::sqrt(dx*dx + dy*dy);
  }
};

struct HW {
  int K = 0;
  std::vector<Vec2> pos;          // coords
  std::vector<int> hand, finger;   // 0/1, 0..4
  std::vector<int> row, col;       // user-defined
  std::vector<float> effort;       // per press

  std::vector<int> perm_keys;      // physical keys usable as "slots"
  std::vector<int> layer_key;      // layer_key[l] gives physical key id used to access layer l (l>=1). size = L-1.

  // home position for each hand/finger (used for static distance)
  std::array<std::array<Vec2,5>,2> home_pos{};
};

static HW make_voyager_like_example() {
  HW hw;

  auto add = [&](int row, int col, float x, float y, int hand, int finger, float effort, bool perm){
    hw.row.push_back(row);
    hw.col.push_back(col);
    hw.pos.push_back({x,y});
    hw.hand.push_back(hand);
    hw.finger.push_back(finger);
    hw.effort.push_back(effort);
    if (perm) hw.perm_keys.push_back(hw.K);
    hw.K++;
  };

  auto eff = [&](int finger, int row, bool thumb)->float{
    float e = 1.0f;
    if (thumb) e *= 0.6f;
    if (finger == (int)FingerId::Index)  e *= 0.85f;
    if (finger == (int)FingerId::Middle) e *= 0.80f;
    if (finger == (int)FingerId::Ring)   e *= 1.05f;
    if (finger == (int)FingerId::Pinky)  e *= 1.25f;
    if (row == 0) e *= 1.08f;
    if (row == 2) e *= 1.12f;
    return e;
  };

  auto finger_for_col = [&](int c)->int{
    if (c==0) return (int)FingerId::Pinky;
    if (c==1) return (int)FingerId::Ring;
    if (c==2) return (int)FingerId::Middle;
    return (int)FingerId::Index; // cols 3,4
  };

  const float gap = 7.0f;

  // 3x5 per hand, permutable
  for (int r=0;r<3;++r) for (int c=0;c<5;++c) {
    int f = finger_for_col(c);
    add(r,c, (float)c,(float)r, (int)HandId::L, f, eff(f,r,false), true);
  }
  for (int r=0;r<3;++r) for (int c=0;c<5;++c) {
    int f = finger_for_col(c);
    add(r,c, gap+(float)c,(float)r, (int)HandId::R, f, eff(f,r,false), true);
  }

  // 3 thumbs per hand, not permutable
  for (int t=0;t<3;++t) {
    add(3,t, 1.5f+(float)t, 3.2f, (int)HandId::L, (int)FingerId::Thumb, eff((int)FingerId::Thumb,3,true), false);
    add(3,t, gap+1.5f+(float)t, 3.2f, (int)HandId::R, (int)FingerId::Thumb, eff((int)FingerId::Thumb,3,true), false);
  }

  // Pick layer keys: you can support multiple layers if you have distinct layer keys.
  // layer_key[0] is for layer 1, layer_key[1] for layer 2, etc.
  // Here: use left thumb 0 for layer1, right thumb 0 for layer2 (if needed).
  const int left_thumb0  = 30; // after 30 perm keys
  const int right_thumb0 = 33; // after 3 left thumbs
  hw.layer_key = { left_thumb0, right_thumb0 };

  // Home positions: choose home-row key for each hand/finger by scanning row==1 and matching finger.
  for (int h=0; h<2; ++h) for (int f=0; f<5; ++f) {
    hw.home_pos[h][f] = {};
  }
  for (int h=0; h<2; ++h) {
    for (int f=0; f<4; ++f) { // exclude thumb
      bool found = false;
      for (int k=0; k<hw.K; ++k) {
        if (hw.hand[k]==h && hw.finger[k]==f && hw.row[k]==1) {
          hw.home_pos[h][f]=hw.pos[k];
          found = true;
          break;
        }
      }
      if (!found) { hw.home_pos[h][f]={}; }
    }
    // thumb home: pick first thumb
    for (int k=0;k<hw.K;++k) if (hw.hand[k]==h && hw.finger[k]==(int)FingerId::Thumb) {
      hw.home_pos[h][(int)FingerId::Thumb]=hw.pos[k];
      break;
    }
  }

  return hw;
}

// ---------------- Layout: assign each symbol-id to a slot (layer,key), or fixed key sequence ----------------

static constexpr int MAX_SEQ = 3; // allow extension (e.g. shift+layer+key)
struct Seq { u8 n=0; std::array<u16,MAX_SEQ> k{}; };

struct Layout {
  int layers = 1;              // number of output layers (0..layers-1). layer>0 uses a layer key press.
  int P = 0;                   // #permutable physical keys
  int slots = 0;               // layers * P

  // slot i corresponds to (layer = i/P, perm_index = i%P, physical key = perm_keys[perm_index])
  std::vector<int> slot_key;         // size slots -> physical key id
  std::vector<int> slot_layer;       // size slots -> layer
  std::vector<float> slot_cost;      // size slots (for greedy init)

  std::unordered_map<u16, int> code_slot; // -1 fixed, otherwise slot
  std::unordered_map<u16, Seq> code_seq;

  // Inverse: slot -> keycode, PAUSE_CODE if empty
  std::vector<u16> slot_code;

  // movable symbol list (those not fixed)
  std::vector<u16> movable;

  // symbol frequency weights for proposal distribution
  std::vector<double> movable_w;
};

static void build_slots(Layout& L, const HW& hw, int layers) {
  L.layers = layers;
  L.P = (int)hw.perm_keys.size();
  L.slots = L.layers * L.P;

  L.slot_key.resize(L.slots);
  L.slot_layer.resize(L.slots);
  L.slot_cost.resize(L.slots);

  for (int layer=0; layer<L.layers; ++layer) {
    for (int pi=0; pi<L.P; ++pi) {
      const int slot = layer*L.P + pi;
      const int key  = hw.perm_keys[pi];
      L.slot_key[slot] = key;
      L.slot_layer[slot] = layer;

      float c = hw.effort[key];
      if (layer>0) {
        const int lk = hw.layer_key[layer-1];
        c += hw.effort[lk];
        c += 0.15f * (float)layer; // tiny extra "mode friction" penalty; tune or remove
      }
      L.slot_cost[slot] = c;
    }
  }
}

static Seq seq_from_slot(const HW& hw, const Layout& L, int slot) {
  Seq s{};
  const int layer = L.slot_layer[slot];
  const int key   = L.slot_key[slot];
  if (layer == 0) {
    s.n = 1; s.k[0] = (u16)key;
  } else {
    const int lk = hw.layer_key[layer-1];
    s.n = 2; s.k[0] = (u16)lk; s.k[1] = (u16)key;
  }
  return s;
}

static inline u16 first_key(const Seq& s) { return s.k[0]; }
static inline u16 last_key (const Seq& s) { return s.k[s.n-1]; }

// ---------------- Metric accumulation (one pass computes many metrics) ----------------

struct Agg {
  // press-unigram denom
  u64 press_n = 0;

  // per-press accumulators
  double effort_sum = 0.0;
  double static_dist_sum = 0.0;
  u64 home_row_presses = 0, top_row_presses = 0, bot_row_presses = 0;
  std::array<std::array<u64,5>,2> finger_presses{}; // [hand][finger]
  u64 left_presses = 0, right_presses = 0;
  u64 pinky_off_home = 0;
  double pinky_dist_sum = 0.0;
  double thumb_effort_sum = 0.0;

  // press-bigram denom
  u64 big_n = 0;

  // bigram accumulators
  double travel_sum = 0.0;
  u64 sfb = 0, sfb_rake = 0, sfb_slide = 0;
  u64 lat_stretch = 0;
  u64 scissors_full = 0, scissors_half = 0;
  u64 alternation = 0;
  u64 roll_in = 0, roll_out = 0;

  // press-trigram denom
  u64 tri_n = 0;

  // trigram accumulators
  u64 redirects = 0, weak_redirects = 0, weakish_redirects = 0;
  u64 skip_sfb = 0;
  u64 trigram_alt = 0;
  u64 roll3_in = 0, roll3_out = 0;
};

// cheap predicates, based only on HW arrays
static inline bool same_hand(const HW& hw, u16 a, u16 b) { return hw.hand[a]==hw.hand[b]; }
static inline bool diff_hand(const HW& hw, u16 a, u16 b) { return hw.hand[a]!=hw.hand[b]; }
static inline bool same_finger(const HW& hw, u16 a, u16 b) { return hw.finger[a]==hw.finger[b]; }

static inline bool adjacent_nonthumb_fingers(const HW& hw, u16 a, u16 b) {
  int fa=hw.finger[a], fb=hw.finger[b];
  if (fa==(int)FingerId::Thumb || fb==(int)FingerId::Thumb) return false;
  return std::abs(fa-fb)==1;
}

static inline bool scissor_full(const HW& hw, u16 a, u16 b) {
  if (!same_hand(hw,a,b)) return false;
  if (!adjacent_nonthumb_fingers(hw,a,b)) return false;
  return (hw.row[a]==0 && hw.row[b]==2) || (hw.row[a]==2 && hw.row[b]==0);
}
static inline bool scissor_half(const HW& hw, u16 a, u16 b) {
  if (!same_hand(hw,a,b)) return false;
  if (!adjacent_nonthumb_fingers(hw,a,b)) return false;
  return (hw.row[a]==1 && (hw.row[b]==0 || hw.row[b]==2)) || (hw.row[b]==1 && (hw.row[a]==0 || hw.row[a]==2));
}
static inline bool lateral_stretch(const HW& hw, u16 a, u16 b) {
  if (!same_hand(hw,a,b)) return false;
  return std::abs(hw.col[a]-hw.col[b]) >= 2;
}
static inline bool rakeable_sfb(const HW& hw, u16 a, u16 b) {
  if (!same_finger(hw,a,b)) return false;
  return (hw.col[a]==hw.col[b]) && (std::abs(hw.row[a]-hw.row[b])==1);
}
static inline bool slideable_sfb(const HW& hw, u16 a, u16 b) {
  if (!same_finger(hw,a,b)) return false;
  return (hw.row[a]==hw.row[b]) && (std::abs(hw.col[a]-hw.col[b])==1);
}
static inline bool roll_in(const HW& hw, u16 a, u16 b) {
  if (!same_hand(hw,a,b)) return false;
  int fa=hw.finger[a], fb=hw.finger[b];
  if (fa==(int)FingerId::Thumb || fb==(int)FingerId::Thumb || fa==fb) return false;
  return fb > fa; // outer->inner
}
static inline bool roll_out(const HW& hw, u16 a, u16 b) {
  if (!same_hand(hw,a,b)) return false;
  int fa=hw.finger[a], fb=hw.finger[b];
  if (fa==(int)FingerId::Thumb || fb==(int)FingerId::Thumb || fa==fb) return false;
  return fb < fa;
}
static inline bool redirect_trigram(const HW& hw, u16 a, u16 b, u16 c) {
  if (!(same_hand(hw,a,b) && same_hand(hw,b,c))) return false;
  int fa=hw.finger[a], fb=hw.finger[b], fc=hw.finger[c];
  if (fa==(int)FingerId::Thumb || fb==(int)FingerId::Thumb || fc==(int)FingerId::Thumb) return false;
  return (fa < fb && fb > fc) || (fa > fb && fb < fc);
}
static inline bool weak_redirect_trigram(const HW& hw, u16 a, u16 b, u16 c) {
  if (!redirect_trigram(hw,a,b,c)) return false;
  int fb=hw.finger[b];
  return fb==(int)FingerId::Ring || fb==(int)FingerId::Pinky;
}
static inline bool weakish_redirect_trigram(const HW& hw, u16 a, u16 b, u16 c) {
  if (!redirect_trigram(hw,a,b,c)) return false;
  int fb=hw.finger[b];
  return fb==(int)FingerId::Middle || fb==(int)FingerId::Ring || fb==(int)FingerId::Pinky;
}
static inline bool tri_alternation(const HW& hw, u16 a, u16 b, u16 c) {
  int ha=hw.hand[a], hb=hw.hand[b], hc=hw.hand[c];
  if (ha<0||hb<0||hc<0) return false;
  return (ha!=hb && hb!=hc && ha==hc);
}
static inline bool roll3_in(const HW& hw, u16 a, u16 b, u16 c) {
  if (!(same_hand(hw,a,b) && same_hand(hw,b,c))) return false;
  int fa=hw.finger[a], fb=hw.finger[b], fc=hw.finger[c];
  if (fa==(int)FingerId::Thumb || fb==(int)FingerId::Thumb || fc==(int)FingerId::Thumb) return false;
  return (fa < fb && fb < fc);
}
static inline bool roll3_out(const HW& hw, u16 a, u16 b, u16 c) {
  if (!(same_hand(hw,a,b) && same_hand(hw,b,c))) return false;
  int fa=hw.finger[a], fb=hw.finger[b], fc=hw.finger[c];
  if (fa==(int)FingerId::Thumb || fb==(int)FingerId::Thumb || fc==(int)FingerId::Thumb) return false;
  return (fa > fb && fb > fc);
}

static inline void add_press(Agg& A, const HW& hw, u16 k, u64 w) {
  A.press_n += w;
  A.effort_sum += (double)w * hw.effort[k];

  const int h = hw.hand[k];
  const int f = hw.finger[k];

  if (h==(int)HandId::L) A.left_presses += w;
  if (h==(int)HandId::R) A.right_presses += w;

  if (h==(int)HandId::L || h==(int)HandId::R) {
    if (0<=f && f<5) A.finger_presses[h][f] += w;
    const Vec2 home = hw.home_pos[h][std::clamp(f,0,4)];
    A.static_dist_sum += (double)w * hw.pos[k].dist(home);
  }

  if (hw.row[k]==1) A.home_row_presses += w;
  if (hw.row[k]==0) A.top_row_presses += w;
  if (hw.row[k]==2) A.bot_row_presses += w;

  if (f==(int)FingerId::Pinky) {
    if (hw.row[k]!=1) A.pinky_off_home += w;
    const Vec2 home = hw.home_pos[h][(int)FingerId::Pinky];
    A.pinky_dist_sum += (double)w * hw.pos[k].dist(home);
  }
  if (f==(int)FingerId::Thumb) A.thumb_effort_sum += (double)w * hw.effort[k];
}

static inline void add_bigram(Agg& A, const HW& hw, u16 a, u16 b, u64 w) {
  A.big_n += w;
  A.travel_sum += (double)w * hw.pos[a].dist(hw.pos[b]);

  if (a!=b && same_finger(hw,a,b)) {
    A.sfb += w;
    if (rakeable_sfb(hw,a,b)) A.sfb_rake += w;
    if (slideable_sfb(hw,a,b)) A.sfb_slide += w;
  }
  if (lateral_stretch(hw,a,b)) A.lat_stretch += w;

  if (scissor_full(hw,a,b)) A.scissors_full += w;
  if (scissor_half(hw,a,b)) A.scissors_half += w;

  if (diff_hand(hw,a,b)) A.alternation += w;

  if (roll_in(hw,a,b))  A.roll_in  += w;
  if (roll_out(hw,a,b)) A.roll_out += w;
}

static inline void add_trigram(Agg& A, const HW& hw, u16 a, u16 b, u16 c, u64 w) {
  A.tri_n += w;
  if (redirect_trigram(hw,a,b,c)) A.redirects += w;
  if (weak_redirect_trigram(hw,a,b,c)) A.weak_redirects += w;
  if (weakish_redirect_trigram(hw,a,b,c)) A.weakish_redirects += w;

  if (a!=c && same_finger(hw,a,c)) A.skip_sfb += w;
  if (tri_alternation(hw,a,b,c)) A.trigram_alt += w;

  if (roll3_in(hw,a,b,c))  A.roll3_in  += w;
  if (roll3_out(hw,a,b,c)) A.roll3_out += w;
}

static Agg accumulate(const HW& hw, const Corpus& C, const Layout& L) {
  Agg A{};

  u16 prev1 = std::numeric_limits<u16>::max();
  u16 prev2 = std::numeric_limits<u16>::max();
  for (u16 code : C.stream) {
    if (code == PAUSE_CODE) {
      prev1 = prev2 = std::numeric_limits<u16>::max();
      continue;
    }
    const Seq& seq = L.code_seq.at(code);
    for (int i = 0; i < (int)seq.n; ++i) {
      const u16 key = seq.k[i];
      add_press(A, hw, key, 1);
      if (prev1 != std::numeric_limits<u16>::max()) add_bigram(A, hw, prev1, key, 1);
      if (prev2 != std::numeric_limits<u16>::max()) add_trigram(A, hw, prev2, prev1, key, 1);
      prev2 = prev1;
      prev1 = key;
    }
  }

  return A;
}

// ---------------- Metrics (ALL discussed metrics are present; many are stubs for now) ----------------

enum Dir { LowerBetter=0, HigherBetter=1 };

struct Metric {
  const char* name;
  double w;
  Dir dir;
  double (*val)(const Agg&);
};

// ---- stubs requested (present; implement later without changing optimizer plumbing)
static double stub0(const Agg&){ return 0.0; }

// default metric list: ALL metrics discussed are present as entries
static std::vector<Metric> default_metrics() {
  return {
    // Static load / effort / distance
    {"effort",            2.0, LowerBetter,  [](const Agg& A){ return (A.press_n? A.effort_sum / (double)A.press_n : 0.0); }},
    {"distance",          1.0, LowerBetter,  [](const Agg& A){ return (A.press_n? A.static_dist_sum / (double)A.press_n : 0.0); }},
    {"thumb_effort",      0.3, LowerBetter,  [](const Agg& A){ return (A.press_n? A.thumb_effort_sum / (double)A.press_n : 0.0); }},
    {"finger_usage",      0.0, LowerBetter,  stub0},   // could be penalty vs targets
    {"column_usage",      0.0, LowerBetter,  stub0},
    {"col5&6",            0.0, LowerBetter,  stub0},
    {"row_home",          0.5, HigherBetter, [](const Agg& A){ return (A.press_n? (double)A.home_row_presses / (double)A.press_n : 0.0); }},
    {"row_top",           0.0, LowerBetter,  [](const Agg& A){ return (A.press_n? (double)A.top_row_presses / (double)A.press_n : 0.0); }},
    {"row_bottom",        0.0, LowerBetter,  [](const Agg& A){ return (A.press_n? (double)A.bot_row_presses / (double)A.press_n : 0.0); }},
    {"pinky_dist",        0.2, LowerBetter,  [](const Agg& A){ return (A.press_n? A.pinky_dist_sum / (double)A.press_n : 0.0); }},
    {"pinky_off_home",    0.3, LowerBetter,  [](const Agg& A){ return (A.press_n? (double)A.pinky_off_home / (double)A.press_n : 0.0); }},
    {"hand_balance",      0.2, LowerBetter,  [](const Agg& A){ if (!A.press_n) return 0.0; double L = (double)A.left_presses, R = (double)A.right_presses; return std::abs(L - R) / (L + R + 1e-12); }},

    // Bigram motion / conflicts
    {"travel",            1.5, LowerBetter,  [](const Agg& A){ return (A.big_n? A.travel_sum / (double)A.big_n : 0.0); }},
    {"sfb",               3.0, LowerBetter,  [](const Agg& A){ return (A.big_n? (double)A.sfb / (double)A.big_n : 0.0); }},
    {"sfb_rakeable",      0.0, LowerBetter,  [](const Agg& A){ return (A.big_n? (double)A.sfb_rake / (double)A.big_n : 0.0); }},
    {"sfb_slideable",     0.0, LowerBetter,  [](const Agg& A){ return (A.big_n? (double)A.sfb_slide / (double)A.big_n : 0.0); }},
    {"sfb_effective",     2.0, LowerBetter,  [](const Agg& A){ if (!A.big_n) return 0.0; double eff = (double)A.sfb - (double)A.sfb_rake - (double)A.sfb_slide; return eff / (double)A.big_n; }},
    {"sfb_2u",            0.0, LowerBetter,  stub0},   // analyzer-specific; implement if you define it
    {"skip_bigrams",      0.0, LowerBetter,  stub0},   // general skip bigrams
    {"skip_bigrams2",     0.0, LowerBetter,  stub0},   // requires 4-grams
    {"other_same_finger", 0.0, LowerBetter,  stub0},

    {"lat_stretch",       0.8, LowerBetter,  [](const Agg& A){ return (A.big_n? (double)A.lat_stretch / (double)A.big_n : 0.0); }},
    {"lsb",               0.0, LowerBetter,  stub0},

    {"scissors",          0.0, LowerBetter,  stub0},
    {"full_scissor",      0.8, LowerBetter,  [](const Agg& A){ return (A.big_n? (double)A.scissors_full / (double)A.big_n : 0.0); }},
    {"half_scissor",      0.5, LowerBetter,  [](const Agg& A){ return (A.big_n? (double)A.scissors_half / (double)A.big_n : 0.0); }},
    {"pinky_scissors",    0.0, LowerBetter,  stub0},
    {"wide_scissors",     0.0, LowerBetter,  stub0},

    {"alternation",       0.6, HigherBetter, [](const Agg& A){ return (A.big_n? (double)A.alternation / (double)A.big_n : 0.0); }},
    {"roll_in",           0.4, HigherBetter, [](const Agg& A){ return (A.big_n? (double)A.roll_in / (double)A.big_n : 0.0); }},
    {"roll_out",          0.4, LowerBetter,  [](const Agg& A){ return (A.big_n? (double)A.roll_out / (double)A.big_n : 0.0); }},
    {"rolls:alts",        0.0, HigherBetter, stub0},
    {"2-roll_in:out",     0.0, HigherBetter, stub0},

    // Trigram flow
    {"skip_sfb",          0.6, LowerBetter,  [](const Agg& A){ return (A.tri_n? (double)A.skip_sfb / (double)A.tri_n : 0.0); }},
    {"trigram_alt",       0.3, HigherBetter, [](const Agg& A){ return (A.tri_n? (double)A.trigram_alt / (double)A.tri_n : 0.0); }},
    {"redir",             0.8, LowerBetter,  [](const Agg& A){ return (A.tri_n? (double)A.redirects / (double)A.tri_n : 0.0); }},
    {"weak_redir",        0.5, LowerBetter,  [](const Agg& A){ return (A.tri_n? (double)A.weak_redirects / (double)A.tri_n : 0.0); }},
    {"weakish_redir",     0.5, LowerBetter,  [](const Agg& A){ return (A.tri_n? (double)A.weakish_redirects / (double)A.tri_n : 0.0); }},
    {"3-roll_in",         0.0, HigherBetter, [](const Agg& A){ return (A.tri_n? (double)A.roll3_in / (double)A.tri_n : 0.0); }},
    {"3-roll_out",        0.0, LowerBetter,  [](const Agg& A){ return (A.tri_n? (double)A.roll3_out / (double)A.tri_n : 0.0); }},

    // Shift/modifier related (your log treats Shift as its own token; these are higher-level chord metrics)
    {"shift_rate",        0.0, LowerBetter,  stub0},
    {"same_hand_shift",   0.0, LowerBetter,  stub0},
    {"shift_chord_cost",  0.0, LowerBetter,  stub0},

    // Meta / reporting
    {"bigram_total",      0.0, HigherBetter, stub0},
  };
}

static double score_layout(const std::vector<Metric>& M, const Agg& A) {
  // Data-simple scoring: sum(weight * signed_value), where HigherBetter is negated so "lower cost is better".
  // Score = -cost, so higher score is better.
  double cost = 0.0;
  for (const auto& m : M) {
    double v = m.val(A);
    cost += m.w * (m.dir==LowerBetter ? v : -v);
  }
  return -cost;
}

// ---------------- Build initial layout (greedy by token frequency vs slot cost) ----------------

static std::optional<u16> code_named(const Corpus& C, std::string_view name) {
  if (auto it = C.name_to_code.find(std::string(name)); it != C.name_to_code.end()) return it->second;
  return std::nullopt;
}

static void build_initial_layout(
    Layout& L, const HW& hw, const Corpus& C)
{
  L.code_slot.clear();
  L.code_seq.clear();
  L.movable.clear();

  // Decide fixed tokens (hardware reality-ish). Detected by Linux names if present.
  // You can extend this list freely.
  std::vector<std::string_view> fixed_names = {
    "KEY_LEFTSHIFT","KEY_RIGHTSHIFT","KEY_LEFTCTRL","KEY_RIGHTCTRL",
    "KEY_LEFTALT","KEY_RIGHTALT","KEY_LEFTMETA","KEY_RIGHTMETA",
    "KEY_SPACE","KEY_ENTER","KEY_BACKSPACE","KEY_TAB","KEY_ESC"
  };

  // Choose physical keys for these fixed tokens (example only; adapt to your actual voyager map).
  // We'll pin them to thumbs here.
  const int left_thumb0  = 30;
  const int left_thumb1  = 31;
  const int left_thumb2  = 32;
  const int right_thumb0 = 33;
  const int right_thumb1 = 34;
  const int right_thumb2 = 35;

  auto fixed_physical = [&](std::string_view nm)->std::optional<int>{
    if (nm=="KEY_SPACE")      return left_thumb1;
    if (nm=="KEY_ENTER")      return right_thumb1;
    if (nm=="KEY_BACKSPACE")  return right_thumb2;
    if (nm=="KEY_TAB")        return left_thumb2;
    if (nm=="KEY_ESC")        return left_thumb0;
    if (nm=="KEY_LEFTSHIFT")  return left_thumb0;  // example: share with esc is bad; change on real setup
    if (nm=="KEY_RIGHTSHIFT") return right_thumb0;
    if (nm=="KEY_LEFTCTRL")   return left_thumb2;
    if (nm=="KEY_RIGHTCTRL")  return right_thumb2;
    return std::nullopt;
  };

  // Mark fixed by name if present in header map.
  for (auto nm : fixed_names) {
    const auto code = code_named(C, nm);
    if (!code) continue;

    auto pk = fixed_physical(nm);
    if (!pk) continue;

    Seq s{}; s.n = 1; s.k[0] = (u16)(*pk);
    L.code_slot[*code] = -1;
    L.code_seq[*code] = s;
  }

  // Movable symbols: everything else that appears.
  for (u16 code : C.occurring_codes) {
    if (code == PAUSE_CODE) continue;
    if (L.code_slot.contains(code)) continue;
    L.movable.push_back(code);
  }

  // Choose number of layers to have enough slots for all movable symbols.
  // Max layers = 1 + number of layer keys available.
  const int P = (int)hw.perm_keys.size();
  const int max_layers = 1 + (int)hw.layer_key.size();
  int layers = (int)((L.movable.size() + (u64)P - 1) / (u64)P);
  layers = std::clamp(layers, 1, max_layers);

  build_slots(L, hw, layers);

  if ((int)L.movable.size() > L.slots) {
    throw std::runtime_error("not enough slots for symbols: movable=" + std::to_string(L.movable.size()) +
                             " slots=" + std::to_string(L.slots) +
                             " (add layers, allow more keys, or mark more tokens fixed)");
  }

  // Sort movable symbols by descending frequency.
  std::sort(L.movable.begin(), L.movable.end(), [&](u16 a, u16 b){
    return C.uni.at(a) > C.uni.at(b);
  });

  // Sort slots by ascending slot_cost.
  std::vector<int> slot_order(L.slots);
  for (int i=0;i<L.slots;++i) slot_order[i]=i;
  std::sort(slot_order.begin(), slot_order.end(), [&](int a, int b){
    return L.slot_cost[a] < L.slot_cost[b];
  });

  // Assign top N symbols to best slots, leave remaining slots empty.
  L.slot_code.assign(L.slots, PAUSE_CODE);

  for (int i=0; i<(int)L.movable.size(); ++i) {
    u16 code = L.movable[i];
    int slot = slot_order[i];
    L.code_slot[code] = slot;
    L.slot_code[slot] = code;
    L.code_seq[code] = seq_from_slot(hw, L, slot);
  }

  // Proposal weights for frequency-biased swaps
  L.movable_w.resize(L.movable.size());
  for (std::size_t i=0;i<L.movable.size();++i) {
    u16 code = L.movable[i];
    L.movable_w[i] = (double)C.uni.at(code) + 1.0;
  }
}

// ---------------- Local search: cosine-cycle annealing ----------------

struct AnnealCfg {
  u64 iters = 200000;
  int cycles = 20;          // cosine restarts
  double t_max = 0.05;
  double t_min = 1e-5;
  u64 seed = 1;
};

static inline double cosine_temp(u64 it, u64 iters, int cycles, double tmax, double tmin) {
  if (iters==0) return tmin;
  u64 per = std::max<u64>(1, iters / std::max(1,cycles));
  u64 phase_it = it % per;
  double phase = (double)phase_it / (double)per; // [0,1)
  double cyc = tmin + 0.5*(tmax - tmin)*(1.0 + std::cos(M_PI * phase));
  // slowly shrink amplitude across run (optional)
  double decay = 1.0 - (double)it / (double)iters;
  return tmin + (cyc - tmin) * (0.25 + 0.75*decay);
}

static void swap_symbols(Layout& L, u16 code_a, u16 code_b, const HW& hw) {
  int sa = L.code_slot.at(code_a);
  int sb = L.code_slot.at(code_b);
  if (sa < 0 || sb < 0) return;

  std::swap(L.code_slot[code_a], L.code_slot[code_b]);
  std::swap(L.slot_code[sa], L.slot_code[sb]);

  L.code_seq[code_a] = seq_from_slot(hw, L, L.code_slot.at(code_a));
  L.code_seq[code_b] = seq_from_slot(hw, L, L.code_slot.at(code_b));
}

static void move_symbol_to_empty(Layout& L, u16 code, int empty_slot, const HW& hw) {
  int sslot = L.code_slot.at(code);
  if (sslot < 0) return;
  if (L.slot_code[empty_slot] != PAUSE_CODE) return;

  L.slot_code[sslot] = PAUSE_CODE;
  L.slot_code[empty_slot] = code;
  L.code_slot[code] = empty_slot;
  L.code_seq[code] = seq_from_slot(hw, L, empty_slot);
}

struct Best {
  double score = -std::numeric_limits<double>::infinity();
  Layout layout;
  Agg agg;
};

static Best anneal(const HW& hw, const Corpus& C, const std::vector<Metric>& M, Layout L0, const AnnealCfg& cfg) {
  std::mt19937_64 rng(cfg.seed);

  std::discrete_distribution<std::size_t> pick_sym(L0.movable_w.begin(), L0.movable_w.end());
  std::uniform_real_distribution<double> U(0.0, 1.0);
  std::uniform_int_distribution<int> pick_slot(0, L0.slots-1);

  auto eval = [&](Layout& L)->std::pair<double,Agg>{
    Agg A = accumulate(hw, C, L);
    double s = score_layout(M, A);
    return {s, A};
  };

  auto [cur_score, cur_agg] = eval(L0);

  Best best;
  best.score = cur_score;
  best.layout = L0;
  best.agg = cur_agg;

  Layout L = std::move(L0);

  for (u64 it=0; it<cfg.iters; ++it) {
    double T = cosine_temp(it, cfg.iters, cfg.cycles, cfg.t_max, cfg.t_min);

    // Proposal: mostly swap two high-frequency symbols.
    // Occasionally, attempt moving a high-frequency symbol into a random empty slot.
    bool try_empty = (U(rng) < 0.10);

    u16 symA = L.movable[pick_sym(rng)];
    u16 symB = symA;

    Layout cand = L;

    if (!try_empty) {
      while (symB == symA) symB = L.movable[pick_sym(rng)];
      swap_symbols(cand, symA, symB, hw);
    } else {
      int slot = pick_slot(rng);
      if (cand.slot_code[slot] == PAUSE_CODE) {
        move_symbol_to_empty(cand, symA, slot, hw);
      } else {
        symB = cand.slot_code[slot];
        swap_symbols(cand, symA, symB, hw);
      }
    }

    auto [cand_score, cand_agg] = eval(cand);
    double delta = cand_score - cur_score;

    bool accept = false;
    if (delta >= 0) accept = true;
    else accept = (U(rng) < std::exp(delta / std::max(1e-12, T)));

    if (accept) {
      L = std::move(cand);
      cur_score = cand_score;
      cur_agg = cand_agg;

      if (cur_score > best.score) {
        best.score = cur_score;
        best.layout = L;
        best.agg = cur_agg;
      }
    }
  }

  return best;
}

// ---------------- Pretty print (minimal) ----------------

static void print_metrics(const std::vector<Metric>& M, const Agg& A) {
  for (const auto& m : M) {
    double v = m.val(A);
    std::cout << "  " << m.name << " = " << v << "  (w=" << m.w << ")\n";
  }
}

static std::string key_label(int k) {
  return "K" + std::to_string(k);
}

static void print_layout_mapping(const HW& hw, const Corpus& C, const Layout& L) {
  std::cout << "\n=== BEST LAYOUT MAPPING (symbol -> key sequence) ===\n";
  for (const auto& [code, freq] : C.uni) {
    const Seq& q = L.code_seq.at(code);
    const std::string name = C.code_to_name.contains(code) ? C.code_to_name.at(code) : ("KEYCODE_" + std::to_string(code));

    std::cout << "[" << code << "] " << name
              << " (freq=" << freq << ")"
              << " -> ";

    for (int i=0; i<(int)q.n; ++i) {
      const int k = (int)q.k[i];
      std::cout << key_label(k);

      // annotate finger/hand for readability
      int h = hw.hand[k];
      int f = hw.finger[k];
      std::cout << "(" << (h==(int)HandId::L ? "L" : "R") << "," << f << ")";
      if (i+1<(int)q.n) std::cout << " + ";
    }
    std::cout << "\n";
  }
}

static void print_slot_table(const HW& hw, const Corpus& C, const Layout& L) {
  // Prints each slot (layer, perm-index) with the assigned symbol.
  std::cout << "\n=== SLOT TABLE (layer, perm_index -> symbol) ===\n";
  for (int layer=0; layer<L.layers; ++layer) {
    std::cout << "Layer " << layer << ":\n";
    for (int pi=0; pi<L.P; ++pi) {
      int slot = layer*L.P + pi;
      int key  = L.slot_key[slot];
      u16 code = L.slot_code[slot];

      std::cout << "  slot(" << layer << "," << pi << ")"
                << " key=" << key_label(key)
                << " -> ";
      if (code == PAUSE_CODE) {
        std::cout << "<EMPTY>\n";
      } else {
        const std::string name = C.code_to_name.contains(code) ? C.code_to_name.at(code) : ("KEYCODE_" + std::to_string(code));
        std::cout << name << " (code=" << code << ")\n";
      }
    }
  }
}

// optional: write a simple machine-readable mapping file
static void write_layout_tsv(const HW& hw, const Corpus& C, const Layout& L, const std::string& out_path) {
  std::ofstream out(out_path);
  if (!out) throw std::runtime_error("cannot open output file " + out_path);

  out << "code\tname\tfreq\tseq_len\tk0\tk1\tk2\n";
  for (const auto& [code, freq] : C.uni) {
    const Seq& q = L.code_seq.at(code);
    const std::string name = C.code_to_name.contains(code) ? C.code_to_name.at(code) : ("KEYCODE_" + std::to_string(code));
    out << code << "\t" << name << "\t" << freq
        << "\t" << int(q.n);
    for (int i=0;i<MAX_SEQ;++i) {
      out << "\t" << (i<int(q.n) ? int(q.k[i]) : -1);
    }
    out << "\n";
  }
}

// ---------------- main ----------------

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      std::cerr << "usage: " << argv[0] << " log.bin [iters] [seed]\n";
      return 2;
    }
    std::string path = argv[1];
    u64 iters = (argc>=3 ? std::stoull(argv[2]) : 200000ull);
    u64 seed  = (argc>=4 ? std::stoull(argv[3]) : 1ull);

    std::vector<u16> codes;
    auto code_to_name = read_evtlog(path, codes);
    Corpus C = build_corpus(codes, code_to_name);

    HW hw = make_voyager_like_example();

    Layout L;
    build_initial_layout(L, hw, C);

    auto M = default_metrics();

    AnnealCfg cfg{.iters=iters, .cycles=20, .t_max=0.05, .t_min=1e-5, .seed=seed};

    Best best = anneal(hw, C, M, L, cfg);

    std::cout << "best_score = " << best.score << "\n";
    std::cout << "press_n=" << best.agg.press_n << " big_n=" << best.agg.big_n << " tri_n=" << best.agg.tri_n << "\n";
    std::cout << "metrics:\n";
    print_metrics(M, best.agg);
    print_layout_mapping(hw, C, best.layout);
    print_slot_table(hw, C, best.layout);
    write_layout_tsv(hw, C, best.layout, "best_layout.tsv");

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
