#!/usr/bin/env python3
import re
import struct
import subprocess
import sys
import signal
from typing import Optional, Tuple, Dict, BinaryIO

# evtest event lines
EV_RE = re.compile(
    r"^Event:\s+time\s+([0-9]+\.[0-9]+),\s+type\s+(\d+)\s+\(([^)]+)\),\s+code\s+(\d+)\s+\(([^)]+)\),\s+value\s+(-?\d+)\s*$"
)
# Supported-events keycode lines:
# "    Event code 115 (KEY_VOLUMEUP)"
MAP_RE = re.compile(r"^\s*Event code\s+(\d+)\s+\(([^)]+)\)\s*$")

# Record format:
#   float64 timestamp_seconds
#   uint16  keycode_number (linux input keycode)
#   int8    value (press=1, release=0, repeat=2)
#   uint8   reserved
#REC_FMT = struct.Struct("<dHbB")
REC_FMT = struct.Struct("<H")   # uint16 keycode only; 0 is reserved for PAUSE

# Header:
#   8s magic
#   uint16 version
#   uint16 record_size
#   uint32 map_entries
HDR_FMT = struct.Struct("<8sHHI")

MAGIC = b"EVTLOG1\0"              # 8 bytes
VERSION = 1
VALUE_TO_ACTION = {0: "release", 1: "press", 2: "repeat"}

def parse_evtest_event_line(line: str) -> Optional[Tuple[float, str, int, str, int]]:
    m = EV_RE.match(line)
    if not m:
        return None
    t = float(m.group(1))
    type_name = m.group(3)
    code_num = int(m.group(4))
    code_name = m.group(5)
    value = int(m.group(6))
    return (t, type_name, code_num, code_name, value)

def write_header(f: BinaryIO, code_to_name: Dict[int, str]) -> None:
    items = sorted(code_to_name.items())
    f.write(HDR_FMT.pack(MAGIC, VERSION, REC_FMT.size, len(items)))
    for code, name in items:
        name_b = name.encode("ascii", errors="replace")
        if len(name_b) > 255:
            name_b = name_b[:255]
        f.write(struct.pack("<HB", int(code), len(name_b)))
        f.write(name_b)

def read_header(f: BinaryIO) -> Tuple[int, Dict[int, str]]:
    hdr = f.read(HDR_FMT.size)
    if len(hdr) != HDR_FMT.size:
        raise ValueError("file too short (missing header)")
    magic, ver, rec_size, m = HDR_FMT.unpack(hdr)
    if magic != MAGIC:
        raise ValueError("bad magic (not an EVTLOG file)")
    if ver != VERSION:
        raise ValueError(f"unsupported version {ver}")
    if rec_size != REC_FMT.size:
        raise ValueError(f"record size mismatch (file {rec_size}, expected {REC_FMT.size})")

    code_to_name: Dict[int, str] = {}
    for _ in range(m):
        entry = f.read(3)
        if len(entry) != 3:
            raise ValueError("truncated mapping table")
        code, n = struct.unpack("<HB", entry)
        name_b = f.read(n)
        if len(name_b) != n:
            raise ValueError("truncated mapping table (name)")
        code_to_name[code] = name_b.decode("ascii", errors="replace")
    return rec_size, code_to_name

def cmd_record(dev: str, out_path: str) -> int:
    cmd = ["stdbuf", "-oL", "evtest", dev]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def shutdown(*_args):
        try:
            proc.terminate()
        except Exception:
            pass

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # State for parsing supported-events section
    in_ev_key_section = False
    mapping_done = False
    code_to_name: Dict[int, str] = {}

    try:
        with open(out_path, "wb", buffering=0) as f:
            assert proc.stdout is not None

            last_t: Optional[float] = None
            PAUSE_CODE = 0
            PAUSE_THRESHOLD = 3.0
            for raw in proc.stdout:
                line = raw.rstrip("\n")

                # 1) Build mapping from the header printed by evtest
                #    We only need to parse until "Testing ..." starts.
                if not mapping_done:
                    if line.strip() == "Event type 1 (EV_KEY)":
                        in_ev_key_section = True
                        continue

                    # Leaving EV_KEY section once another "Event type ..." occurs
                    if line.strip().startswith("Event type ") and line.strip() != "Event type 1 (EV_KEY)":
                        in_ev_key_section = False

                    if in_ev_key_section:
                        mm = MAP_RE.match(line)
                        if mm:
                            code = int(mm.group(1))
                            name = mm.group(2)
                            code_to_name[code] = name
                            continue

                    if line.strip().startswith("Testing ..."):
                        # write header once, then switch to event logging
                        write_header(f, code_to_name)
                        mapping_done = True
                        continue

                    # Ignore all other preamble lines
                    continue

                # 2) After header: parse and log only whitelisted events
                parsed = parse_evtest_event_line(line.strip())
                if not parsed:
                    continue
                t, type_name, code_num, code_name, value = parsed

                if type_name != "EV_KEY":
                    continue
                if VALUE_TO_ACTION[value] != "press": #meaning "value != 1"
                        continue
                if value < -128 or value > 127:
                    continue

                if last_t is not None and (t - last_t) > PAUSE_THRESHOLD:
                    f.write(REC_FMT.pack(PAUSE_CODE))
                f.write(REC_FMT.pack(code_num))

                last_t = t
                #f.write(REC_FMT.pack(t, code_num, value, 0))

    finally:
        shutdown()
        try:
            proc.wait(timeout=1.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    return 0

def cmd_read(path: str) -> int:
    with open(path, "rb") as f:
        _rec_size, code_to_name = read_header(f)

        i = 0
        while True:
            chunk = f.read(REC_FMT.size)
            if not chunk:
                break
            if len(chunk) != REC_FMT.size:
                print(f"trailing {len(chunk)} bytes (corrupt/incomplete record)", file=sys.stderr)
                break

            #t, code_num, value, _ = REC_FMT.unpack(chunk)<
            (code_num,) = REC_FMT.unpack(chunk)
            if code_num == 0:
                print(f"{i:08d}  <PAUSE>")
            else:
                name = code_to_name.get(code_num, f"KEYCODE_{code_num}")
#                action = VALUE_TO_ACTION.get(value, f"value={value}")
#                print(f"{i:08d}  t={t:.6f}  {name:<24}  {action}")
                print(f"{i:08d}  {name}")
            i += 1

    return 0

def usage() -> int:
    print(
        "usage:\n"
        f"  {sys.argv[0]} record /dev/input/eventX OUTPUT.bin\n"
        f"  {sys.argv[0]} read OUTPUT.bin\n",
        file=sys.stderr,
    )
    return 2

def main() -> int:
    if len(sys.argv) < 2:
        return usage()
    mode = sys.argv[1]
    if mode == "record":
        if len(sys.argv) != 4:
            return usage()
        return cmd_record(sys.argv[2], sys.argv[3])
    if mode == "read":
        if len(sys.argv) != 3:
            return usage()
        return cmd_read(sys.argv[2])
    return usage()

if __name__ == "__main__":
    raise SystemExit(main())
