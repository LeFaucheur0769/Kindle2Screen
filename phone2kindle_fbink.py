#!/usr/bin/env python3
"""
phone2kindle_fbink.py - use a jailbroken Kindle (Scribe) as a screen + touchpad
for a rooted Android phone.  Runs on the PHONE (Termux).

    pkg install python python-pillow openssh
    python phone2kindle_fbink.py --calibrate          # once: teach it your touchscreen
    python phone2kindle_fbink.py --phone-width 1240   # stream + touch control

Design
  * ONE persistent root shell  -> a single Magisk prompt, no per-frame `su`.
  * Raw `screencap` (no PNG encode on the phone), converted with Pillow.
  * Capture and drawing run in parallel threads; stale frames are dropped.
  * Only the changed bounding box is sent, as one FBInk call over a
    multiplexed SSH connection (no handshake per frame).
  * Adaptive e-ink: fast waveform (DU) while things move, then a clean GC16
    pass once the screen settles, with a real black flash only when ghosting
    has had time to build up.
  * Touch: reads the Kindle's touchscreen over SSH and injects tap / long
    press / swipe into the phone (`input tap|swipe`), mapped through the exact
    same transform used to draw the frame.
  * --phone-width resizes the phone's display to the Kindle's aspect ratio
    (restored on exit) so nothing is cropped or letterboxed.

If the script is ever killed with -9, restore the phone with:
    su -c "wm size reset"
"""
import argparse
import io
import json
import math
import os
import re
import secrets
import select
import signal
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

from PIL import Image, ImageChops

RES = getattr(Image, "Resampling", Image)
CAL_PATH = os.path.expanduser("~/.phone2kindle.json")
REMOTE_TMP = "/tmp/_p2k.png"
TAP_SLOP = 40          # px on the Kindle: less movement than this is a tap
LONG_PRESS_MS = 600


# --------------------------------------------------------------------------- #
# Root shell on the phone (single su session)
# --------------------------------------------------------------------------- #
class RootShell:
    def __init__(self):
        self.wlock = threading.Lock()   # protects stdin writes (short)
        self.xlock = threading.Lock()   # serialises request/response exchanges
        self.mark = secrets.token_hex(8).encode()
        self.p = None
        self._spawn()

    def _spawn(self):
        if self.p:
            try:
                self.p.kill()
            except OSError:
                pass
        self.p = subprocess.Popen(["su"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, bufsize=0)
        self.fd = self.p.stdout.fileno()

    def _write(self, data: bytes):
        with self.wlock:
            try:
                self.p.stdin.write(data)
            except OSError:
                self._spawn()

    def run(self, cmd: str):
        """Fire and forget. Queued behind whatever the shell is doing."""
        self._write(f"{cmd} </dev/null >/dev/null 2>&1\n".encode())

    def exchange(self, cmd: str, timeout: float = 10.0) -> Optional[bytearray]:
        """Run cmd, return its stdout, or None on timeout/failure."""
        m = self.mark
        with self.xlock:
            self._write(f"{cmd} </dev/null 2>/dev/null; printf %s {m.decode()}\n".encode())
            buf = bytearray()
            end = time.time() + timeout
            while not buf.endswith(m):
                left = end - time.time()
                if left <= 0 or not select.select([self.fd], [], [], left)[0]:
                    self._spawn()
                    return None
                chunk = os.read(self.fd, 1 << 20)
                if not chunk:
                    self._spawn()
                    return None
                buf += chunk
            del buf[-len(m):]
            return buf


class Capturer:
    """screencap -> 8-bit greyscale PIL image."""

    def __init__(self, shell: RootShell):
        self.sh = shell
        self.raw_ok = True

    @staticmethod
    def _decode_raw(buf) -> Optional[Image.Image]:
        w, h, fmt = struct.unpack_from("<III", buf, 0)
        n = w * h * 4
        hdr = len(buf) - n            # 12 (Android <9) or 16 (Android 9+)
        if fmt not in (1, 2) or not (0 < w <= 8192 and 0 < h <= 16384) or hdr not in (12, 16):
            return None
        return Image.frombuffer("RGBA", (w, h), memoryview(buf)[hdr:], "raw", "RGBA", 0, 1).convert("L")

    def grab(self) -> Optional[Image.Image]:
        if self.raw_ok:
            buf = self.sh.exchange("screencap", 10)
            if buf is None or len(buf) < 16:
                return None
            img = self._decode_raw(buf)
            if img is not None:
                return img
            print("[!] raw screencap format unsupported, falling back to PNG capture")
            self.raw_ok = False
        buf = self.sh.exchange("screencap -p", 10)
        if not buf:
            return None
        try:
            return Image.open(io.BytesIO(bytes(buf))).convert("L")
        except Exception:  # noqa: BLE001
            return None


# --------------------------------------------------------------------------- #
# Frame preparation
# --------------------------------------------------------------------------- #
@dataclass
class ViewMap:
    """kindle_px = phone_px * scale + offset"""
    sx: float
    sy: float
    ox: int
    oy: int


def make_lut(invert: bool):
    return [((((255 - v) if invert else v)) >> 4) * 17 for v in range(256)]  # 16 greys


def prepare(raw: Image.Image, W: int, H: int, aspect: str, lut):
    sw, sh = raw.size
    scale = (max if aspect == "crop" else min)(W / sw, H / sh)
    nw, nh = round(sw * scale), round(sh * scale)
    if abs(nw - W) <= 2:
        nw = W
    if abs(nh - H) <= 2:
        nh = H
    img = raw if (nw, nh) == raw.size else raw.resize(
        (nw, nh), RES.LANCZOS if scale < 1 else RES.BICUBIC)
    img = img.point(lut)
    ox, oy = (W - nw) // 2, (H - nh) // 2
    if aspect == "crop":
        img = img.crop((-ox, -oy, -ox + W, -oy + H))
    else:
        canvas = Image.new("L", (W, H), 255)
        canvas.paste(img, (ox, oy))
        img = canvas
    return img, ViewMap(nw / sw, nh / sh, ox, oy)


def snap_box(box, W, H, pad=8):
    """Pad and align to 8px columns - the EPDC likes aligned partial updates."""
    x0, y0, x1, y1 = box
    return (max(0, (x0 - pad) // 8 * 8), max(0, y0 - pad),
            min(W, (x1 + pad + 7) // 8 * 8), min(H, y1 + pad))


def union(a, b):
    if a is None:
        return b
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


# --------------------------------------------------------------------------- #
# Kindle link (OpenSSH with connection multiplexing)
# --------------------------------------------------------------------------- #
class Kindle:
    def __init__(self, a):
        self.a = a
        self.target = f"{a.user}@{a.kindle}"
        self.ssh = [
            "ssh", "-T", "-p", str(a.port), "-i", a.key,
            "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=10",
            "-o", "ControlMaster=auto", "-o", "ControlPath=%d/.p2k-cm-%C",
            "-o", "ControlPersist=120",
        ]
        self.no_wf = False

    def run(self, cmd, data=None, timeout=30):
        return subprocess.run(self.ssh + [self.target, cmd], input=data,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)

    def popen(self, cmd):
        return subprocess.Popen(self.ssh + [self.target, cmd], stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def close(self):
        try:
            subprocess.run(self.ssh + ["-O", "exit", self.target], capture_output=True, timeout=5)
        except Exception:  # noqa: BLE001
            pass

    def draw(self, png: bytes, x: int, y: int, wf: str, flash: bool) -> bool:
        """Upload a PNG and draw it with ONE fbink call."""
        parts = [self.a.fbink, "-q"]
        if flash:
            parts.append("-f")
        if wf and not self.no_wf:
            parts += ["-W", wf]
        parts += ["-g", f"file={REMOTE_TMP},x={x},y={y}"]
        try:
            r = self.run(f"cat > {REMOTE_TMP} && {' '.join(parts)}", png)
        except subprocess.TimeoutExpired:
            print("[-] Kindle timed out")
            return False
        if r.returncode == 0:
            return True
        err = r.stderr.decode(errors="replace").strip()
        if r.returncode != 255 and wf and not self.no_wf:
            print(f"[!] fbink rejected -W {wf} ({err[:80]}); continuing without waveform hints")
            self.no_wf = True
            return self.draw(png, x, y, wf, flash)
        print(f"[-] draw failed ({r.returncode}): {err[:160]}")
        return False


# --------------------------------------------------------------------------- #
# Shared state
# --------------------------------------------------------------------------- #
class Shared:
    def __init__(self):
        self.cond = threading.Condition()
        self.frame = None
        self.vm: Optional[ViewMap] = None
        self.src = (1, 1)
        self.seq = 0
        self.last_activity = time.time()


def capture_thread(cap: Capturer, st: Shared, a, stop: threading.Event):
    lut = make_lut(a.invert)
    while not stop.is_set():
        t0 = time.time()
        raw = cap.grab()
        if raw is None:
            stop.wait(1.0)
            continue
        img, vm = prepare(raw, a.width, a.height, a.aspect, lut)
        with st.cond:
            st.frame, st.vm, st.src = img, vm, raw.size
            st.seq += 1
            st.cond.notify_all()
        idle = time.time() - st.last_activity > a.idle_after
        period = 1.0 / (a.idle_fps if idle else a.fps)
        stop.wait(max(0.0, period - (time.time() - t0)))


def encode(frame: Image.Image, box) -> bytes:
    buf = io.BytesIO()
    frame.crop(box).save(buf, "PNG", compress_level=1)
    return buf.getvalue()


def stream_loop(a, kindle: Kindle, st: Shared, stop: threading.Event):
    W, H = a.width, a.height
    full = (0, 0, W, H)
    screen_area = W * H
    thr_lut = [255 if v > a.threshold else 0 for v in range(256)]
    displayed = None
    seq_done = 0
    dirty = None
    last_change = 0.0
    area_since_flash = 0
    drawn = 0
    t_report = time.time()

    while not stop.is_set():
        with st.cond:
            st.cond.wait_for(lambda: st.seq != seq_done or stop.is_set(), timeout=0.25)
            frame, seq = st.frame, st.seq
        if frame is None:
            continue
        now = time.time()
        fresh = seq != seq_done
        seq_done = seq

        if displayed is None:                       # first frame: clean full flash
            if kindle.draw(encode(frame, full), 0, 0, a.clean_wf, True):
                displayed = frame.copy()
                last_change = now
            else:
                stop.wait(2.0)
            continue

        box = ImageChops.difference(displayed, frame).point(thr_lut).getbbox() if fresh else None

        if box:
            box = snap_box(box, W, H)
            area = (box[2] - box[0]) * (box[3] - box[1])
            big = area > a.big_ratio * screen_area
            if big:
                box = full
            wf = a.big_wf if big else a.active_wf
            if kindle.draw(encode(frame, box), box[0], box[1], wf, False):
                displayed.paste(frame.crop(box), box[:2])
                dirty = union(dirty, box)
                last_change = now
                st.last_activity = now
                area_since_flash += area
                drawn += 1
                if a.verbose:
                    print(f"[draw] {box} {100 * area // screen_area}% wf={wf}")
            else:
                stop.wait(1.0)
        elif dirty and now - last_change >= a.settle:
            flash = area_since_flash > a.flash_after * screen_area
            if kindle.draw(encode(frame, dirty), dirty[0], dirty[1], a.clean_wf, flash):
                displayed.paste(frame.crop(dirty), dirty[:2])
                if a.verbose:
                    print(f"[clean] {dirty} {'FLASH' if flash else ''}")
                if flash:
                    area_since_flash = 0
                dirty = None
            else:
                stop.wait(1.0)

        if not a.verbose and now - t_report >= 10:
            print(f"[*] {drawn / (now - t_report):.1f} updates/s")
            drawn, t_report = 0, now


# --------------------------------------------------------------------------- #
# Touch: Kindle evdev -> phone `input`
# --------------------------------------------------------------------------- #
def parse_bitmap(s: str) -> int:
    words = s.split()
    if not words:
        return 0
    bits = len(words[1]) * 4 if len(words) > 1 else 64
    v = 0
    for w in words:
        v = (v << bits) | int(w, 16)
    return v


def find_touch_dev(text: str):
    """Pick the multitouch (finger) input device from /proc/bus/input/devices."""
    best = None
    for block in re.split(r"\n\s*\n", text):
        name = re.search(r'Name="([^"]*)"', block)
        ev = re.search(r"\b(event\d+)\b", block)
        absm = re.search(r"B: ABS=(.*)", block)
        if not (name and ev):
            continue
        n = name.group(1)
        has_mt = bool(absm and (parse_bitmap(absm.group(1)) >> 0x35) & 1)
        pen = re.search(r"pen|wacom|stylus|digitizer|emr", n, re.I)
        score = (2 if has_mt else 0) + (1 if re.search(r"touch|zforce|cyttsp|goodix|elan|ft5", n, re.I) else 0)
        if pen:
            score -= 3
        if score > 0 and (best is None or score > best[0]):
            best = (score, f"/dev/input/{ev.group(1)}", n)
    return best[1:] if best else None


def detect_esz(data: bytes) -> Optional[int]:
    for esz, fmt in ((16, "<llHHi"), (24, "<qqHHi")):
        n = len(data) // esz
        if n < 6:
            continue
        s = struct.Struct(fmt)
        evs = [s.unpack_from(data, i * esz) for i in range(min(n, 40))]
        if all(e[2] in (0, 1, 3, 4) for e in evs) and any(e[2] == 0 and e[3] == 0 for e in evs):
            return esz
    return None


def read_events(stream, esz):
    s = struct.Struct("<llHHi" if esz == 16 else "<qqHHi")
    while True:
        b = stream.read(esz)
        if len(b) < esz:
            return
        yield s.unpack(b)[2:]


class TouchTracker:
    """Turns raw evdev events into ('down'|'move'|'up', rawx, rawy) for the first finger."""

    def __init__(self):
        self.slot = 0
        self.track_seen = False
        self.want = False
        self.down = False
        self.x = self.y = 0

    def feed(self, t, c, v):
        if t == 3:
            if c == 0x2F:
                self.slot = v
            elif self.slot == 0:
                if c == 0x39:
                    self.track_seen = True
                    self.want = v != -1
                elif c in (0x35, 0x00):
                    self.x = v
                elif c in (0x36, 0x01):
                    self.y = v
        elif t == 1 and c == 0x14A and not self.track_seen:
            self.want = bool(v)
        elif t == 0 and c == 0:
            if self.want and not self.down:
                self.down = True
                return ("down", self.x, self.y)
            if self.want:
                return ("move", self.x, self.y)
            if self.down:
                self.down = False
                return ("up", self.x, self.y)
        return None


def raw_to_screen(cal, rx, ry, W, H):
    r = {"x": rx, "y": ry}

    def norm(c):
        d = c["v1"] - c["v0"]
        return 0.0 if d == 0 else min(max((r[c["axis"]] - c["v0"]) / d, 0.0), 1.0)

    return norm(cal["x"]) * (W - 1), norm(cal["y"]) * (H - 1)


def inject(sh: RootShell, st: Shared, g):
    x0, y0, x1, y1, ms = g
    vm, (pw, ph) = st.vm, st.src
    if vm is None:
        return

    def conv(x, y, clamp):
        px, py = (x - vm.ox) / vm.sx, (y - vm.oy) / vm.sy
        if not clamp and not (0 <= px < pw and 0 <= py < ph):
            return None
        return int(min(max(px, 0), pw - 1)), int(min(max(py, 0), ph - 1))

    if math.hypot(x1 - x0, y1 - y0) < TAP_SLOP:
        p = conv(x0, y0, False)
        if p is None:
            return
        if ms >= LONG_PRESS_MS:
            sh.run("input swipe %d %d %d %d %d" % (p + p + (max(ms, 700),)))
        else:
            sh.run("input tap %d %d" % p)
    else:
        p, q = conv(x0, y0, True), conv(x1, y1, True)
        sh.run("input swipe %d %d %d %d %d" % (p + q + (min(max(ms, 80), 800),)))
    st.last_activity = time.time()


def touch_thread(a, kindle: Kindle, cal, st: Shared, sh: RootShell, stop: threading.Event):
    while not stop.is_set():
        proc = kindle.popen(f"cat {cal['dev']}")
        tr = TouchTracker()
        start = last = None
        try:
            for t, c, v in read_events(proc.stdout, cal["esz"]):
                if stop.is_set():
                    break
                r = tr.feed(t, c, v)
                if not r:
                    continue
                kind, rx, ry = r
                sx, sy = raw_to_screen(cal, rx, ry, a.width, a.height)
                if kind == "down":
                    start, last = (sx, sy, time.time()), (sx, sy)
                    st.last_activity = time.time()
                elif kind == "move":
                    last = (sx, sy)
                elif kind == "up" and start:
                    ex, ey = (sx, sy) if (sx or sy) else last
                    inject(sh, st, (start[0], start[1], ex, ey, int((time.time() - start[2]) * 1000)))
                    start = None
        finally:
            proc.kill()
        if not stop.is_set():
            print("[!] touch stream ended, retrying in 2s")
            stop.wait(2.0)


def calibrate(a, kindle: Kindle):
    r = kindle.run("cat /proc/bus/input/devices", timeout=15)
    found = find_touch_dev(r.stdout.decode(errors="replace"))
    dev = a.touch_dev or (found[0] if found else None)
    if not dev:
        sys.exit("Could not find the touchscreen. Pass --touch-dev /dev/input/eventN "
                 "(see: ssh kindle cat /proc/bus/input/devices)")
    print(f"[*] Touch device: {dev}" + (f" ({found[1]})" if found and not a.touch_dev else ""))

    def say(msg):
        print("    >>>", msg)
        kindle.run(f'{a.fbink} -q -m -M "{msg}"', timeout=10)

    # 1) event size (32 vs 64-bit timeval)
    print("[1/2] Touch the Kindle screen and drag around for ~3 seconds...")
    p = kindle.popen(f"cat {dev}")
    data = b""
    end = time.time() + 3.5
    while time.time() < end:
        if select.select([p.stdout], [], [], 0.2)[0]:
            chunk = os.read(p.stdout.fileno(), 4096)
            if not chunk:
                break
            data += chunk
    p.kill()
    esz = detect_esz(data)
    if not esz:
        sys.exit("Could not decode touch events (got %d bytes). Try again and keep dragging." % len(data))
    print(f"      event size: {esz} bytes")

    # 2) three corners -> axis mapping, flips and range
    pts = {}
    p = kindle.popen(f"cat {dev}")
    tr = TouchTracker()
    for key, label in (("tl", "TOP-LEFT"), ("tr", "TOP-RIGHT"), ("bl", "BOTTOM-LEFT")):
        say(f"Tap the very {label} corner")
        for t, c, v in read_events(p.stdout, esz):
            r = tr.feed(t, c, v)
            if r and r[0] == "up":
                pts[key] = (r[1], r[2])
                print(f"      {label}: raw {pts[key]}")
                break
        time.sleep(0.8)
    p.kill()
    tl, trr, bl = pts["tl"], pts["tr"], pts["bl"]
    ax = "x" if abs(trr[0] - tl[0]) >= abs(trr[1] - tl[1]) else "y"
    ay = "x" if abs(bl[0] - tl[0]) > abs(bl[1] - tl[1]) else "y"
    if ax == ay:
        sys.exit("Could not tell the axes apart - tap the real corners and run --calibrate again.")
    idx = {"x": 0, "y": 1}
    cal = {"dev": dev, "esz": esz,
           "x": {"axis": ax, "v0": tl[idx[ax]], "v1": trr[idx[ax]]},
           "y": {"axis": ay, "v0": tl[idx[ay]], "v1": bl[idx[ay]]}}
    with open(CAL_PATH, "w") as f:
        json.dump(cal, f, indent=2)
    say("Calibration saved")
    print(f"[+] Saved {CAL_PATH}:\n{json.dumps(cal, indent=2)}")


# --------------------------------------------------------------------------- #
# Phone display size
# --------------------------------------------------------------------------- #
def set_phone_size(sh: RootShell, a) -> bool:
    out = (sh.exchange("wm size") or b"").decode(errors="replace")
    m = re.search(r"Physical size:\s*(\d+)x(\d+)", out)
    if not m:
        print("[!] could not read `wm size`; leaving the phone resolution alone")
        return False
    pw = int(m.group(1))
    w = min(a.phone_width, pw)
    h = round(w * a.height / a.width / 2) * 2
    print(f"[*] Phone display -> {w}x{h} (restored on exit)")
    sh.run(f"wm size {w}x{h}")
    time.sleep(1.5)
    return True


# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("Kindle connection")
    g.add_argument("--kindle", default="192.168.1.14", help="Kindle IP")
    g.add_argument("--port", type=int, default=2022)
    g.add_argument("--user", default="root")
    g.add_argument("--key", default=os.path.expanduser("~/.ssh/kindle_key"))
    g.add_argument("--fbink", default="/mnt/us/libkh/bin/fbink")
    g = p.add_argument_group("Picture")
    g.add_argument("--width", type=int, default=1860, help="Kindle visible width (Scribe 1860)")
    g.add_argument("--height", type=int, default=2480, help="Kindle visible height (Scribe 2480)")
    g.add_argument("--aspect", choices=["fit", "crop"], default="fit")
    g.add_argument("--phone-width", type=int, default=0,
                   help="resize the phone display to the Kindle aspect at this width "
                        "(e.g. 1240; bigger = sharper but slower). 0 = leave alone")
    g.add_argument("--invert", action="store_true", help="invert colours (dark-mode phone)")
    g = p.add_argument_group("Speed / e-ink")
    g.add_argument("--fps", type=float, default=8.0, help="max capture rate while active")
    g.add_argument("--idle-fps", type=float, default=1.0, help="capture rate when nothing happens")
    g.add_argument("--idle-after", type=float, default=3.0)
    g.add_argument("--threshold", type=int, default=40, help="pixel change (0-255) that counts")
    g.add_argument("--active-wf", default="DU", help="waveform for small fast updates")
    g.add_argument("--big-wf", default="GL16", help="waveform for large changes")
    g.add_argument("--clean-wf", default="GC16", help="waveform for the settle/cleanup pass")
    g.add_argument("--big-ratio", type=float, default=0.55, help="area fraction counted as large")
    g.add_argument("--settle", type=float, default=1.0, help="seconds of calm before cleanup pass")
    g.add_argument("--flash-after", type=float, default=4.0,
                   help="flash during cleanup after this many screens of redraws (ghosting)")
    g = p.add_argument_group("Input")
    g.add_argument("--calibrate", action="store_true", help="calibrate the Kindle touchscreen")
    g.add_argument("--no-touch", action="store_true", help="display only")
    g.add_argument("--touch-dev", help="e.g. /dev/input/event1 (auto-detected otherwise)")
    g = p.add_argument_group("Misc")
    g.add_argument("--freeze-ui", action="store_true",
                   help="SIGSTOP the Kindle UI (awesome) while streaming, resume on exit")
    g.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main():
    a = parse_args()
    kindle = Kindle(a)

    r = None
    try:
        r = kindle.run("echo ok", timeout=20)
    except subprocess.TimeoutExpired:
        pass
    if not r or r.returncode != 0:
        sys.exit(f"Cannot SSH to {kindle.target}:{a.port} with key {a.key}.\n"
                 f"{r.stderr.decode(errors='replace') if r else 'timeout'}")
    if kindle.run(f"test -x {a.fbink}").returncode != 0:
        sys.exit(f"FBInk not found at {a.fbink} on the Kindle (use --fbink).")

    if a.calibrate:
        calibrate(a, kindle)
        kindle.close()
        return

    cal = None
    if not a.no_touch:
        try:
            with open(CAL_PATH) as f:
                cal = json.load(f)
        except OSError:
            print("[!] No touch calibration yet - run with --calibrate. Streaming display-only.")

    stop = threading.Event()
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, lambda *_: stop.set())

    sh = RootShell()                       # <- the one Magisk prompt
    resized = False
    try:
        if a.phone_width:
            resized = set_phone_size(sh, a)

        cap = Capturer(sh)
        if cap.grab() is None:
            sys.exit("screencap failed - is root granted to Termux in Magisk?")

        kindle.run("lipc-set-prop com.lab126.powerd preventScreenSaver 1")
        if a.freeze_ui:
            kindle.run("killall -STOP awesome")

        st = Shared()
        threading.Thread(target=capture_thread, args=(cap, st, a, stop), daemon=True).start()
        if cal:
            threading.Thread(target=touch_thread, args=(a, kindle, cal, st, sh, stop),
                             daemon=True).start()
        print(f"[*] Streaming to {kindle.target} at {a.width}x{a.height}"
              f"{' with touch' if cal else ''}. Ctrl+C to stop.")
        stream_loop(a, kindle, st, stop)
    finally:
        stop.set()
        print("\n[*] Cleaning up...")
        try:
            if resized:
                sh.exchange("wm size reset", 10)
            kindle.run("lipc-set-prop com.lab126.powerd preventScreenSaver 0", timeout=10)
            if a.freeze_ui:
                kindle.run("killall -CONT awesome", timeout=10)
        except Exception:  # noqa: BLE001
            pass
        kindle.close()


if __name__ == "__main__":
    main()
