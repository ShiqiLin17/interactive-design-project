"""Christmas Fog Window — fog up the window with your breath, wipe it with your hand, decorate it with stickers.

Your face:     make an "O" with your lips and blow at the window to fog it up (smiling won't)
Sticker hand:  touch your thumb to a finger to pop out a sticker
               (each fingertip shows which one)   N = next page of stickers
Other hand:    raise only your index finger to write in the fog (index + middle up = pen up); thumb + middle grabs a sticker (closer = bigger);
               thumb + ring switches your face filter
Sticker hand:  close it into a fist and wipe with the pinky side

Keys: V switch camera · E face filter · S swap hands · R fog all · W wipe all · H help · Z undo · C clear stickers · P screenshot · F fullscreen · Q quit
"""
import math
import os
import random
import ssl
import sys
import time
import urllib.request

import cv2
import numpy as np

from stickers import build_all, make_clear
from window import RealWindow
from droplets import Droplets
from facefx import FaceFX
import text

# ---------------------------------------------------------------- settings
CAMERA_INDEX = None          # None = remember your last choice (press V to switch), or prefer the built-in laptop camera
WIDTH, HEIGHT = 1280, 720
FOG_REGROW_SECONDS = 0       # 0 = fog only comes from your breath; e.g. 25 = slowly fogs back up
BREATH_FOG_SECONDS = 1.8     # how long you blow to fog up the whole mirror
MOUTH_OPEN = 0.045           # how open your mouth must be to count as blowing (raise if too sensitive)
O_ROUNDNESS = 0.45           # an "O" is tall for its width (smiles are wide and flat)
STICKER_SIZE = 150           # on-screen sticker size in pixels
PINCH_ON = 0.32              # fingertip distance (relative to hand size) to start a pinch
PINCH_OFF = 0.48             # ...and to release it
SPAWN_COOLDOWN = 0.45        # seconds between stickers
STICKER_OPACITY = 0.8         # 1.0 = solid, lower = more see-through (clear window-cling look)
MINI_SIZE = 46               # size of the little stickers shown on your fingertips
DRAW_WIDTH = 22              # thickness of the line when you draw in the fog
ZOOM_MIN, ZOOM_MAX = 0.4, 3.5
HELP_SECONDS = 12            # the help text fades away after this long (press H to show it again)
WIPE_PAUSE_AFTER_DRAW = 1.0  # seconds after drawing before a fist wipes again

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                            "hand_landmarker/float16/latest/hand_landmarker.task",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                            "face_landmarker/float16/latest/face_landmarker.task",
}

TIPS = [8, 12, 16, 20]          # index, middle, ring, pinky
MY_STICKERS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "my_stickers")
THUMB_TIP = 4


# ---------------------------------------------------------------- helpers
def load_my_stickers():
    """Your own sticker images (transparent PNGs) from my_stickers/, in filename order. None if there are none."""
    if not os.path.isdir(MY_STICKERS):
        return None
    imgs = {}
    for fn in sorted(os.listdir(MY_STICKERS)):
        if fn.lower().endswith(".png"):
            img = cv2.imread(os.path.join(MY_STICKERS, fn), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            if img.ndim == 2 or img.shape[2] == 3:      # no transparency: add a solid alpha channel
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA if img.ndim == 2 else cv2.COLOR_BGR2BGRA)
            s = 320 / max(img.shape[:2])
            if s < 1:
                img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            imgs[os.path.splitext(fn)[0]] = img
    return imgs or None



def download_model(name):
    path = os.path.join(HERE, name)
    url = MODELS[name]
    if os.path.exists(path):
        return path
    print(f"Downloading {name} (first run only)...")
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(url, context=ctx) as r, open(path, "wb") as f:
            f.write(r.read())
    except Exception as e:
        if os.path.exists(path):
            os.remove(path)
        print(f"\nCouldn't download the model ({e}).\nRun this in this folder, then try again:\n"
              f"  curl -L -o {name} {url}\n")
        sys.exit(1)
    return path


def ease_out_back(t):
    c1 = 1.9
    c3 = c1 + 1
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2


def rotate_scale(img, angle, scale):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    bw, bh = int(h * sin + w * cos) + 2, int(h * cos + w * sin) + 2
    M[0, 2] += bw / 2 - w / 2
    M[1, 2] += bh / 2 - h / 2
    return cv2.warpAffine(img, M, (bw, bh), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def paste(frame, rgba, cx, cy, shadow=0.0, lift=0, opacity=1.0):
    """Alpha-blend an RGBA image onto frame, centered at (cx, cy)."""
    h, w = rgba.shape[:2]
    H, W = frame.shape[:2]
    alpha = rgba[..., 3].astype(np.float32) / 255 * opacity

    def blend(color, a, ox, oy):
        x0, y0 = int(cx - w / 2 + ox), int(cy - h / 2 + oy)
        x1, y1 = max(x0, 0), max(y0, 0)
        x2, y2 = min(x0 + w, W), min(y0 + h, H)
        if x1 >= x2 or y1 >= y2:
            return
        sa = a[y1 - y0:y2 - y0, x1 - x0:x2 - x0, None]
        roi = frame[y1:y2, x1:x2].astype(np.float32)
        src = color if np.isscalar(color) else color[y1 - y0:y2 - y0, x1 - x0:x2 - x0]
        frame[y1:y2, x1:x2] = (roi * (1 - sa) + src * sa).astype(np.uint8)

    if shadow > 0:
        soft = cv2.GaussianBlur(alpha, (0, 0), 6 + lift / 2) * shadow
        blend(0.0, soft, 5 + lift, 8 + lift * 1.5)
    blend(rgba[..., :3].astype(np.float32), alpha, 0, 0)


def draw_badge(out, pos, icon, scale=1.0, active=False):
    """Minimal fingertip icon: a thin ring with a tiny pen (write) or 4-way arrow (move)."""
    r = int(14 * scale)
    c = (int(pos[0]), int(pos[1]))
    ink = (170, 225, 255) if active else (255, 255, 255)
    layer = out.copy()
    cv2.circle(layer, c, r, (30, 25, 20), -1, cv2.LINE_AA)
    cv2.addWeighted(layer, 0.25, out, 0.75, 0, out)          # faint smoky disc behind the icon
    cv2.circle(out, c, r, ink, 1, cv2.LINE_AA)
    k = r * 0.5
    if icon == "move":
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            tip = (int(c[0] + dx * k), int(c[1] + dy * k))
            cv2.line(out, c, tip, ink, 1, cv2.LINE_AA)
            px, py = -dy, dx
            for s in (1, -1):
                cv2.line(out, tip, (int(tip[0] - dx * k * 0.4 + px * s * k * 0.3),
                                    int(tip[1] - dy * k * 0.4 + py * s * k * 0.3)), ink, 1, cv2.LINE_AA)
    elif icon == "filter":
        for ang in range(4):
            t = ang * math.pi / 2
            p = (int(c[0] + math.cos(t) * k), int(c[1] + math.sin(t) * k))
            cv2.line(out, c, p, ink, 1, cv2.LINE_AA)
        cv2.circle(out, c, max(1, int(k * 0.3)), ink, -1, cv2.LINE_AA)
    else:
        a = (int(c[0] - k * 0.8), int(c[1] + k * 0.8))
        b = (int(c[0] + k * 0.8), int(c[1] - k * 0.8))
        cv2.line(out, a, b, ink, 2, cv2.LINE_AA)
        cv2.circle(out, a, 1, ink, -1, cv2.LINE_AA)


# ---------------------------------------------------------------- fog
class Fog:
    def __init__(self, w, h, scale=2):
        self.w, self.h, self.s = w, h, scale
        self.mask = np.zeros((h // scale, w // scale), np.float32)  # starts clear
        ys, xs = np.mgrid[0:h // scale, 0:w // scale]
        self.grid = (xs.astype(np.float32), ys.astype(np.float32))
        rng = np.random.default_rng(7)
        # fine mist: countless tiny beads of water with a dark edge and a bright point
        mist = np.zeros((h, w), np.float32)
        for _ in range(int(w * h / 60)):
            x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
            r = int(rng.choice([1, 1, 1, 1, 2]))
            cv2.circle(mist, (x, y), r, -0.5, -1, cv2.LINE_AA)
            cv2.circle(mist, (x, y - (r > 1)), max(1, r - 1), 0.9, -1, cv2.LINE_AA)
        mist = cv2.GaussianBlur(mist, (0, 0), 0.5)
        haze = cv2.resize(rng.random((h // 50, w // 50)).astype(np.float32), (w, h),
                          interpolation=cv2.INTER_CUBIC)
        tex = mist * 3 + (haze - 0.5) * 14
        self.tex_pos = np.repeat(np.clip(tex, 0, 255)[..., None], 3, 2).astype(np.uint8)
        self.tex_neg = np.repeat(np.clip(-tex, 0, 255)[..., None], 3, 2).astype(np.uint8)
        self.m_full = np.zeros((h, w), np.float32)

    def wipe(self, p0, p1, radius):
        s = self.s
        a = (int(p0[0] / s), int(p0[1] / s))
        b = (int(p1[0] / s), int(p1[1] / s))
        r = max(1, int(round(radius / s)))
        cv2.line(self.mask, a, b, 0.0, 2 * r, cv2.LINE_AA)
        cv2.circle(self.mask, b, r, 0.0, -1, cv2.LINE_AA)

    def wipe_sweep(self, edge0, edge1, radius):
        """Wipe the area swept by a line segment moving from edge0 to edge1 (like the side of a fist)."""
        s = self.s
        pts = np.array([edge0[0], edge0[1], edge1[1], edge1[0]], np.float32) / s
        cv2.fillPoly(self.mask, [pts.astype(np.int32)], 0.0, cv2.LINE_AA)
        for e in (edge0, edge1):
            self.wipe(e[0], e[1], radius)

    def breathe(self, x, y, radius, amount):
        """Add fog in a soft cloud around (x, y)."""
        s = self.s
        d2 = (self.grid[0] - x / s) ** 2 + (self.grid[1] - y / s) ** 2
        blob = np.exp(-d2 / (2 * (radius / s) ** 2))
        self.mask += blob * amount
        np.clip(self.mask, 0, 1, out=self.mask)

    def value_at(self, x, y):
        xi, yi = int(x / self.s), int(y / self.s)
        if 0 <= yi < self.mask.shape[0] and 0 <= xi < self.mask.shape[1]:
            return float(self.mask[yi, xi])
        return 0.0

    def coverage(self):
        return float(self.mask.mean())

    def regrow(self, dt):
        if FOG_REGROW_SECONDS > 0:
            self.mask += dt / FOG_REGROW_SECONDS
            np.clip(self.mask, 0, 1, out=self.mask)

    def render(self, frame, tint):
        """tint: function(shape) -> fog color at low resolution."""
        h, w = frame.shape[:2]
        sh = (h // 16, w // 16)
        small = cv2.resize(frame, (sh[1], sh[0]), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 1.6).astype(np.float32) * 0.42 + tint(sh) * 0.58
        fogged = cv2.resize(cv2.convertScaleAbs(small), (w, h), interpolation=cv2.INTER_CUBIC)
        fogged = cv2.subtract(cv2.add(fogged, self.tex_pos), self.tex_neg)
        m = cv2.GaussianBlur(self.mask, (0, 0), 1.2)
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        rim = (m * (1 - m) * 4) ** 2           # fog piles up a little at the edge of a wipe
        m = np.clip(0.03 + 0.97 * m + rim * 0.2, 0, 1)
        self.m_full = m
        return cv2.blendLinear(frame, fogged, 1 - m, m)


class BreathPuffs:
    """Little misty clouds that drift out of your mouth while you blow."""
    def __init__(self):
        self.p = []

    def emit(self, x, y, size):
        if random.random() < 0.5:
            return
        for _ in range(1):
            a = random.uniform(0, math.tau)
            v = random.uniform(40, 120)
            self.p.append([x, y, math.cos(a) * v, math.sin(a) * v, 0.9, size * random.uniform(0.15, 0.3)])

    def update_draw(self, frame, dt):
        if not self.p:
            return
        layer = np.zeros_like(frame)
        alive = []
        for s in self.p:
            s[0] += s[2] * dt
            s[1] += s[3] * dt
            s[4] -= dt
            s[5] += 60 * dt
            if s[4] > 0:
                v = int(22 * s[4])
                cv2.circle(layer, (int(s[0]), int(s[1])), int(s[5]), (v, v, v), -1, cv2.LINE_AA)
                alive.append(s)
        self.p = alive
        layer = cv2.GaussianBlur(layer, (0, 0), 12)
        cv2.add(frame, layer, frame)


class Sparkles:
    def __init__(self):
        self.p = []

    def burst(self, x, y, n=18):
        for _ in range(n):
            a = random.uniform(0, 6.28)
            v = random.uniform(120, 340)
            col = random.choice([(120, 215, 250), (200, 240, 255), (255, 255, 255)])
            self.p.append([x, y, math.cos(a) * v, math.sin(a) * v, 0.6, col])

    def update_draw(self, frame, dt):
        alive = []
        for s in self.p:
            s[0] += s[2] * dt
            s[1] += s[3] * dt
            s[3] += 300 * dt
            s[4] -= dt
            if s[4] > 0:
                r = max(1, int(4 * s[4] / 0.6))
                x, y = int(s[0]), int(s[1])
                cv2.line(frame, (x - r, y), (x + r, y), s[5], 2, cv2.LINE_AA)
                cv2.line(frame, (x, y - r), (x, y + r), s[5], 2, cv2.LINE_AA)
                alive.append(s)
        self.p = alive


# ---------------------------------------------------------------- stickers on the mirror
class Sticker:
    def __init__(self, kind, img, x, y, now):
        self.kind, self.x, self.y = kind, x, y
        self.angle = random.uniform(-15, 15)
        self.born = now
        self.base = img                                  # full resolution, so zooming in stays sharp
        self.base_scale = STICKER_SIZE / max(img.shape[:2])
        self.zoom = 1.0
        self.held = False
        self.dropped = None

    def radius(self):
        return STICKER_SIZE * 0.5 * self.zoom

    def scale(self, now):
        t = (now - self.born) / 0.35
        s = ease_out_back(min(1, max(0.0, t))) if t < 1 else 1.0
        if self.held:
            s *= 1.12
        elif self.dropped is not None:
            d = (now - self.dropped) / 0.25
            if d < 1:
                s *= 1 + 0.12 * (1 - d) * math.cos(d * math.pi * 1.5)
        return max(s, 0.01)

    def draw(self, frame, now):
        img = rotate_scale(self.base, self.angle, self.scale(now) * self.base_scale * self.zoom)
        paste(frame, img, self.x, self.y, shadow=0.18 if self.held else 0.06,
              lift=10 if self.held else 0)


# ---------------------------------------------------------------- hands
class HandState:
    def __init__(self):
        self.pinch = None      # which finger is touching the thumb, or None
        self.last_pen = None
        self.pen_s = None
        self.lost_since = None
        self.draw_start = 0.0
        self.drew_at = -10.0   # when this hand last drew (no wiping right after, so letters survive)
        self.last_edge = None
        self.pinch_since = 0.0
        self.spawned = False
        self.smooth = None


def hand_size(lm):
    return np.linalg.norm(lm[0] - lm[9]) + 1e-6


def is_fist(lm):
    """A closed hand: at least 3 fingertips curled in (closer to the wrist than their knuckles)."""
    wrist = lm[0]
    curled = sum(np.linalg.norm(lm[tip] - wrist) < np.linalg.norm(lm[tip - 3] - wrist) * 1.15
                 for tip in (8, 12, 16, 20))
    return curled >= 3


def still_writing(lm, margin=0.12):
    """Keep writing at any finger angle: middle, ring and pinky stay folded, and the index is the finger
    sticking out the most (measured from the palm, so turning or tilting the finger doesn't matter)."""
    palm = lm[[0, 5, 9, 13, 17]].mean(axis=0)
    size = hand_size(lm)
    out = {t: float(np.linalg.norm(lm[t] - palm)) / size for t in (8, 12, 16, 20)}
    folded = max(out[12], out[16], out[20]) < 0.95
    return folded and out[8] > max(out[12], out[16], out[20]) + margin


def is_sideways(lm, already=False):
    """Hand turned on its side, pinky edge toward the screen: the knuckles line up, so the palm looks narrow."""
    palm = float(np.linalg.norm(lm[5] - lm[17]))         # index knuckle to pinky knuckle, as seen in the picture
    return palm < hand_size(lm) * (0.5 if already else 0.4)


def is_pointing_at_screen(lm, z, already=False):
    """Index finger pointing toward the camera/screen (not up at the ceiling), other fingers bent.

    Pointing at the screen, the index finger looks short in the picture and its tip is much closer to
    the camera than its knuckle (MediaPipe gives a depth value z: smaller = closer).
    """
    if z is None:
        return False
    size = hand_size(lm)
    wrist = lm[0]
    others_bent = all(np.linalg.norm(lm[t] - wrist) < np.linalg.norm(lm[t - 3] - wrist) * 1.25 for t in (12, 16, 20))
    if not others_bent:
        return False
    flat = float(np.linalg.norm(lm[8] - lm[5]))          # index length as seen in the picture
    toward = float(z[5] - z[8])                          # how much closer the tip is than the knuckle
    k = 0.6 if already else 0.9                          # a bit easier to keep writing than to start
    return toward > max(flat * k, size * 0.18)


def is_pointing(lm):
    """Index finger straight, middle, ring and pinky bent down (like pointing)."""
    wrist = lm[0]
    def straight(tip):
        return np.linalg.norm(lm[tip] - wrist) > np.linalg.norm(lm[tip - 2] - wrist) * 1.15
    def bent(tip):
        return np.linalg.norm(lm[tip] - wrist) < np.linalg.norm(lm[tip - 2] - wrist) * 1.05
    return straight(8) and bent(12) and bent(16) and bent(20)


def fist_edge(lm, size):
    """The pinky side of a closed fist, from the pinky knuckle down to the wrist (the part you wipe with)."""
    a, b = lm[17], lm[0]
    d = b - a
    a2, b2 = a - d * 0.2, b + d * 0.15
    n = np.array([-d[1], d[0]])
    n = n / (np.linalg.norm(n) + 1e-6)
    if np.dot(n, a - lm[9]) < 0:          # point away from the palm, toward the pinky edge
        n = -n
    off = n * size * 0.12
    return a2 + off, b2 + off


class FogWindow:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.fog = Fog(w, h)
        self.window = RealWindow(w, h)
        self.drops = Droplets(w, h, self.window.warm)
        self.facefx = FaceFX()
        self.face_pts = None
        self.sparkles = Sparkles()
        self.puffs = BreathPuffs()
        self.breath_time = 0.0
        self.blowing = False
        self.show_help = True
        self.hand_z = {}
        self.help_since = None
        self.sticker_imgs = load_my_stickers() or build_all(clear=False)
        self.sticker_imgs = {k: make_clear(v, STICKER_OPACITY) if STICKER_OPACITY < 1 else v
                             for k, v in self.sticker_imgs.items()}
        kinds = list(self.sticker_imgs)
        self.pages = [[kinds[(p + j) % len(kinds)] for j in range(4)] for p in range(0, len(kinds), 4)]
        self.page = 0
        self.page_changed_at = -10.0
        self.stickers = []
        # mini stickers that ride on your fingertips so you know which finger makes what
        self.mini = {k: cv2.resize(img, None, fx=MINI_SIZE / max(img.shape[:2]), fy=MINI_SIZE / max(img.shape[:2]),
                                   interpolation=cv2.INTER_AREA)
                     for k, img in self.sticker_imgs.items()}
        self.states = {"Left": HandState(), "Right": HandState()}
        self.sticker_hand = "Right"
        self.grabbed = None
        self.grab_offset = (0, 0)
        self.last_spawn = 0.0

    def finger_tips(self):
        """The 4 stickers on the current page, mapped to index/middle/ring/pinky fingertips."""
        return dict(zip(self.pages[self.page], TIPS))

    def next_page(self, now):
        self.page = (self.page + 1) % len(self.pages)
        self.page_changed_at = now

    def move_hand(self):
        return "Left" if self.sticker_hand == "Right" else "Right"

    # --- one frame of logic
    # hands = list of (label, 21x2 pixel landmarks); face = (mouth_x, mouth_y, mouth_open, face_width) or None
    def update(self, hands, face, now, dt):
        self.blowing = face is not None and face[2] > MOUTH_OPEN
        if self.blowing:
            mx, my, _, fw = face
            self.breath_time += dt
            radius = fw * (0.5 + self.breath_time * 1.4)
            self.fog.breathe(mx, my, radius, dt / BREATH_FOG_SECONDS * 2.5)
            self.puffs.emit(mx, my, fw)
        else:
            self.breath_time = max(0.0, self.breath_time - dt * 3)

        seen = set()
        for label, lm in hands:
            if label in seen:  # two hands with the same label: skip the second
                continue
            seen.add(label)
            st = self.states[label]
            st.smooth = lm if st.smooth is None else st.smooth * 0.45 + lm * 0.55
            lm = st.smooth
            size = hand_size(lm)
            # closed fist: wipe with the pinky side of the hand (unless you're holding an ornament)
            fist = label == self.sticker_hand and is_fist(lm)      # only the sticker hand wipes
            if fist and st.pinch != "grab":
                if st.pinch is not None and not (label == self.sticker_hand and st.spawned):
                    st.pinch = None              # a pinch that was really a fist closing: cancel it
                if st.pinch is None:
                    e0, e1 = fist_edge(lm, size)
                    if st.last_edge is not None:
                        self.fog.wipe_sweep(st.last_edge, (e0, e1), size * 0.2)
                    else:
                        self.fog.wipe(e0, e1, size * 0.2)
                    st.last_edge = (e0, e1)
                    continue
            st.last_edge = None
            if label == self.sticker_hand:
                self._sticker_hand(st, lm, size, now)
            else:
                self._move_hand(st, lm, size, now)

        for label, st in self.states.items():
            if label not in seen:
                st.smooth, st.last_edge = None, None
                if st.pinch is not None:
                    st.pinch = None
                    if label == self.move_hand():
                        self._drop(now)
        self.fog.regrow(dt)
        self.drops.update_runners(self.fog, self.window.glass, dt)

    def _sticker_hand(self, st, lm, size, now):
        tips = self.finger_tips()
        dists = {k: np.linalg.norm(lm[THUMB_TIP] - lm[i]) / size for k, i in tips.items()}
        kind = min(dists, key=dists.get)
        if st.pinch is None:
            if dists[kind] < PINCH_ON:
                st.pinch, st.pinch_since, st.spawned = kind, now, False
        elif st.pinch not in dists or dists[st.pinch] > PINCH_OFF:
            st.pinch = None
        # spawn once the touch has been held a moment (so closing your hand into a fist doesn't spawn one)
        if st.pinch and not st.spawned and now - st.pinch_since > 0.12:
            st.spawned = True
            if now - self.last_spawn > SPAWN_COOLDOWN:
                x, y = (lm[THUMB_TIP] + lm[tips[st.pinch]]) / 2
                self.stickers.append(Sticker(st.pinch, self.sticker_imgs[st.pinch], x, y, now))
                self.sparkles.burst(x, y)
                self.last_spawn = now

    def _move_hand(self, st, lm, size, now):
        d_idx = np.linalg.norm(lm[THUMB_TIP] - lm[8]) / size
        d_mid = np.linalg.norm(lm[THUMB_TIP] - lm[12]) / size
        d_ring = np.linalg.norm(lm[THUMB_TIP] - lm[16]) / size
        # thumb + ring: next face filter
        if st.pinch is None and d_ring < PINCH_ON and d_ring < d_mid and d_ring < d_idx:
            st.pinch = "filter"
            st.pinch_since = now
            self.facefx.next(now)
            return
        if st.pinch == "filter":
            if d_ring > PINCH_OFF:
                st.pinch = None
            return
        pen = lm[8]                              # the index fingertip writes in the fog
        grab_pt = (lm[THUMB_TIP] + lm[12]) / 2   # thumb + middle: grab, move and zoom a sticker
        # write "air canvas" style: only the index finger out = pen down; index + middle out = pen up.
        # Once writing, keep going at any angle (like holding a ballpoint), as long as the other fingers stay folded.
        if st.pinch == "draw":
            ok = still_writing(lm) and d_mid > PINCH_ON
            if ok:
                st.lost_since = None
            elif st.lost_since is None:
                st.lost_since = now
            pointing = ok or now - st.lost_since < 0.25          # ride out a moment of shaky tracking
        else:
            pointing = still_writing(lm, margin=0.25) and d_mid > PINCH_ON
        if st.pinch is None:
            if d_mid < PINCH_ON and d_mid < d_idx:
                self._start_grab(st, grab_pt, size)
            elif pointing:
                st.pinch = "draw"
                st.last_pen = None
                st.draw_start = now
        elif st.pinch == "grab" and d_mid > PINCH_OFF:
            st.pinch = None
            self._drop(now)
        elif st.pinch == "draw" and not pointing:
            st.pinch = None
            st.last_pen = None

        if st.pinch == "grab" and self.grabbed:
            g = self.grabbed
            # hand closer to the camera looks bigger -> sticker gets bigger
            target = float(np.clip(self.grab_zoom * (size / self.grab_size) ** 1.5, ZOOM_MIN, ZOOM_MAX))
            g.zoom = g.zoom * 0.7 + target * 0.3
            k = g.zoom / self.grab_zoom
            g.x = grab_pt[0] + self.grab_offset[0] * k
            g.y = grab_pt[1] + self.grab_offset[1] * k
        if st.pinch == "draw" and now - st.draw_start > 0.1:    # tiny delay so a passing pose doesn't leave a dot
            # smooth the pen so letters come out clean, not shaky
            st.pen_s = pen if st.pen_s is None else st.pen_s * 0.55 + pen * 0.45
            if st.last_pen is not None:
                self.fog.wipe(st.last_pen, st.pen_s, DRAW_WIDTH / 2)
            st.last_pen = st.pen_s.copy()
            st.drew_at = now
        if st.pinch != "draw":
            st.pen_s = None

    def _start_grab(self, st, grab_pt, size):
        st.pinch = "grab"
        st.last_pen = None
        for s in reversed(self.stickers):  # top-most first
            if math.hypot(s.x - grab_pt[0], s.y - grab_pt[1]) < s.radius() * 1.1:
                self.grabbed = s
                s.held = True
                self.grab_offset = (s.x - grab_pt[0], s.y - grab_pt[1])
                self.grab_size, self.grab_zoom = size, s.zoom
                self.stickers.remove(s)
                self.stickers.append(s)  # bring to front
                break

    def _drop(self, now):
        if self.grabbed:
            g = self.grabbed
            g.held, g.dropped = False, now
            if self._in_trash(g.x, g.y):
                self.stickers.remove(g)
                self.sparkles.burst(g.x, g.y, 10)
            self.grabbed = None

    def _in_trash(self, x, y):
        tx, ty = self._trash_pos()
        return math.hypot(x - tx, y - ty) < 60

    def _trash_pos(self):
        gx1, gy1, gx2, gy2 = self.window.glass
        return gx2 - 70, gy2 - 70

    # --- drawing
    def draw(self, frame, hands, now, dt):
        frame = self.facefx.apply(frame, self.face_pts, now)    # face filter first, so it sits under the fog
        outside = self.window.outside(frame, now, dt)
        out = self.fog.render(outside, lambda sh: self.window.fog_tint(now, sh))
        out = self.drops.apply(out, outside, self.fog.m_full)
        self.window.draw_ice(out)
        self.window.glass_fx(out, now)
        self.puffs.update_draw(out, dt)
        for s in self.stickers:
            s.draw(out, now)
        self.sparkles.update_draw(out, dt)
        self.window.draw_frame(out, now)
        if self.fog.coverage() < 0.04 and not self.stickers:
            k = 0.5 + 0.5 * math.sin(now * 2)
            msg = text.render("breathe on it...", 58, (255, 250, 240), "script", 0.5 + 0.3 * k)
            if msg is not None:
                paste(out, msg, self.w // 2, int(self.h * 0.36))
        if self.grabbed:
            hot = self._in_trash(self.grabbed.x, self.grabbed.y)
            c = self._trash_pos()
            ring = (120, 140, 255) if hot else (255, 255, 255)
            cv2.circle(out, c, 52, ring, 1, cv2.LINE_AA)
            label = text.render("drop to remove", 22, (255, 255, 255), "script", 0.7)
            if label is not None:
                paste(out, label, c[0], c[1])
        for label, _ in hands:
            st = self.states[label]
            if st.smooth is None or st.last_edge is not None:   # hide previews while wiping with a fist
                continue
            lm = st.smooth
            size = hand_size(lm)
            if label != self.sticker_hand:
                # drag hand previews: index = draw in the fog, middle = move/zoom
                for tip, icon in ((8, "draw"), (12, "move"), (16, "filter")):
                    d = lm[tip] - lm[tip - 1]
                    d = d / (np.linalg.norm(d) + 1e-6)
                    pos = lm[tip] + d * 26
                    close = np.linalg.norm(lm[THUMB_TIP] - lm[tip]) / size
                    grow = 1 + 0.4 * float(np.clip((0.9 - close) / 0.6, 0, 1))
                    active = st.pinch == {"move": "grab", "draw": "draw", "filter": "filter"}[icon]
                    draw_badge(out, pos, icon, 1.35 if active else grow, active)
                continue
            for kind, tip in self.finger_tips().items():
                d = lm[tip] - lm[tip - 1]
                d = d / (np.linalg.norm(d) + 1e-6)
                pos = lm[tip] + d * MINI_SIZE * 0.55           # just past the fingertip
                close = np.linalg.norm(lm[THUMB_TIP] - lm[tip]) / size
                grow = 1 + 0.45 * float(np.clip((0.9 - close) / 0.6, 0, 1))  # grows as it nears your thumb
                if st.pinch == kind:
                    grow = 1.5
                tilt = math.degrees(math.atan2(d[0], -d[1])) * 0.5
                paste(out, rotate_scale(self.mini[kind], -tilt, grow), int(pos[0]), int(pos[1]),
                      shadow=0.35, lift=2)
        for label, _ in hands:
            st = self.states[label]
            if st.smooth is None:
                continue
            lm = st.smooth
            stick = label == self.sticker_hand
            tip = lm[8] if st.pinch == "draw" else (lm[THUMB_TIP] + lm[12]) / 2
            p = (int(tip[0]), int(tip[1]))
            if st.pinch == "draw":
                cv2.circle(out, p, DRAW_WIDTH // 2 + 3, (255, 255, 255), 1, cv2.LINE_AA)
            elif st.pinch:
                cv2.circle(out, p, 11, (170, 225, 255), 1, cv2.LINE_AA)
                cv2.circle(out, p, 3, (170, 225, 255), -1, cv2.LINE_AA)
            else:
                cv2.circle(out, p, 7, (255, 255, 255), 1, cv2.LINE_AA)
            if self.show_help:
                tag = text.render("stickers" if stick else "write  \u00b7  move  \u00b7  filter", 24, (255, 255, 255), "script", 0.7)
                if self.grabbed is not None and not stick and st.pinch == "grab":
                    tag = text.render("closer for bigger, farther for smaller", 24, (255, 255, 255), "script", 0.7)
                if tag is not None:
                    w0 = lm[0]
                    paste(out, tag, int(w0[0]), int(w0[1] + 34))
        if len(self.pages) > 1 and now - self.page_changed_at < 1.8:
            k = 1 - max(0.0, (now - self.page_changed_at - 1.0) / 0.8)
            label = text.render(f"stickers  {self.page + 1} of {len(self.pages)}", 40, (255, 250, 242), "script", 0.6)
            if label is not None:
                paste(out, label, self.w // 2, int(self.h * 0.12), opacity=k)
        if now - self.facefx.changed_at < 1.8:   # show the filter name when you switch
            k = 1 - max(0.0, (now - self.facefx.changed_at - 1.0) / 0.8)
            label = text.render(self.facefx.name, 40, (255, 250, 242), "script", 0.6)
            if label is not None:
                paste(out, label, self.w // 2, int(self.h * 0.12), opacity=k)
        self._hud(out)
        return out

    def _hud(self, out):
        if not self.show_help:
            return
        if self.help_since is None:
            self.help_since = time.time()
        fade = 1 - np.clip((time.time() - self.help_since - HELP_SECONDS) / 1.5, 0, 1)
        if fade <= 0:
            return
        gx1, gy1, gx2, gy2 = self.window.glass
        side = "right" if self.sticker_hand == "Right" else "left"
        line1 = (f"breathe to fog  \u00b7  {side} hand: fist to wipe, thumb to a finger for a sticker"
                 "  \u00b7  N for more")
        line2 = "other hand: index finger up to write, two fingers to lift  \u00b7  thumb + middle moves a sticker (closer = bigger)  \u00b7  thumb + ring: face filter"
        line3 = "S swap hands   \u00b7   R fog   \u00b7   W clear   \u00b7   Z undo   \u00b7   C clear all   \u00b7   P photo   \u00b7   N stickers   \u00b7   E filter   \u00b7   V camera   \u00b7   H hide   \u00b7   Q quit"
        y = self.h - 74
        for i, line in enumerate((line1, line2, line3)):
            img = text.render(line, 21 if i < 2 else 17, (255, 248, 238) if i < 2 else (235, 225, 210),
                              "script" if i < 2 else "sans", 0.9)
            if img is not None:
                paste(out, img, self.w // 2, y + i * 25, opacity=fade)


def is_o_mouth(roundness, scores):
    """True when the lips make an "O" like blowing on glass, not a smile.

    roundness = lip gap / mouth width (an O is tall and narrow, a smile is wide and flat).
    scores = MediaPipe expression scores, when available (funnel/pucker = O-lips, smile = smiling).
    """
    smile = max(scores.get("mouthSmileLeft", 0), scores.get("mouthSmileRight", 0))
    if smile > 0.35:
        return False
    if scores:
        o_lips = max(scores.get("mouthFunnel", 0), scores.get("mouthPucker", 0))
        if o_lips > 0.25:
            return True
    return roundness > O_ROUNDNESS


CAMERA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".camera")


def save_camera(index):
    try:
        with open(CAMERA_FILE, "w") as f:
            f.write(str(index))
    except OSError:
        pass


def pick_camera():
    """Your last choice if there is one; otherwise, on a Mac, the built-in FaceTime camera (not the iPhone)."""
    try:
        with open(CAMERA_FILE) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        pass
    if sys.platform == "darwin":
        try:
            import json as _json
            import subprocess
            info = subprocess.run(["system_profiler", "SPCameraDataType", "-json"], capture_output=True,
                                  text=True, timeout=8).stdout
            names = [c.get("_name", "") for c in _json.loads(info).get("SPCameraDataType", [])]
            for i, name in enumerate(names):
                if any(k in name.lower() for k in ("facetime", "built-in", "macbook")):
                    print(f"Using the laptop camera: {name} (press V to switch)")
                    return i
        except Exception:
            pass
    return 0


def open_camera(index):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    return cap


# ---------------------------------------------------------------- main loop
def main():
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    hand_model = download_model("hand_landmarker.task")
    face_model = download_model("face_landmarker.task")
    face_landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=face_model),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,     # expression scores (O-mouth vs smile) for breathing
    ))
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=hand_model),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        cam_index = int(sys.argv[1])
    elif CAMERA_INDEX is not None:
        cam_index = CAMERA_INDEX
    else:
        cam_index = pick_camera()
    cap = open_camera(cam_index)
    ok, frame = cap.read()
    if not ok:
        print("Couldn't open the camera. On a Mac, allow camera access for Terminal in "
              "System Settings > Privacy & Security > Camera, then try again.")
        sys.exit(1)
    h, w = frame.shape[:2]
    app = FogWindow(w, h)
    app.facefx.changed_at = time.time() + 1.5   # show the face filter name for a moment at startup
    cam_label_at = time.time()

    win = "Christmas Fog Window"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, w, h)
    fullscreen = False
    start = last = time.time()
    last_ts = -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        frame = cv2.flip(frame, 1)  # mirror
        now = time.time()
        dt = min(now - last, 0.1)
        last = now

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts = max(int((now - start) * 1000), last_ts + 1)
        last_ts = ts
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_img, ts)
        face_result = face_landmarker.detect_for_video(mp_img, ts)

        face = None
        if face_result.face_landmarks:
            f = face_result.face_landmarks[0]
            def px(i):
                return np.array([f[i].x * w, f[i].y * h])
            face_h = np.linalg.norm(px(10) - px(152)) + 1e-6      # forehead to chin
            gap = np.linalg.norm(px(13) - px(14))                  # inner lips gap
            width = np.linalg.norm(px(78) - px(308)) + 1e-6          # inner mouth corners
            mouth_open = gap / face_h
            scores = {}
            if face_result.face_blendshapes:
                scores = {c.category_name: c.score for c in face_result.face_blendshapes[0]}
            if not is_o_mouth(gap / width, scores):
                mouth_open = 0.0                                     # smiling or talking: no fog
            mouth = (px(13) + px(14)) / 2
            face = (mouth[0], mouth[1], mouth_open, np.linalg.norm(px(234) - px(454)))
            app.face_pts = np.array([[p.x * w, p.y * h] for p in f], np.float32)
        else:
            app.face_pts = None

        hands = []
        for lms, handed in zip(result.hand_landmarks, result.handedness):
            pts = np.array([[p.x * w, p.y * h] for p in lms], np.float32)
            hands.append((handed[0].category_name, pts))
            app.hand_z[handed[0].category_name] = np.array([p.z * w for p in lms], np.float32)  # depth, in pixels

        app.update(hands, face, now, dt)
        out = app.draw(frame, hands, now, dt)
        if time.time() - cam_label_at < 2.5:
            tag = text.render(f"camera {cam_index}   \u00b7   press V to switch", 22, (255, 250, 242), "script", 0.8)
            if tag is not None:
                paste(out, tag, w // 2, 30)
        cv2.imshow(win, out)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("s"):
            app.sticker_hand = app.move_hand()
        elif key == ord("r"):
            app.fog.mask[:] = 1
        elif key == ord("w"):
            app.fog.mask[:] = 0
        elif key == ord("c"):
            app.stickers.clear()
            app.grabbed = None
        elif key == ord("z") and app.stickers:
            app.stickers.pop()
            app.grabbed = None
        elif key == ord("p"):
            os.makedirs("screenshots", exist_ok=True)
            path = os.path.join("screenshots", time.strftime("christmas-fog-window-%Y%m%d-%H%M%S.png"))
            cv2.imwrite(path, out)
            print("Saved", path)
        elif key == ord("v"):                     # switch to the next camera (e.g. laptop vs iPhone)
            for step_i in range(1, 5):
                nxt = (cam_index + step_i) % 5
                c2 = open_camera(nxt)
                ok2, _ = c2.read()
                if ok2:
                    cap.release()
                    cap, cam_index = c2, nxt
                    cam_label_at = time.time()
                    save_camera(cam_index)
                    print(f"Using camera {cam_index} (remembered for next time)")
                    break
                c2.release()
        elif key == ord("n"):
            app.next_page(now)
        elif key == ord("e"):
            app.facefx.next(now)
        elif key == ord("h"):
            app.show_help = True
            app.help_since = time.time()
        elif key == ord("f"):
            fullscreen = not fullscreen
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    landmarker.close()
    face_landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
