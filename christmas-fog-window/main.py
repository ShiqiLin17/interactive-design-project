"""Christmas Fog Window — fog up the window with your breath, wipe it with your hand, decorate it with stickers.

Your face:     open your mouth and blow at the mirror to fog it up
Sticker hand:  touch your thumb to a finger to pop out a sticker
               index = tree, middle = snowman, ring = gift, pinky = candy cane
Move hand:     pinch thumb + index on a sticker to grab it, release to drop it
Either hand:   open palm wipes the fog away

Keys: S swap hands · R fog all · W wipe all · H help · Z undo · C clear stickers · P screenshot · F fullscreen · Q quit
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

from stickers import build_all
from window import ChristmasWindow
import text

# ---------------------------------------------------------------- settings
CAMERA_INDEX = 0
WIDTH, HEIGHT = 1280, 720
FOG_REGROW_SECONDS = 0       # 0 = fog only comes from your breath; e.g. 25 = slowly fogs back up
BREATH_FOG_SECONDS = 1.8     # how long you blow to fog up the whole mirror
MOUTH_OPEN = 0.045           # how open your mouth must be to count as blowing (raise if too sensitive)
STICKER_SIZE = 150           # on-screen sticker size in pixels
PINCH_ON = 0.32              # fingertip distance (relative to hand size) to start a pinch
PINCH_OFF = 0.48             # ...and to release it
SPAWN_COOLDOWN = 0.45        # seconds between stickers
MINI_SIZE = 46               # size of the little stickers shown on your fingertips

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                            "hand_landmarker/float16/latest/hand_landmarker.task",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                            "face_landmarker/float16/latest/face_landmarker.task",
}

FINGER_TIPS = {"tree": 8, "snowman": 12, "gift": 16, "candy cane": 20}
THUMB_TIP = 4


# ---------------------------------------------------------------- helpers
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


def paste(frame, rgba, cx, cy, shadow=0.0, lift=0):
    """Alpha-blend an RGBA image onto frame, centered at (cx, cy)."""
    h, w = rgba.shape[:2]
    H, W = frame.shape[:2]
    alpha = rgba[..., 3].astype(np.float32) / 255

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


# ---------------------------------------------------------------- fog
class Fog:
    def __init__(self, w, h, scale=4):
        self.w, self.h, self.s = w, h, scale
        self.mask = np.zeros((h // scale, w // scale), np.float32)  # starts clear
        ys, xs = np.mgrid[0:h // scale, 0:w // scale]
        self.grid = (xs.astype(np.float32), ys.astype(np.float32))
        rng = np.random.default_rng(7)
        # condensation: thousands of tiny droplets, each with a dark edge and a bright highlight
        drops = np.zeros((h, w), np.float32)
        for _ in range(int(w * h / 90)):
            x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
            r = int(rng.choice([1, 1, 1, 2, 2, 3]))
            cv2.circle(drops, (x, y), r, -0.6, -1, cv2.LINE_AA)
            cv2.circle(drops, (x - (r > 1), y - (r > 1)), max(1, r - 1), 1.0, -1, cv2.LINE_AA)
        drops = cv2.GaussianBlur(drops, (0, 0), 0.6)
        haze = cv2.resize(rng.random((h // 40, w // 40)).astype(np.float32), (w, h),
                          interpolation=cv2.INTER_CUBIC)
        self.texture = (drops * 16 + (haze - 0.5) * 22)[..., None].astype(np.float32)
        self.tint = np.array([236, 230, 224], np.float32)  # soft misty white (BGR)

    def wipe(self, p0, p1, radius):
        s = self.s
        a = (int(p0[0] / s), int(p0[1] / s))
        b = (int(p1[0] / s), int(p1[1] / s))
        r = max(2, int(radius / s))
        cv2.line(self.mask, a, b, 0.0, 2 * r, cv2.LINE_AA)
        cv2.circle(self.mask, b, r, 0.0, -1, cv2.LINE_AA)

    def breathe(self, x, y, radius, amount):
        """Add fog in a soft cloud around (x, y)."""
        s = self.s
        d2 = (self.grid[0] - x / s) ** 2 + (self.grid[1] - y / s) ** 2
        blob = np.exp(-d2 / (2 * (radius / s) ** 2))
        self.mask += blob * amount
        np.clip(self.mask, 0, 1, out=self.mask)

    def coverage(self):
        return float(self.mask.mean())

    def regrow(self, dt):
        if FOG_REGROW_SECONDS > 0:
            self.mask += dt / FOG_REGROW_SECONDS
            np.clip(self.mask, 0, 1, out=self.mask)

    def render(self, frame):
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (w // 8, h // 8), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 4).astype(np.float32) * 0.5 + self.tint * 0.5
        fogged = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC) + self.texture
        m = cv2.GaussianBlur(self.mask, (0, 0), 2.0)
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        rim = (m * (1 - m) * 4) ** 2           # denser droplets at the edge of a wipe
        m = np.clip(0.04 + 0.96 * m + rim * 0.25, 0, 1)
        return cv2.blendLinear(frame, cv2.convertScaleAbs(fogged), 1 - m, m)


# ---------------------------------------------------------------- snow + sparkles
class Snow:
    def __init__(self, w, h, n=140):
        self.w, self.h = w, h
        self.flakes = [self._new(random.uniform(0, h)) for _ in range(n)]

    def _new(self, y=None):
        r = random.choice([1, 1, 2, 2, 3, 4])
        return [random.uniform(0, self.w), -10 if y is None else y, r,
                random.uniform(25, 60) * (0.6 + r / 4), random.uniform(0, 6.28)]

    def update_draw(self, frame, dt, t):
        layer = frame.copy()
        for i, f in enumerate(self.flakes):
            f[1] += f[3] * dt
            f[0] += math.sin(t * 1.3 + f[4]) * 12 * dt
            if f[1] > self.h + 10:
                self.flakes[i] = f = self._new()
            cv2.circle(layer, (int(f[0]), int(f[1])), f[2], (255, 255, 255), -1, cv2.LINE_AA)
        cv2.addWeighted(layer, 0.75, frame, 0.25, 0, frame)


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
            col = random.choice([(80, 215, 255), (255, 255, 255), (120, 120, 255), (140, 230, 140)])
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
        self.base = cv2.resize(img, None, fx=STICKER_SIZE / img.shape[1],
                               fy=STICKER_SIZE / img.shape[1], interpolation=cv2.INTER_AREA)
        self.held = False
        self.dropped = None

    def radius(self):
        return STICKER_SIZE * 0.5

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
        img = rotate_scale(self.base, self.angle, self.scale(now))
        paste(frame, img, self.x, self.y, shadow=0.45 if self.held else 0.3,
              lift=10 if self.held else 0)


# ---------------------------------------------------------------- hands
class HandState:
    def __init__(self):
        self.pinch = None      # which finger is touching the thumb, or None
        self.last_palm = None
        self.smooth = None


def hand_size(lm):
    return np.linalg.norm(lm[0] - lm[9]) + 1e-6


def is_open(lm):
    wrist = lm[0]
    up = sum(np.linalg.norm(lm[tip] - wrist) > np.linalg.norm(lm[tip - 2] - wrist) * 1.1
             for tip in (8, 12, 16, 20))
    return up >= 3


class FogWindow:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.fog = Fog(w, h)
        self.window = ChristmasWindow(w, h)
        self.snow = Snow(w, h)
        self.sparkles = Sparkles()
        self.puffs = BreathPuffs()
        self.breath_time = 0.0
        self.blowing = False
        self.show_help = True
        self.sticker_imgs = build_all()
        self.stickers = []
        # mini stickers that ride on your fingertips so you know which finger makes what
        self.mini = {k: cv2.resize(img, None, fx=MINI_SIZE / img.shape[1], fy=MINI_SIZE / img.shape[1],
                                   interpolation=cv2.INTER_AREA)
                     for k, img in self.sticker_imgs.items()}
        self.states = {"Left": HandState(), "Right": HandState()}
        self.sticker_hand = "Right"
        self.grabbed = None
        self.grab_offset = (0, 0)
        self.last_spawn = 0.0

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
            if label == self.sticker_hand:
                self._sticker_hand(st, lm, size, now)
            else:
                self._move_hand(st, lm, size, now)
            # wipe with an open palm (only when not pinching)
            palm = lm[[0, 5, 9, 13, 17]].mean(axis=0)
            if st.pinch is None and is_open(lm):
                self.fog.wipe(st.last_palm if st.last_palm is not None else palm, palm, size * 0.95)
                st.last_palm = palm
            else:
                st.last_palm = None

        for label, st in self.states.items():
            if label not in seen:
                st.smooth, st.last_palm = None, None
                if st.pinch is not None:
                    st.pinch = None
                    if label == self.move_hand():
                        self._drop(now)
        self.fog.regrow(dt)

    def _sticker_hand(self, st, lm, size, now):
        dists = {k: np.linalg.norm(lm[THUMB_TIP] - lm[i]) / size for k, i in FINGER_TIPS.items()}
        kind = min(dists, key=dists.get)
        if st.pinch is None:
            if dists[kind] < PINCH_ON:
                st.pinch = kind
                if now - self.last_spawn > SPAWN_COOLDOWN:
                    x, y = (lm[THUMB_TIP] + lm[FINGER_TIPS[kind]]) / 2
                    self.stickers.append(Sticker(kind, self.sticker_imgs[kind], x, y, now))
                    self.sparkles.burst(x, y)
                    self.last_spawn = now
        elif dists[st.pinch] > PINCH_OFF:
            st.pinch = None

    def _move_hand(self, st, lm, size, now):
        d = np.linalg.norm(lm[THUMB_TIP] - lm[8]) / size
        point = (lm[THUMB_TIP] + lm[8]) / 2
        if st.pinch is None and d < PINCH_ON:
            st.pinch = "grab"
            for s in reversed(self.stickers):  # top-most first
                if math.hypot(s.x - point[0], s.y - point[1]) < s.radius() * 1.1:
                    self.grabbed = s
                    s.held = True
                    self.grab_offset = (s.x - point[0], s.y - point[1])
                    self.stickers.remove(s)
                    self.stickers.append(s)  # bring to front
                    break
        elif st.pinch is not None and d > PINCH_OFF:
            st.pinch = None
            self._drop(now)
        if st.pinch and self.grabbed:
            self.grabbed.x = point[0] + self.grab_offset[0]
            self.grabbed.y = point[1] + self.grab_offset[1]

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
        outside = self.window.reflection(frame)
        self.snow.update_draw(outside, dt, now)       # snow falls outside the glass
        out = self.fog.render(outside)
        self.puffs.update_draw(out, dt)
        for s in self.stickers:
            s.draw(out, now)
        self.sparkles.update_draw(out, dt)
        self.window.draw_frame(out, now)
        if self.fog.coverage() < 0.04 and not self.stickers:
            k = 0.5 + 0.5 * math.sin(now * 2)
            msg = text.render("blow on the window...", 54, (255, 250, 240), "script", 0.5 + 0.3 * k)
            if msg is not None:
                paste(out, msg, self.w // 2, int(self.h * 0.36))
        if self.grabbed:
            hot = self._in_trash(self.grabbed.x, self.grabbed.y)
            c = self._trash_pos()
            ring = (90, 90, 255) if hot else (255, 255, 255)
            cv2.circle(out, c, 48, ring, 2, cv2.LINE_AA)
            cv2.circle(out, c, 44, ring, 1, cv2.LINE_AA)
            label = text.render("drop to remove", 15, (255, 255, 255))
            if label is not None:
                paste(out, label, c[0], c[1])
        for label, _ in hands:
            st = self.states[label]
            if st.smooth is None or label != self.sticker_hand:
                continue
            lm = st.smooth
            size = hand_size(lm)
            for kind, tip in FINGER_TIPS.items():
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
            accent = (90, 90, 235) if stick else (120, 215, 140)
            tip = (lm[THUMB_TIP] + lm[8]) / 2
            p = (int(tip[0]), int(tip[1]))
            if st.pinch:
                cv2.circle(out, p, 12, accent, -1, cv2.LINE_AA)
                cv2.circle(out, p, 12, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.circle(out, p, 9, (255, 255, 255), 2, cv2.LINE_AA)
            if self.show_help:
                tag = text.render("stickers" if stick else "move", 17, (255, 255, 255))
                if tag is not None:
                    w0 = lm[0]
                    paste(out, tag, int(w0[0]), int(w0[1] + 34))
        self._hud(out)
        return out

    def _hud(self, out):
        if not self.show_help:
            return
        gx1, gy1, gx2, gy2 = self.window.glass
        side = "right" if self.sticker_hand == "Right" else "left"
        line1 = (f"blow to fog   \u00b7   open palm to wipe   \u00b7   {side} hand: thumb + index tree, "
                 "middle snowman, ring gift, pinky candy cane   \u00b7   other hand: pinch to move")
        line2 = "S swap hands    R fog    W clear fog    Z undo    C clear    P photo    H hide help    Q quit"
        y = gy2 + (self.h - gy2) // 2 + 12
        for i, line in enumerate((line1, line2)):
            img = text.render(line, 15 if i == 0 else 13, (255, 246, 230) if i == 0 else (225, 210, 190), "sans", 0.9)
            if img is not None:
                paste(out, img, self.w // 2, y + i * 20)


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

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    ok, frame = cap.read()
    if not ok:
        print("Couldn't open the camera. On a Mac, allow camera access for Terminal in "
              "System Settings > Privacy & Security > Camera, then try again.")
        sys.exit(1)
    h, w = frame.shape[:2]
    app = FogWindow(w, h)

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
            mouth_open = np.linalg.norm(px(13) - px(14)) / face_h  # inner lips gap
            mouth = (px(13) + px(14)) / 2
            face = (mouth[0], mouth[1], mouth_open, np.linalg.norm(px(234) - px(454)))

        hands = []
        for lms, handed in zip(result.hand_landmarks, result.handedness):
            pts = np.array([[p.x * w, p.y * h] for p in lms], np.float32)
            hands.append((handed[0].category_name, pts))

        app.update(hands, face, now, dt)
        out = app.draw(frame, hands, now, dt)
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
        elif key == ord("h"):
            app.show_help = not app.show_help
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
