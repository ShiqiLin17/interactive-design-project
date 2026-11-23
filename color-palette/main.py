"""Personal Color Test — a Korean-style color drape test you can do at home.

1. Pick a color type: point at a card with your index finger and hold (or pinch, or press 1-8).
2. Your 4 test colors appear on your fingertips. Touch your thumb to a finger:
   that color is draped around your face and a photo is taken.
3. After all 4 colors, the photos are shown side by side so you can compare.
SPACE = start over and pick another type.

Keys: 1-8 pick a type · 1-4 test a color · SPACE restart · V switch camera · P save screen · F fullscreen · Q quit
"""
import os
import ssl
import sys
import time
import urllib.request

import cv2
import numpy as np

import text
from palettes import PALETTES, bgr

# ---------------------------------------------------------------- settings
CAMERA_INDEX = None          # None = remember your last choice (press V to switch), or prefer the laptop camera
WIDTH, HEIGHT = 1280, 720
HOLD_TO_PICK = 1.0          # seconds pointing at a card to choose it
DRAPE_BEFORE_PHOTO = 1.1    # seconds the color shows on you before the photo
PINCH_ON, PINCH_OFF = 0.32, 0.48
SWATCH_R = 22               # fingertip swatch size
NEUTRAL = (150, 150, 150)   # neutral grey shown around your face while photos are taken (BGR)

BG = (238, 242, 246)        # warm off-white (BGR)
INK = (60, 52, 48)          # soft charcoal (BGR)
INK_RGB = (48, 52, 60)
MUTED_RGB = (130, 125, 122)

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                            "hand_landmarker/float16/latest/hand_landmarker.task",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                            "face_landmarker/float16/latest/face_landmarker.task",
}
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
             152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
FINGER_TIPS = [8, 12, 16, 20]   # index, middle, ring, pinky
THUMB_TIP = 4


# ---------------------------------------------------------------- helpers
def download_model(name):
    path = os.path.join(HERE, name)
    if os.path.exists(path):
        return path
    url = MODELS[name]
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


def paste(frame, rgba, cx, cy, opacity=1.0):
    """Alpha-blend a BGRA image onto frame, centered at (cx, cy)."""
    if rgba is None:
        return
    h, w = rgba.shape[:2]
    H, W = frame.shape[:2]
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    x1, y1, x2, y2 = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if x1 >= x2 or y1 >= y2:
        return
    src = rgba[y1 - y0:y2 - y0, x1 - x0:x2 - x0]
    a = src[..., 3:].astype(np.float32) / 255 * opacity
    roi = frame[y1:y2, x1:x2].astype(np.float32)
    frame[y1:y2, x1:x2] = (roi * (1 - a) + src[..., :3] * a).astype(np.uint8)


def put_text(frame, s, x, y, size, color=INK_RGB, style="sans", glow=0.0, anchor="center", opacity=1.0):
    img = text.render(s, size, color, style, glow)
    if img is None:
        cv2.putText(frame, s, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size / 30, color[::-1], 1, cv2.LINE_AA)
        return
    w = img.shape[1]
    if anchor == "left":
        x += w / 2
    elif anchor == "right":
        x -= w / 2
    paste(frame, img, x, y, opacity)


def rounded_rect(frame, x1, y1, x2, y2, r, color, alpha=1.0, shadow=0.0):
    """Draw a soft rounded rectangle (with optional drop shadow)."""
    H, W = frame.shape[:2]
    pad = 30
    bx1, by1 = max(int(x1) - pad, 0), max(int(y1) - pad, 0)
    bx2, by2 = min(int(x2) + pad, W), min(int(y2) + pad, H)
    if bx1 >= bx2 or by1 >= by2:
        return
    m = np.zeros((by2 - by1, bx2 - bx1), np.float32)
    ox, oy = bx1, by1
    r = int(r)
    p1, p2 = (int(x1 - ox), int(y1 - oy)), (int(x2 - ox), int(y2 - oy))
    cv2.rectangle(m, (p1[0] + r, p1[1]), (p2[0] - r, p2[1]), 1, -1)
    cv2.rectangle(m, (p1[0], p1[1] + r), (p2[0], p2[1] - r), 1, -1)
    for cx, cy in ((p1[0] + r, p1[1] + r), (p2[0] - r, p1[1] + r), (p1[0] + r, p2[1] - r), (p2[0] - r, p2[1] - r)):
        cv2.circle(m, (cx, cy), r, 1, -1, cv2.LINE_AA)
    roi = frame[by1:by2, bx1:bx2].astype(np.float32)
    if shadow > 0:
        s = cv2.GaussianBlur(np.roll(m, 8, axis=0), (0, 0), 10) * shadow
        roi = roi * (1 - s[..., None])
    a = (m * alpha)[..., None]
    roi = roi * (1 - a) + np.array(color, np.float32) * a
    frame[by1:by2, bx1:bx2] = np.clip(roi, 0, 255).astype(np.uint8)


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def hand_size(lm):
    return np.linalg.norm(lm[0] - lm[9]) + 1e-6


# ---------------------------------------------------------------- the drape
class Drape:
    """A cape of colored fabric that fills everything except your face."""

    def __init__(self, w, h):
        rng = np.random.default_rng(4)
        folds = cv2.resize(rng.random((3, 14)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        fine = cv2.resize(rng.random((8, 50)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        weave = rng.normal(0, 1, (h, w)).astype(np.float32)
        weave = cv2.GaussianBlur(weave, (0, 0), 0.7)
        y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        light = 1.04 - 0.12 * y                          # a little lighter near the top
        self.shade = (light * (0.9 + 0.12 * folds + 0.05 * fine) + weave * 0.012)[..., None]

    @staticmethod
    def face_alpha(face_pts, shape):
        h, w = shape[:2]
        oval = face_pts[FACE_OVAL]
        c = oval.mean(axis=0)
        oval = c + (oval - c) * np.array([1.0, 1.04])    # a hair taller so the forehead isn't cut
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [oval.astype(np.int32)], 255, cv2.LINE_AA)
        fh = np.linalg.norm(face_pts[10] - face_pts[152])
        m = cv2.GaussianBlur(m, (0, 0), max(1.5, fh * 0.012))
        return m.astype(np.float32) / 255

    def apply(self, frame, face_pts, color, amount=1.0):
        if face_pts is None or amount <= 0:
            return frame
        a = self.face_alpha(face_pts, frame.shape)
        cloth = np.clip(np.array(color, np.float32) * self.shade, 0, 255)
        k = (1 - a)[..., None] * amount
        out = frame.astype(np.float32) * (1 - k) + cloth * k
        return out.astype(np.uint8)


# ---------------------------------------------------------------- app
class ColorTest:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.drape = Drape(w, h)
        self.restart()

    def restart(self):
        self.mode = "pick"
        self.palette = None
        self.hover = None
        self.hover_since = 0.0
        self.pinching = False
        self.active = None          # index of the color being tried
        self.active_since = 0.0
        self.captured_at = None
        self.photos = [None] * 4
        self.flash = 0.0
        self.done_at = None
        self.saved = False
        self.face_s = None

    # --- layout of the 8 cards on the pick screen
    def cards(self):
        cols = 4
        mx, top, gap = self.w * 0.07, self.h * 0.27, self.w * 0.018
        cw = (self.w - 2 * mx - gap * (cols - 1)) / cols
        ch = self.h * 0.27
        out = []
        for i in range(len(PALETTES)):
            r, c = divmod(i, cols)
            x1 = mx + c * (cw + gap)
            y1 = top + r * (ch + gap * 1.6)
            out.append((x1, y1, x1 + cw, y1 + ch))
        return out

    def choose(self, i, now):
        self.palette = i
        self.mode = "test"
        self.active = None
        self.photos = [None] * 4

    def try_color(self, k, now):
        self.active = k
        self.active_since = now
        self.captured_at = None

    # --- one frame
    def step(self, frame, face_pts, hands, now):
        if face_pts is not None:
            self.face_s = face_pts if self.face_s is None else self.face_s * 0.4 + face_pts * 0.6
        else:
            self.face_s = None
        if self.mode == "pick":
            return self._pick(frame, hands, now)
        if self.mode == "test":
            return self._test(frame, hands, now)
        return self._compare(now)

    # ------------------------------------------------ screen 1: pick a type
    def _pick(self, frame, hands, now):
        small = cv2.resize(frame, (self.w // 12, self.h // 12), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 2)
        bg = cv2.resize(small, (self.w, self.h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        out = (bg * 0.25 + np.array(BG, np.float32) * 0.75).astype(np.uint8)

        put_text(out, "Personal Color Test", self.w / 2, self.h * 0.1, 46, INK_RGB, "serif")
        put_text(out, "point at your color type and hold  ·  or press 1–8", self.w / 2, self.h * 0.17, 17, MUTED_RGB)

        tip = None
        if hands:
            lm = hands[0][1]
            tip = lm[8]
        cards = self.cards()
        hover = None
        if tip is not None:
            for i, (x1, y1, x2, y2) in enumerate(cards):
                if x1 <= tip[0] <= x2 and y1 <= tip[1] <= y2:
                    hover = i
        if hover != self.hover:
            self.hover, self.hover_since = hover, now
        progress = (now - self.hover_since) / HOLD_TO_PICK if hover is not None else 0

        # pinch also picks
        if hands and hover is not None:
            lm = hands[0][1]
            d = np.linalg.norm(lm[THUMB_TIP] - lm[8]) / hand_size(lm)
            if not self.pinching and d < PINCH_ON:
                progress = 1
            self.pinching = d < PINCH_OFF if self.pinching else d < PINCH_ON

        for i, (x1, y1, x2, y2) in enumerate(cards):
            name, desc, colors = PALETTES[i]
            lift = 6 if i == hover else 0
            rounded_rect(out, x1, y1 - lift, x2, y2 - lift, 22, (252, 252, 253), 0.97,
                         shadow=0.22 if i == hover else 0.1)
            cw = x2 - x1
            cx = (x1 + x2) / 2
            put_text(out, f"{i + 1}", x1 + 22, y1 + 26 - lift, 13, MUTED_RGB)
            put_text(out, name, cx, y1 + 0.3 * (y2 - y1) - lift, 21, INK_RGB, "serif")
            put_text(out, desc, cx, y1 + 0.48 * (y2 - y1) - lift, 13, MUTED_RGB)
            r = int(min(cw / 12, 20))
            for k, (_, hx) in enumerate(colors):
                px = int(cx + (k - 1.5) * r * 2.6)
                py = int(y1 + 0.74 * (y2 - y1) - lift)
                cv2.circle(out, (px, py + 2), r, (215, 215, 218), -1, cv2.LINE_AA)
                cv2.circle(out, (px, py), r, bgr(hx), -1, cv2.LINE_AA)
            if i == hover and progress > 0:
                bw = (cw - 44) * min(progress, 1)
                cv2.line(out, (int(x1 + 22), int(y2 - 14 - lift)), (int(x1 + 22 + bw), int(y2 - 14 - lift)),
                         INK, 3, cv2.LINE_AA)
        if tip is not None:
            cv2.circle(out, (int(tip[0]), int(tip[1])), 11, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(out, (int(tip[0]), int(tip[1])), 11, INK, 2, cv2.LINE_AA)
        if hover is not None and progress >= 1:
            self.choose(hover, now)
        return out

    # ------------------------------------------------ screen 2: try the colors
    def _test(self, frame, hands, now):
        name, desc, colors = PALETTES[self.palette]
        face = self.face_s
        # which hand shows the swatches: right hand if visible, else any
        lm = None
        for label, pts in hands:
            if lm is None or label == "Right":
                lm = pts

        # thumb touches a finger -> try that color
        if lm is not None:
            size = hand_size(lm)
            dists = [np.linalg.norm(lm[THUMB_TIP] - lm[t]) / size for t in FINGER_TIPS]
            k = int(np.argmin(dists))
            if not self.pinching and dists[k] < PINCH_ON:
                self.pinching = True
                self.try_color(k, now)
            elif self.pinching and min(dists) > PINCH_OFF:
                self.pinching = False
        else:
            self.pinching = False

        # drape + photo
        amount = 0.0
        if self.active is not None:
            t = now - self.active_since
            amount = ease(t / 0.3)
            if self.captured_at is None and t >= DRAPE_BEFORE_PHOTO and face is not None:
                shot = self.drape.apply(frame, face, bgr(colors[self.active][1]), 1.0)
                self.photos[self.active] = self._portrait(shot, face, bgr(colors[self.active][1]))
                self.captured_at = now
                self.flash = 1.0
                if all(p is not None for p in self.photos):
                    self.done_at = now
            if self.captured_at is not None:
                amount *= 1 - ease((now - self.captured_at - 0.9) / 0.4)
                if now - self.captured_at > 1.3:
                    self.active = None
        out = frame
        if self.active is not None and face is not None:
            # the screen stays NEUTRAL grey while the photo is taken, so the screen's light on your face is the same
            # for every color; the real color is added to the photo afterwards (fair comparison, no color cast)
            out = self.drape.apply(frame, face, NEUTRAL, amount)
        else:
            out = frame.copy()

        # flash
        if self.flash > 0:
            cv2.addWeighted(out, 1 - self.flash * 0.8, np.full_like(out, 255), self.flash * 0.8, 0, out)
            self.flash = max(0.0, self.flash - 0.08)

        # header pill
        rounded_rect(out, self.w / 2 - 230, 18, self.w / 2 + 230, 92, 26, (252, 252, 253), 0.88)
        put_text(out, name, self.w / 2, 46, 26, INK_RGB, "serif")
        put_text(out, "touch your thumb to a finger to try that color", self.w / 2, 76, 14, MUTED_RGB)

        # countdown ring while the color is on
        if self.active is not None and self.captured_at is None and face is not None:
            t = (now - self.active_since) / DRAPE_BEFORE_PHOTO
            fw = np.linalg.norm(face[234] - face[454])
            c = face[FACE_OVAL].mean(axis=0)
            p = (int(c[0] - fw * 0.95), int(c[1]))
            cv2.circle(out, p, 18, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.ellipse(out, p, (18, 18), -90, 0, 360 * min(t, 1), (255, 255, 255), 3, cv2.LINE_AA)
            put_text(out, colors[self.active][0], p[0], p[1] + 36, 18, (255, 255, 255), "sans", 0.8)
            cv2.circle(out, (p[0], p[1] + 66), 13, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(out, (p[0], p[1] + 66), 11, bgr(colors[self.active][1]), -1, cv2.LINE_AA)
        if face is None:
            put_text(out, "move into the frame so I can see your face", self.w / 2, self.h * 0.5, 22,
                     (255, 255, 255), "sans", 0.9)

        # swatches on the fingertips
        if lm is not None and self.captured_at is None:
            size = hand_size(lm)
            for k, tip in enumerate(FINGER_TIPS):
                d = lm[tip] - lm[tip - 1]
                d = d / (np.linalg.norm(d) + 1e-6)
                p = lm[tip] + d * SWATCH_R * 1.2
                close = np.linalg.norm(lm[THUMB_TIP] - lm[tip]) / size
                r = int(SWATCH_R * (1 + 0.4 * float(np.clip((0.9 - close) / 0.6, 0, 1))))
                cv2.circle(out, (int(p[0]), int(p[1]) + 3), r + 3, (40, 40, 40), -1, cv2.LINE_AA)
                cv2.circle(out, (int(p[0]), int(p[1])), r + 3, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(out, (int(p[0]), int(p[1])), r, bgr(colors[k][1]), -1, cv2.LINE_AA)
                if self.photos[k] is not None:  # already tried: small check mark
                    q = (int(p[0] + r * 0.7), int(p[1] - r * 0.7))
                    cv2.circle(out, q, 8, INK, -1, cv2.LINE_AA)
                    cv2.line(out, (q[0] - 4, q[1]), (q[0] - 1, q[1] + 3), (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.line(out, (q[0] - 1, q[1] + 3), (q[0] + 4, q[1] - 3), (255, 255, 255), 2, cv2.LINE_AA)

        # photo tray on the right
        th = int((self.h - 120 - 50 - 3 * 14) / 4)
        tw = int(th * 0.75)
        x2 = self.w - 22
        for k in range(4):
            y1 = 120 + k * (th + 14)
            rounded_rect(out, x2 - tw - 6, y1 - 6, x2 + 6, y1 + th + 6, 14, (252, 252, 253), 0.9, shadow=0.15)
            if self.photos[k] is not None:
                thumb = cv2.resize(self.photos[k], (tw, th), interpolation=cv2.INTER_AREA)
                out[y1:y1 + th, x2 - tw:x2] = thumb
            else:
                # neutral slot with just a small dot of the color (big color blocks would tint your face)
                cv2.rectangle(out, (x2 - tw, y1), (x2, y1 + th), (232, 230, 228), -1)
                cv2.circle(out, (x2 - tw // 2, y1 + th // 2 - 10), 9, bgr(colors[k][1]), -1, cv2.LINE_AA)
                put_text(out, colors[k][0], x2 - tw / 2, y1 + th / 2 + 14, 12, MUTED_RGB, "sans")

        put_text(out, "SPACE  start over", 24, self.h - 24, 13, (255, 255, 255), "sans", 0.8, anchor="left")

        if self.done_at is not None and now - self.done_at > 1.6:
            self.mode = "compare"
        return out

    def _portrait(self, img, face, fill):
        """Crop a 3:4 portrait around the face."""
        c = face[FACE_OVAL].mean(axis=0)
        fh = np.linalg.norm(face[10] - face[152])
        ch = int(fh * 2.5)
        cw = int(ch * 0.75)
        pad = max(ch, cw)
        big = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=fill)
        cx, cy = int(c[0]) + pad, int(c[1] + fh * 0.18) + pad
        crop = big[cy - ch // 2:cy - ch // 2 + ch, cx - cw // 2:cx - cw // 2 + cw]
        return cv2.resize(crop, (480, 640), interpolation=cv2.INTER_AREA)

    # ------------------------------------------------ screen 3: compare
    def _compare(self, now):
        name, desc, colors = PALETTES[self.palette]
        out = np.full((self.h, self.w, 3), BG, np.uint8)
        put_text(out, name, self.w / 2, 52, 38, INK_RGB, "serif")
        put_text(out, "which one makes your skin glow?", self.w / 2, 92, 17, MUTED_RGB)
        mx, gap = 40, 22
        pw = (self.w - 2 * mx - 3 * gap) / 4
        ph = pw * 4 / 3
        maxh = self.h - 130 - 80
        if ph > maxh:
            ph, pw = maxh, maxh * 0.75
        total = 4 * pw + 3 * gap
        x0 = (self.w - total) / 2
        y0 = 122 + (maxh - ph) / 2
        for k in range(4):
            x = x0 + k * (pw + gap)
            rounded_rect(out, x, y0, x + pw, y0 + ph, 18, (255, 255, 255), 1.0, shadow=0.18)
            img = cv2.resize(self.photos[k], (int(pw), int(ph)), interpolation=cv2.INTER_AREA)
            m = np.zeros((int(ph), int(pw)), np.uint8)
            rounded_rect_mask(m, 16)
            a = (m.astype(np.float32) / 255)[..., None]
            roi = out[int(y0):int(y0) + int(ph), int(x):int(x) + int(pw)]
            roi[:] = (roi * (1 - a) + img * a).astype(np.uint8)
            cy = y0 + ph + 30
            cv2.circle(out, (int(x + 16), int(cy)), 9, bgr(colors[k][1]), -1, cv2.LINE_AA)
            put_text(out, colors[k][0], x + 32, cy, 18, INK_RGB, "sans", anchor="left")
            put_text(out, colors[k][1].upper(), x + pw, cy, 13, MUTED_RGB, "sans", anchor="right")
        put_text(out, "SPACE  try another color type", self.w / 2, self.h - 26, 15, MUTED_RGB)
        if not self.saved:
            os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
            fn = f"{name.lower().replace(' ', '-')}-{time.strftime('%Y%m%d-%H%M%S')}.png"
            cv2.imwrite(os.path.join(HERE, "results", fn), out)
            print("Saved results/" + fn)
            self.saved = True
        return out


def rounded_rect_mask(m, r):
    h, w = m.shape
    cv2.rectangle(m, (r, 0), (w - r, h), 255, -1)
    cv2.rectangle(m, (0, r), (w, h - r), 255, -1)
    for cx, cy in ((r, r), (w - r - 1, r), (r, h - r - 1), (w - r - 1, h - r - 1)):
        cv2.circle(m, (cx, cy), r, 255, -1, cv2.LINE_AA)


CAMERA_FILE = os.path.join(HERE, ".camera")


def save_camera(index):
    try:
        with open(CAMERA_FILE, "w") as f:
            f.write(str(index))
    except OSError:
        pass


def pick_camera():
    """Your last choice; otherwise, on a Mac, the built-in FaceTime camera (not the iPhone)."""
    try:
        with open(CAMERA_FILE) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        pass
    if sys.platform == "darwin":
        try:
            import json
            import subprocess
            info = subprocess.run(["system_profiler", "SPCameraDataType", "-json"], capture_output=True,
                                  text=True, timeout=8).stdout
            names = [c.get("_name", "") for c in json.loads(info).get("SPCameraDataType", [])]
            for i, name in enumerate(names):
                if any(k in name.lower() for k in ("facetime", "built-in", "macbook")):
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
    hands_lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=hand_model),
        running_mode=vision.RunningMode.VIDEO, num_hands=2))
    face_lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=face_model),
        running_mode=vision.RunningMode.VIDEO, num_faces=1))

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
    app = ColorTest(w, h)

    win = "Personal Color Test"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, w, h)
    fullscreen = False
    start, last_ts = time.time(), -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        now = time.time()
        ts = max(int((now - start) * 1000), last_ts + 1)
        last_ts = ts
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        face = None
        hands = []
        if app.mode != "compare":
            fr = face_lm.detect_for_video(mp_img, ts)
            if fr.face_landmarks:
                face = np.array([[p.x * w, p.y * h] for p in fr.face_landmarks[0]], np.float32)
            hr = hands_lm.detect_for_video(mp_img, ts)
            for lms, handed in zip(hr.hand_landmarks, hr.handedness):
                hands.append((handed[0].category_name,
                              np.array([[p.x * w, p.y * h] for p in lms], np.float32)))

        out = app.step(frame, face, hands, now)
        cv2.imshow(win, out)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            app.restart()
        elif ord("1") <= key <= ord("8"):
            n = key - ord("1")
            if app.mode == "pick":
                app.choose(n, now)
            elif app.mode == "test" and n < 4:
                app.try_color(n, now)
        elif key == ord("v"):                     # switch camera (e.g. laptop vs iPhone); remembered
            for step_i in range(1, 5):
                nxt = (cam_index + step_i) % 5
                c2 = open_camera(nxt)
                if c2.read()[0]:
                    cap.release()
                    cap, cam_index = c2, nxt
                    save_camera(cam_index)
                    print(f"Using camera {cam_index} (remembered for next time)")
                    break
                c2.release()
        elif key == ord("p"):
            os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
            path = os.path.join(HERE, "results", time.strftime("screen-%Y%m%d-%H%M%S.png"))
            cv2.imwrite(path, out)
            print("Saved", path)
        elif key == ord("f"):
            fullscreen = not fullscreen
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    hands_lm.close()
    face_lm.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
