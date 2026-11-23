"""Air Throw — grab things in the air and throw them to your friends' phones and laptops.

On the Mac: your camera (with your beauty filter). Pinch thumb + index on a sticker, photo or card to grab it,
then flick your hand up toward a friend's name at the top of the screen to throw it to them.
Friends join by scanning the QR code (same Wi-Fi). Things they throw to you fly in and land on your screen.

Keys: S next sticker · P snap a photo · C your profile card · X remove last · V switch camera
      F fullscreen · Q quit
Put pictures in the my_photos/ folder and they appear on your screen, ready to throw.
"""
import json
import math
import os
import queue
import shutil
import ssl
import sys
import threading
import time
import urllib.request

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FOG = os.path.join(os.path.dirname(HERE), "christmas-fog-window")   # shared: beauty filter, stickers, fonts
sys.path.insert(0, FOG)

import text  # noqa: E402  (elegant fonts, from christmas-fog-window)
from server import Room  # noqa: E402

# ---------------------------------------------------------------- settings
CAMERA_INDEX = None          # None = remember your last choice (V to switch), or prefer the laptop camera
WIDTH, HEIGHT = 1280, 720
BEAUTY_FILTER = True         # Shiqi's Douyin-style filter on the camera
PINCH_ON, PINCH_OFF = 0.32, 0.48
THROW_SPEED = 900            # how fast (pixels/second) an upward flick must be to count as a throw
STICKER_W = 170
PHOTO_W = 210

PHOTOS_IN = os.path.join(HERE, "my_photos")      # drop pictures here to throw them
RECEIVED = os.path.join(HERE, "received")        # everything thrown to you is saved here
PROFILE = os.path.join(HERE, "profile.json")
MODELS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                            "hand_landmarker/float16/latest/hand_landmarker.task",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                            "face_landmarker/float16/latest/face_landmarker.task",
}
INK = (48, 52, 60)          # RGB for text
MUTED = (140, 129, 122)


# ---------------------------------------------------------------- helpers
def download_model(name):
    for folder in (HERE, FOG):                     # reuse the fog window's copy if it's there
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    path = os.path.join(HERE, name)
    print(f"Downloading {name} (first run only)...")
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    with urllib.request.urlopen(MODELS[name], context=ctx) as r, open(path, "wb") as f:
        f.write(r.read())
    return path


def paste(frame, rgba, cx, cy, opacity=1.0):
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


def transform(rgba, angle, scale):
    """Rotate + scale an RGBA image, with room for a soft shadow."""
    h, w = rgba.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    bw, bh = int(h * s + w * c) + 40, int(h * c + w * s) + 40
    M[0, 2] += bw / 2 - w / 2
    M[1, 2] += bh / 2 - h / 2
    return cv2.warpAffine(rgba, M, (bw, bh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))


def with_shadow(rgba, strength=0.28):
    a = rgba[..., 3].astype(np.float32) / 255
    sh = cv2.GaussianBlur(np.roll(a, (8, 3), (0, 1)), (0, 0), 7) * strength
    out = rgba.copy().astype(np.float32)
    out_a = a + sh * (1 - a)
    out[..., :3] = (rgba[..., :3] * a[..., None] + np.float32([40, 30, 25]) * (sh * (1 - a))[..., None]) / \
        np.maximum(out_a[..., None], 1e-4)
    out[..., 3] = out_a * 255
    return np.clip(out, 0, 255).astype(np.uint8)


def rounded_mask(h, w, r):
    m = np.zeros((h, w), np.uint8)
    cv2.rectangle(m, (r, 0), (w - r, h), 255, -1)
    cv2.rectangle(m, (0, r), (w, h - r), 255, -1)
    for cx, cy in ((r, r), (w - r - 1, r), (r, h - r - 1), (w - r - 1, h - r - 1)):
        cv2.circle(m, (cx, cy), r, 255, -1, cv2.LINE_AA)
    return m


def polaroid(bgr, width=PHOTO_W):
    s = width / bgr.shape[1]
    img = cv2.resize(bgr, (width, int(bgr.shape[0] * s)), interpolation=cv2.INTER_AREA)
    pad, bottom = 9, 34
    out = np.zeros((img.shape[0] + pad + bottom, width + 2 * pad, 4), np.uint8)
    out[..., :3] = (246, 250, 253)
    out[..., 3] = 255
    out[pad:pad + img.shape[0], pad:pad + width, :3] = img
    return out


def stamp(canvas, rgba, cx, cy):
    """Paste an RGBA image onto an RGBA canvas (for building cards)."""
    h, w = rgba.shape[:2]
    H, W = canvas.shape[:2]
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    x1, y1, x2, y2 = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if x1 >= x2 or y1 >= y2:
        return
    src = rgba[y1 - y0:y2 - y0, x1 - x0:x2 - x0]
    a = src[..., 3:].astype(np.float32) / 255
    roi = canvas[y1:y2, x1:x2]
    roi[..., :3] = (roi[..., :3] * (1 - a) + src[..., :3] * a).astype(np.uint8)
    roi[..., 3] = np.maximum(roi[..., 3], src[..., 3])


def render_card(card, photo_bgr=None):
    """Your profile card, drawn like the one on the phones."""
    W, H = 280, 300
    out = np.zeros((H, W, 4), np.uint8)
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    out[..., :3] = np.clip(np.float32([251, 253, 255]) * (1 - yy) + np.float32([234, 239, 247]) * yy, 0, 255)
    out[..., 3] = rounded_mask(H, W, 22)
    c = (W // 2, 70)
    if photo_bgr is not None:
        ph = cv2.resize(photo_bgr, (88, 88), interpolation=cv2.INTER_AREA)
        m = np.zeros((88, 88), np.uint8)
        cv2.circle(m, (44, 44), 43, 255, -1, cv2.LINE_AA)
        rgba = np.dstack([ph, m])
        stamp(out, rgba, *c)
    else:
        cv2.circle(out, c, 44, (207, 216, 233, 255), -1, cv2.LINE_AA)
        ini = text.render((card.get("name") or "?")[0], 44, (255, 255, 255), "script", 0)
        if ini is not None:
            stamp(out, ini, c[0], c[1] + 2)
    y = 140
    for line, size, col, style in [(card.get("name", ""), 40, INK, "script")] + \
            [(t, 15, MUTED, "sans") for t in (card.get("instagram") and "@" + card["instagram"],
                                                card.get("xiaohongshu") and "xiaohongshu  " + card["xiaohongshu"]) if t] + \
            [("tap to connect", 11, (180, 167, 159), "sans")]:
        img = text.render(line, size, col, style, 0)
        if img is not None:
            stamp(out, img, W // 2, y + img.shape[0] // 2)
            y += img.shape[0] - (4 if style == "script" else 6)
    return out


def load_profile():
    if not os.path.exists(PROFILE):
        with open(PROFILE, "w") as f:
            json.dump({"name": "Shiqi", "instagram": "shiqi.017", "xiaohongshu": "", "linkedin": "", "phone": "",
                       "photo": "me.jpg"}, f, indent=2)
    with open(PROFILE) as f:
        return json.load(f)


# ---------------------------------------------------------------- things on screen
class Thing:
    def __init__(self, rgba, item, x, y, angle=0.0):
        self.base = rgba
        self.item = item            # what gets sent to a friend
        self.x, self.y, self.angle, self.scale = x, y, angle, 1.0
        self.anim = None            # (start time, duration, from, to, angle from/to, on_done)
        self.cache = None
        self.held = False

    def radius(self):
        h, w = self.base.shape[:2]
        return max(h, w) * 0.5 * self.scale

    def animate(self, now, to, dur, angle_to=None, scale_to=None, done=None):
        self.anim = (now, dur, (self.x, self.y, self.angle, self.scale),
                     (to[0], to[1], self.angle if angle_to is None else angle_to,
                      self.scale if scale_to is None else scale_to), done)

    def update(self, now):
        if not self.anim:
            return
        t0, dur, a, b, done = self.anim
        k = min(1.0, (now - t0) / dur)
        e = 1 - (1 - k) ** 3
        self.x, self.y = a[0] + (b[0] - a[0]) * e, a[1] + (b[1] - a[1]) * e
        self.angle = a[2] + (b[2] - a[2]) * e
        self.scale = a[3] + (b[3] - a[3]) * e + math.sin(k * math.pi) * 0.08
        if k >= 1:
            self.anim = None
            if done:
                done(self)

    def draw(self, frame):
        key = (round(self.angle, 1), round(self.scale * (1.06 if self.held else 1), 2))
        if self.cache is None or self.cache[0] != key:
            self.cache = (key, transform(with_shadow(self.base), key[0], key[1]))
        paste(frame, self.cache[1], self.x, self.y)


# ---------------------------------------------------------------- camera + tracking thread
class Tracker(threading.Thread):
    def __init__(self, cap, w, h, hand_lm, face_lm, mp):
        super().__init__(daemon=True)
        self.cap, self.w, self.h, self.hand_lm, self.face_lm, self.mp = cap, w, h, hand_lm, face_lm, mp
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.latest, self.seq, self.running = None, 0, True
        self.t0, self.last_ts = time.time(), -1

    def run(self):
        while self.running:
            with self.lock:
                cap = self.cap
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            if frame.shape[:2] != (self.h, self.w):
                frame = cv2.resize(frame, (self.w, self.h))
            frame = cv2.flip(frame, 1)
            now = time.time()
            ts = max(int((now - self.t0) * 1000), self.last_ts + 1)
            self.last_ts = ts
            img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            hands = self.hand_lm.detect_for_video(img, ts)
            face = self.face_lm.detect_for_video(img, ts) if self.face_lm else None
            with self.cond:
                self.latest = (frame, hands, face, now)
                self.seq += 1
                self.cond.notify_all()

    def get(self, last_seq, timeout=1.0):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != last_seq or not self.running, timeout)
            return (self.seq, self.latest) if self.seq != last_seq else (last_seq, None)

    def stop(self):
        self.running = False
        self.join(timeout=2)


CAMERA_FILE = os.path.join(HERE, ".camera")


def pick_camera():
    try:
        with open(CAMERA_FILE) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        pass
    if sys.platform == "darwin":
        try:
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


# ---------------------------------------------------------------- the app
class AirThrow:
    def __init__(self, w, h, room):
        self.w, self.h, self.room = w, h, room
        self.things = []
        self.held = {}              # hand label -> (thing, offset, recent positions)
        self.pinch = {}             # hand label -> bool
        self.flash = 0.0
        self.toast = ("", 0.0)
        self.profile = load_profile()
        self.stickers = self._load_stickers()
        self.next_sticker = 0
        self.sticker_urls = {}
        self.seen_photos = set()
        self.facefx = None
        if BEAUTY_FILTER:
            try:
                from facefx import FaceFX
                self.facefx = FaceFX()
                self.facefx.acc = 0          # beauty only, no hats
            except Exception as e:
                print("Beauty filter unavailable:", e)
        os.makedirs(PHOTOS_IN, exist_ok=True)
        os.makedirs(RECEIVED, exist_ok=True)
        self.join_card = self._join_card()
        self.chips = []

    # ---------- content
    def _load_stickers(self):
        folder = os.path.join(FOG, "my_stickers")
        out = []
        if os.path.isdir(folder):
            for fn in sorted(os.listdir(folder)):
                if fn.lower().endswith(".png"):
                    img = cv2.imread(os.path.join(folder, fn), cv2.IMREAD_UNCHANGED)
                    if img is not None and img.ndim == 3 and img.shape[2] == 4:
                        s = STICKER_W / max(img.shape[:2])
                        out.append((fn, cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)))
        return out

    def _join_card(self):
        import qrcode
        q = qrcode.QRCode(border=1, box_size=4)
        q.add_data(self.room.url)
        q.make(fit=True)
        qr = np.array(q.make_image(fill_color=(59, 52, 48), back_color=(255, 253, 250)).convert("RGB"))[..., ::-1]
        qr = cv2.resize(qr, (112, 112), interpolation=cv2.INTER_NEAREST)
        W, H = 300, 136
        card = np.zeros((H, W, 4), np.uint8)
        card[..., :3] = (250, 252, 255)
        card[..., 3] = (rounded_mask(H, W, 18) * 0.9).astype(np.uint8)
        card[12:124, 12:124, :3] = qr
        for line, size, col, style, y in (("join me", 30, INK, "script", 36),
                                          ("scan with your camera", 12, MUTED, "sans", 66),
                                          (self.room.url.replace("http://", ""), 12, MUTED, "sans", 88),
                                          ("same Wi-Fi", 11, (180, 167, 159), "sans", 108)):
            img = text.render(line, size, col, style, 0)
            if img is not None:
                stamp(card, img, 212, y)
        return card

    def add(self, rgba, item, x=None, y=None, now=None):
        x = self.w * 0.5 if x is None else x
        y = self.h * 0.55 if y is None else y
        t = Thing(rgba, item, x, self.h + rgba.shape[0], np.random.uniform(-8, 8))
        t.animate(now or time.time(), (x, y), 0.6, angle_to=np.random.uniform(-8, 8))
        self.things.append(t)
        return t

    def add_sticker(self, now):
        if not self.stickers:
            self.say("no stickers found", now)
            return
        name, img = self.stickers[self.next_sticker % len(self.stickers)]
        self.next_sticker += 1
        if name not in self.sticker_urls:
            self.sticker_urls[name] = self.room.save_item_image(cv2.imencode(".png", img)[1].tobytes())
        self.add(img, {"kind": "image", "url": self.sticker_urls[name], "w": 160},
                 self.w * np.random.uniform(0.3, 0.7), self.h * np.random.uniform(0.45, 0.65), now)

    def add_photo(self, bgr, now, x=None, y=None):
        small = cv2.resize(bgr, None, fx=min(1, 1400 / max(bgr.shape[:2])), fy=min(1, 1400 / max(bgr.shape[:2])))
        url = self.room.save_item_image(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes(), "jpg")
        self.add(polaroid(bgr), {"kind": "image", "url": url, "polaroid": True, "w": 200}, x, y, now)

    def add_card(self, now):
        p = dict(self.profile)
        photo = None
        path = os.path.join(HERE, p.get("photo") or "")
        if p.get("photo") and os.path.isfile(path):
            photo = cv2.imread(path)
            if photo is not None:
                h, w = photo.shape[:2]
                s = min(h, w)
                photo = photo[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
                if "photo_url" not in self.__dict__:
                    self.photo_url = self.room.save_item_image(cv2.imencode(".jpg", cv2.resize(photo, (300, 300)))[1].tobytes(), "jpg")
                p["photo"] = self.photo_url
        else:
            p["photo"] = ""
        card = render_card(p, photo)
        self.add(cv2.resize(card, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA), {"kind": "card", "card": p},
                 None, None, now)

    def check_photo_folder(self, now):
        for fn in sorted(os.listdir(PHOTOS_IN)):
            path = os.path.join(PHOTOS_IN, fn)
            if fn in self.seen_photos or not fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue
            self.seen_photos.add(fn)
            img = cv2.imread(path)
            if img is not None:
                self.add_photo(img, now, self.w * np.random.uniform(0.3, 0.7), self.h * np.random.uniform(0.45, 0.65))

    def say(self, msg, now):
        self.toast = (msg, now)

    # ---------- receiving
    def receive(self, now):
        while True:
            try:
                sender_id, sender, item = self.room.inbox.get_nowait()
            except queue.Empty:
                return
            rgba = None
            if item.get("kind") == "card":
                card = dict(item.get("card") or {})
                photo = self._local(card.get("photo", ""))
                rgba = cv2.resize(render_card(card, cv2.imread(photo) if photo else None), None, fx=0.75, fy=0.75,
                                  interpolation=cv2.INTER_AREA)
            else:
                path = self._local(item.get("url", ""))
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED) if path else None
                if img is not None:
                    if img.ndim == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    stamp_name = time.strftime("%Y%m%d-%H%M%S") + f"-from-{sender}" + os.path.splitext(path)[1]
                    shutil.copy(path, os.path.join(RECEIVED, stamp_name.replace("/", "_")))
                    if item.get("polaroid") or img.shape[2] == 3:
                        rgba = polaroid(img[..., :3])
                    else:
                        s = STICKER_W / max(img.shape[:2])
                        rgba = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            if rgba is None:
                continue
            chip = next((c for c in self.chips if c[0] == sender_id), None)
            sx = chip[1] if chip else self.w / 2
            t = Thing(rgba, item, sx, -rgba.shape[0], np.random.uniform(-30, 30))
            t.animate(now, (self.w * np.random.uniform(0.3, 0.7), self.h * np.random.uniform(0.4, 0.62)), 0.9,
                      angle_to=np.random.uniform(-8, 8))
            self.things.append(t)
            self.say(f"from {sender}", now)

    @staticmethod
    def _local(url):
        if url.startswith("/items/"):
            path = os.path.join(HERE, "items", os.path.basename(url))
            return path if os.path.isfile(path) else ""
        return ""

    # ---------- hands: grab, move, throw
    def update(self, hands, now):
        seen = set()
        for label, lm in hands:
            if label in seen:
                continue
            seen.add(label)
            size = np.linalg.norm(lm[0] - lm[9]) + 1e-6
            d = np.linalg.norm(lm[4] - lm[8]) / size
            p = (lm[4] + lm[8]) / 2
            was = self.pinch.get(label, False)
            pinched = d < (PINCH_OFF if was else PINCH_ON)
            self.pinch[label] = pinched
            if pinched and not was:
                for t in reversed(self.things):
                    if t.anim is None and math.hypot(t.x - p[0], t.y - p[1]) < t.radius() * 0.9:
                        self.things.remove(t)
                        self.things.append(t)
                        t.held = True
                        self.held[label] = (t, (t.x - p[0], t.y - p[1]), [(now, p[0], p[1])])
                        break
            elif pinched and label in self.held:
                t, off, hist = self.held[label]
                t.x, t.y = p[0] + off[0], p[1] + off[1]
                hist.append((now, p[0], p[1]))
                del hist[:-8]
            elif not pinched and label in self.held:
                self._release(label, now)
        for label in list(self.held):
            if label not in seen:
                self._release(label, now, throw=False)
        for t in self.things:
            t.update(now)

    def _release(self, label, now, throw=True):
        t, off, hist = self.held.pop(label)
        t.held = False
        if not throw or len(hist) < 3:
            return
        recent = [h for h in hist if now - h[0] < 0.18] or hist[-3:]
        (t0, x0, y0), (t1, x1, y1) = recent[0], recent[-1]
        dt = max(t1 - t0, 1e-3)
        vx, vy = (x1 - x0) / dt, (y1 - y0) / dt
        if vy > -THROW_SPEED or not self.chips:
            return
        # follow the throw up to the names row and pick whoever is closest
        row_y = 46
        k = (row_y - t.y) / vy
        x_at = t.x + vx * k
        target = min(self.chips, key=lambda c: abs(c[1] - x_at))
        tid, tx, name = target[0], target[1], target[2]

        def sent(th, tid=tid, vx=vx, vy=vy, name=name):
            if th in self.things:
                self.things.remove(th)
            self.room.throw_from_host(tid, th.item, vx, vy, vx / 40)
        t.animate(now, (tx, -t.base.shape[0]), 0.35, angle_to=t.angle + vx / 30, scale_to=0.7, done=sent)
        self.say(f"to {name}", now)

    # ---------- drawing
    def draw(self, frame, now, face_pts):
        if self.facefx is not None and face_pts is not None:
            frame = self.facefx.apply(frame, face_pts, now)
        # soft, warm, slightly dreamy backdrop so the things on top stand out
        out = cv2.addWeighted(frame, 0.82, np.full_like(frame, (236, 240, 247)), 0.18, 0)
        people = self.room.people()
        self.chips = self._draw_people(out, people)
        for t in self.things:
            t.draw(out)
        paste(out, self.join_card, 24 + self.join_card.shape[1] / 2, self.h - 24 - self.join_card.shape[0] / 2, 0.95)
        msg, at = self.toast
        if msg and now - at < 2.2:
            img = text.render(msg, 30, (255, 252, 246), "script", 0.8)
            if img is not None:
                paste(out, img, self.w / 2, self.h * 0.16, 1 - max(0.0, (now - at - 1.4) / 0.8))
        if self.flash > 0:
            cv2.addWeighted(out, 1 - self.flash, np.full_like(out, 255), self.flash, 0, out)
            self.flash = max(0.0, self.flash - 0.1)
        return out

    def _draw_people(self, out, people):
        if not people:
            img = text.render("waiting for friends to join…", 26, (255, 252, 246), "script", 0.8)
            if img is not None:
                paste(out, img, self.w / 2, 40)
            return []
        chips = []
        imgs = []
        for p in people:
            img = text.render(p["name"], 26, INK, "script", 0)
            imgs.append(img)
        widths = [(i.shape[1] if i is not None else 60) + 56 for i in imgs]
        gap = 14
        x = (self.w - sum(widths) - gap * (len(widths) - 1)) / 2
        for p, img, wd in zip(people, imgs, widths):
            H = 44
            chip = np.zeros((H, wd, 4), np.uint8)
            chip[..., :3] = (250, 252, 255)
            chip[..., 3] = (rounded_mask(H, wd, H // 2) * 0.88).astype(np.uint8)
            hexc = p.get("color", "#c98b8b").lstrip("#")
            col = (int(hexc[4:6], 16), int(hexc[2:4], 16), int(hexc[0:2], 16))
            cv2.circle(chip, (H // 2, H // 2), 15, (*col, 255), -1, cv2.LINE_AA)
            ini = text.render(p["name"][:1].upper() or "?", 14, (255, 255, 255), "sans", 0)
            if ini is not None:
                stamp(chip, ini, H // 2, H // 2)
            if img is not None:
                stamp(chip, img, H + (wd - H) // 2 - 6, H // 2 + 2)
            cx = x + wd / 2
            paste(out, chip, cx, 40)
            chips.append((p["id"], cx, p["name"]))
            x += wd + gap
        return chips


# ---------------------------------------------------------------- main loop
def main():
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    hand_lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=download_model("hand_landmarker.task")),
        running_mode=vision.RunningMode.VIDEO, num_hands=2))
    face_lm = None
    if BEAUTY_FILTER:
        face_lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=download_model("face_landmarker.task")),
            running_mode=vision.RunningMode.VIDEO, num_faces=1))

    profile = load_profile()
    room = Room(profile.get("name") or "Mac")
    url = room.start()
    print(f"\n  Friends join at {url}  (same Wi-Fi, or scan the QR code on screen)\n")

    cam_index = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else \
        (CAMERA_INDEX if CAMERA_INDEX is not None else pick_camera())
    cap = open_camera(cam_index)
    ok, frame = cap.read()
    if not ok:
        print("Couldn't open the camera. On a Mac, allow camera access for Terminal in "
              "System Settings > Privacy & Security > Camera, then try again.")
        sys.exit(1)
    h, w = frame.shape[:2]
    app = AirThrow(w, h, room)

    win = "Air Throw"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, w, h)
    fullscreen = False
    tracker = Tracker(cap, w, h, hand_lm, face_lm, mp)
    tracker.start()
    seq, last_check = 0, 0.0
    while True:
        seq, item = tracker.get(seq)
        if item is None:
            if not tracker.is_alive():
                break
            continue
        frame, hres, fres, now = item
        hands = [(hd[0].category_name, np.array([[p.x * w, p.y * h] for p in lms], np.float32))
                 for lms, hd in zip(hres.hand_landmarks, hres.handedness)]
        face_pts = None
        if fres is not None and fres.face_landmarks:
            face_pts = np.array([[p.x * w, p.y * h] for p in fres.face_landmarks[0]], np.float32)
        if now - last_check > 1.0:
            app.check_photo_folder(now)
            last_check = now
        app.receive(now)
        app.update(hands, now)
        out = app.draw(frame, now, face_pts)
        for label, lm in hands:
            p = (lm[4] + lm[8]) / 2
            on = app.pinch.get(label)
            cv2.circle(out, (int(p[0]), int(p[1])), 10 if on else 7, (255, 255, 255), 2 if on else 1, cv2.LINE_AA)
        cv2.imshow(win, out)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("s"):
            app.add_sticker(now)
        elif key == ord("p"):
            shot = app.facefx.apply(frame, face_pts, now) if (app.facefx is not None and face_pts is not None) else frame
            app.flash = 0.8
            app.add_photo(shot, now)
        elif key == ord("c"):
            app.add_card(now)
        elif key == ord("x") and app.things:
            app.things.pop()
        elif key == ord("v"):
            for step in range(1, 5):
                nxt = (cam_index + step) % 5
                c2 = open_camera(nxt)
                if c2.read()[0]:
                    with tracker.lock:
                        old, tracker.cap = tracker.cap, c2
                    old.release()
                    cam_index = nxt
                    with open(CAMERA_FILE, "w") as f:
                        f.write(str(nxt))
                    break
                c2.release()
        elif key == ord("f"):
            fullscreen = not fullscreen
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    tracker.stop()
    tracker.cap.release()
    hand_lm.close()
    if face_lm:
        face_lm.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
