"""Douyin-style beauty filter + Christmas face filters, all following MediaPipe's 478 face landmarks.

Three layers, applied in this order:
  1. Face shaping   - slimmer jaw, slightly bigger eyes set a touch lower, narrower nose wings, lower hairline
  2. Beauty skin    - airbrushed, brighter, pinkish-fair skin, softer nasolabial folds (法令纹), blemish removal
     + makeup       - champagne glitter shadow, brown wing + white liner, blush, freckles, snowflake/star
                      face stickers, nose shimmer, glossy gradient lips
  3. Accessories    - Santa hat, reindeer, tree hat, garland crown, earmuffs, stars, mustache, snowy hair
                      (switch with thumb + ring finger on the grab hand, or the E key)

Every strength is a setting below: 0 turns that effect off.
"""
import math

import cv2
import numpy as np

# ------------------------------------------------------------------ settings (0 = off)
FACE_SLIM = 0.07        # pull the jawline/cheeks inward
EYE_ENLARGE = 0.12      # make the eyes a bit bigger
EYE_LOWER = 0.012       # move the eyes down a little (fraction of face height)
NOSE_SLIM = 0.22        # narrow the nose wings (鼻翼)
HAIRLINE_LOWER = 0.045  # bring the hairline down (fraction of face height)
SMOOTH = 0.85           # skin airbrushing
BRIGHTEN = 14           # skin brightening
FOLDS = 0.8             # soften nasolabial folds (法令纹)
BLEMISH = True          # remove small spots and pimples
MAKEUP = True

# ------------------------------------------------------------------ landmarks
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
             152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
L_EYE = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
R_EYE = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]
L_UPPER = [33, 246, 161, 160, 159, 158, 157, 173, 133]
R_UPPER = [263, 466, 388, 387, 386, 385, 384, 398, 362]
L_LOWER = [33, 7, 163, 144, 145, 153, 154, 155, 133]
R_LOWER = [263, 249, 390, 373, 374, 380, 381, 382, 362]
L_BROW = [70, 63, 105, 66, 107]
R_BROW = [300, 293, 334, 296, 336]
LIPS_OUT = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
LIPS_IN = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
JAW_L = [132, 58, 172, 136, 150, 149, 176]
JAW_R = [361, 288, 397, 365, 379, 378, 400]
ALAE = [(64, 294), (129, 358), (98, 327)]

ACCESSORIES = ["none", "santa hat", "reindeer", "tree hat", "garland crown", "earmuffs", "starry",
               "mustache", "snowy hair"]


# ------------------------------------------------------------------ helpers
def _unit(v):
    return v / (np.linalg.norm(v) + 1e-6)


def _falloff(d2, R):
    return np.clip(1 - d2 / (R * R), 0, 1) ** 2


def _paste(out, rgba, cx, cy, opacity=1.0):
    h, w = rgba.shape[:2]
    H, W = out.shape[:2]
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    x1, y1, x2, y2 = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if x1 >= x2 or y1 >= y2:
        return
    src = rgba[y1 - y0:y2 - y0, x1 - x0:x2 - x0]
    a = src[..., 3:].astype(np.float32) / 255 * opacity
    roi = out[y1:y2, x1:x2].astype(np.float32)
    out[y1:y2, x1:x2] = (roi * (1 - a) + src[..., :3] * a).astype(np.uint8)


def _place(out, sprite, anchor_xy, sprite_anchor, width_px, angle_deg, opacity=1.0):
    """Scale a sprite to width_px, rotate it, and put its anchor point (fractions) at anchor_xy."""
    h, w = sprite.shape[:2]
    s = width_px / w
    if s <= 0.01:
        return
    M = cv2.getRotationMatrix2D((w * sprite_anchor[0], h * sprite_anchor[1]), angle_deg, s)
    size = int(max(w, h) * s * 2.2) + 4
    M[0, 2] += size / 2 - w * sprite_anchor[0]
    M[1, 2] += size / 2 - h * sprite_anchor[1]
    img = cv2.warpAffine(sprite, M, (size, size), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    _paste(out, img, anchor_xy[0], anchor_xy[1], opacity)


def _over(dst, src):
    """Alpha-composite src over dst (both BGRA uint8, same size)."""
    a = src[..., 3:].astype(np.float32) / 255
    da = dst[..., 3:].astype(np.float32) / 255
    out_a = a + da * (1 - a)
    rgb = (src[..., :3] * a + dst[..., :3] * da * (1 - a)) / np.maximum(out_a, 1e-4)
    res = np.zeros_like(dst)
    res[..., :3] = np.clip(rgb, 0, 255)
    res[..., 3] = np.clip(out_a[..., 0] * 255, 0, 255)
    return res


def _paste_rgba(canvas, sprite, cx, cy):
    h, w = sprite.shape[:2]
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    H, W = canvas.shape[:2]
    if x0 < 0 or y0 < 0 or x0 + w > W or y0 + h > H:
        return
    canvas[y0:y0 + h, x0:x0 + w] = _over(canvas[y0:y0 + h, x0:x0 + w], sprite)


def _sphere_rgba(R, base):
    """Glossy shaded ball (BGRA)."""
    n = int(R * 2 + 4)
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    nx, ny = (x - n / 2) / R, (y - n / 2) / R
    r2 = nx * nx + ny * ny
    nz = np.sqrt(np.clip(1 - r2, 0, 1))
    L = _unit(np.array([-0.5, -0.65, 0.6], np.float32))
    Hh = _unit(L + np.array([0, 0, 1], np.float32))
    diff = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, 1)
    spec = np.clip(nx * Hh[0] + ny * Hh[1] + nz * Hh[2], 0, 1) ** 70
    rgb = np.float32(base) * (0.25 + 0.9 * diff[..., None]) + spec[..., None] * 255
    rgb += ((1 - nz) ** 3)[..., None] * 60
    a = np.clip((1 - np.sqrt(r2)) * R, 0, 1)
    out = np.zeros((n, n, 4), np.uint8)
    out[..., :3] = np.clip(rgb, 0, 255)
    out[..., 3] = (a * 255).astype(np.uint8)
    return out


def _fur(mask, base, rng, fuzz=6.0, strands=0.35):
    """Fluffy plush texture with soft, fuzzy edges. mask: uint8 0/255."""
    h, w = mask.shape
    inside = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5)
    sd = inside - outside
    noise = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 1.2)
    streak = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), sigmaX=0.6, sigmaY=3.5)
    a = np.clip((sd + (noise - 0.5) * fuzz * 2 + 1) / 2.5, 0, 1)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    lit = 1.0 + 0.22 * ((0.5 - xx / w) * 0.6 + (0.5 - yy / h) * 0.8)   # soft light from the top-left
    depth = np.clip(inside / (fuzz * 4), 0, 1)
    shade = lit * (0.72 + 0.28 * depth) * (0.88 + strands * (streak - 0.5) + 0.14 * (noise - 0.5))
    rgb = np.float32(base) * shade[..., None]
    out = np.zeros((h, w, 4), np.uint8)
    out[..., :3] = np.clip(rgb, 0, 255)
    out[..., 3] = (a * 255).astype(np.uint8)
    return out


def _star_pts(c, r_out, r_in, n=5, rot=-math.pi / 2):
    return np.array([(c[0] + (r_out if i % 2 == 0 else r_in) * math.cos(rot + i * math.pi / n),
                      c[1] + (r_out if i % 2 == 0 else r_in) * math.sin(rot + i * math.pi / n))
                     for i in range(2 * n)], np.int32)


def _pine(canvas, path, rng, scale=1.0):
    """Layered pine needles along a path (dark & soft at the back, bright & crisp at the front)."""
    greens = [(25, 58, 18), (34, 78, 26), (45, 98, 36), (60, 120, 50), (82, 142, 72)]
    for gi, (per, L) in enumerate([(8, 26), (7, 22), (6, 19), (4, 16), (3, 12)]):
        lay = np.zeros_like(canvas)
        for (x, y) in path:
            for _ in range(per):
                a = rng.uniform(0, math.tau)
                ln = rng.uniform(0.6, 1.0) * L * scale
                ox, oy = rng.normal(0, 5 * scale), rng.normal(0, 5 * scale)
                p0 = (int(x + ox), int(y + oy))
                p1 = (int(x + ox + math.cos(a) * ln), int(y + oy + math.sin(a) * ln * 0.85))
                col = tuple(int(c + rng.integers(-8, 9)) for c in greens[gi]) + (255,)
                cv2.line(lay, p0, p1, col, max(1, int(2 * scale)), cv2.LINE_AA)
        if gi == 0:
            lay = cv2.GaussianBlur(lay, (0, 0), 1.2)
        canvas[:] = _over(canvas, lay)


# ------------------------------------------------------------------ accessory sprites (made once)
def make_santa_hat(rng):
    W, H = 640, 560
    hat = np.zeros((H, W, 4), np.uint8)
    cone = np.zeros((H, W), np.uint8)
    pts = np.array([(70, 450), (150, 250), (300, 110), (450, 70), (560, 120), (590, 230),
                    (560, 300), (520, 210), (460, 190), (440, 300), (530, 450)], np.int32)
    cv2.fillPoly(cone, [pts], 255, cv2.LINE_AA)
    cone = (cv2.GaussianBlur(cone, (0, 0), 3) > 127).astype(np.uint8) * 255
    red = _fur(cone, (40, 35, 200), rng, fuzz=5, strands=0.5)
    fold = np.zeros((H, W), np.float32)
    cv2.line(fold, (440, 300), (470, 180), 1.0, 30, cv2.LINE_AA)
    fold = cv2.GaussianBlur(fold, (0, 0), 18)
    red[..., :3] = (red[..., :3] * (1 - 0.45 * fold[..., None])).astype(np.uint8)
    hat = _over(hat, red)
    trim = np.zeros((H, W), np.uint8)
    cv2.ellipse(trim, (310, 455), (275, 62), -4, 0, 360, 255, -1, cv2.LINE_AA)
    hat = _over(hat, _fur(trim, (244, 244, 248), rng, fuzz=10, strands=0.2))
    pom = np.zeros((H, W), np.uint8)
    cv2.circle(pom, (560, 315), 58, 255, -1, cv2.LINE_AA)
    hat = _over(hat, _fur(pom, (246, 246, 250), rng, fuzz=12, strands=0.15))
    return hat, (0.48, 0.86)          # anchor: middle of the white trim


def make_antlers(rng):
    W, H = 900, 520
    img = np.zeros((H, W, 4), np.uint8)
    m = np.zeros((H, W), np.uint8)
    for s in (-1, 1):
        cx = W / 2 + s * 200
        top = (int(cx + s * 60), 130)
        cv2.line(m, (int(cx), 470), (int(cx + s * 25), 300), 255, 48, cv2.LINE_AA)
        cv2.line(m, (int(cx + s * 25), 300), top, 255, 42, cv2.LINE_AA)
        for (p, q, t) in (((cx + s * 20, 330), (cx + s * 150, 250), 36), ((cx + s * 45, 220), (cx - s * 60, 120), 34),
                          ((cx + s * 120, 270), (cx + s * 200, 170), 30), ((cx + s * 55, 170), (cx + s * 150, 90), 28)):
            cv2.line(m, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), 255, t, cv2.LINE_AA)
            cv2.circle(m, (int(q[0]), int(q[1])), t // 2, 255, -1, cv2.LINE_AA)
        cv2.circle(m, top, 21, 255, -1, cv2.LINE_AA)
    img = _over(img, _fur(m, (45, 80, 120), rng, fuzz=4, strands=0.4))
    band = np.zeros((H, W), np.uint8)
    cv2.ellipse(band, (W // 2, 640), (330, 200), 0, 200, 340, 255, 34, cv2.LINE_AA)
    img = _over(img, _fur(band, (40, 40, 205), rng, fuzz=2, strands=0.2))
    return img, (0.5, 0.86)


def make_tree_hat(rng):
    W, H = 640, 720
    img = np.zeros((H, W, 4), np.uint8)
    path = []
    for i in range(140):
        t = i / 139
        yy = 80 + t * 560
        half = 30 + t * 260
        for _ in range(3):
            path.append((W / 2 + rng.uniform(-half, half), yy + rng.normal(0, 6)))
    _pine(img, path, rng, 1.3)
    for k in range(5):  # gold garland swoops
        y0 = 180 + k * 95
        half = 30 + (y0 - 80) / 560 * 260
        pts = [(int(W / 2 - half + i * 2 * half / 30), int(y0 + 25 * math.sin(i / 30 * math.pi))) for i in range(31)]
        for p, q in zip(pts, pts[1:]):
            cv2.line(img, p, q, (60, 190, 235, 255), 6, cv2.LINE_AA)
    for _ in range(16):
        y = rng.uniform(150, 620)
        half = 30 + (y - 80) / 560 * 250
        x = W / 2 + rng.uniform(-half * 0.85, half * 0.85)
        col = [(45, 40, 210), (60, 190, 240), (210, 150, 60), (90, 180, 90)][int(rng.integers(0, 4))]
        _paste_rgba(img, _sphere_rgba(rng.uniform(13, 19), col), x, y)
    st = np.zeros((H, W, 4), np.uint8)
    cv2.fillPoly(st, [_star_pts((W // 2, 62), 52, 22)], (60, 200, 245, 255), cv2.LINE_AA)
    img = _over(img, st)
    return img, (0.5, 0.93)


def _bell(r):
    n = int(r * 3)
    m = np.zeros((n, n), np.uint8)
    c = n // 2
    pts = np.array([(c - r * 0.25, c - r * 0.9), (c + r * 0.25, c - r * 0.9), (c + r * 0.75, c + r * 0.4),
                    (c + r * 0.95, c + r * 0.75), (c - r * 0.95, c + r * 0.75), (c - r * 0.75, c + r * 0.4)], np.int32)
    cv2.fillPoly(m, [pts], 255, cv2.LINE_AA)
    cv2.ellipse(m, (c, int(c - r * 0.55)), (int(r * 0.45), int(r * 0.45)), 0, 180, 360, 255, -1, cv2.LINE_AA)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    shade = 0.6 + 0.6 * np.clip(1 - np.abs((xx - c + r * 0.3) / (r * 0.8)), 0, 1)
    out = np.zeros((n, n, 4), np.uint8)
    out[..., :3] = np.clip(np.float32([45, 165, 215]) * shade[..., None], 0, 255)
    out[..., 3] = m
    _paste_rgba(out, _sphere_rgba(r * 0.22, (40, 140, 190)), c, c + r * 0.85)
    return out


def make_garland_crown(rng, holly_img=None):
    W, H = 900, 380
    img = np.zeros((H, W, 4), np.uint8)
    path = [(W / 2 + math.cos(a) * 400, 330 - math.sin(a) * 150)
            for a in np.linspace(math.radians(10), math.radians(170), 130)]
    _pine(img, path, rng, 1.2)
    for (x, y) in (path[25], path[105]):
        _paste_rgba(img, _bell(34), x, y + 30)
    gx, gy = W // 2, 150
    box = np.zeros((110, 110, 4), np.uint8)
    box[..., 3] = 255
    yy, xx = np.mgrid[0:110, 0:110]
    box[..., :3] = (np.float32([45, 40, 200]) * (0.75 + 0.3 * (1 - xx / 110))[..., None]).astype(np.uint8)
    for _ in range(14):
        cv2.drawMarker(box, (int(rng.uniform(8, 100)), int(rng.uniform(8, 100))), (240, 240, 245, 255),
                       cv2.MARKER_STAR, 10, 1, cv2.LINE_AA)
    box[:, 48:62, :3] = (40, 110, 40)
    box[48:62, :, :3] = (40, 110, 40)
    _paste_rgba(img, box, gx, gy)
    if holly_img is not None:
        _paste_rgba(img, cv2.resize(holly_img, (150, 150), interpolation=cv2.INTER_AREA), gx + 110, gy + 10)
    for _ in range(14):
        p = path[int(rng.integers(5, len(path) - 5))]
        _paste_rgba(img, _sphere_rgba(8, (35, 30, 190)), p[0] + rng.normal(0, 8), p[1] + rng.normal(0, 8))
    return img, (0.5, 0.82)


def make_earmuff(rng):
    m = np.zeros((260, 260), np.uint8)
    cv2.ellipse(m, (130, 130), (95, 110), 0, 0, 360, 255, -1, cv2.LINE_AA)
    return _fur(m, (48, 46, 50), rng, fuzz=14, strands=0.4), (0.5, 0.5)


def make_mustache():
    W, H = 400, 160
    m = np.zeros((H, W), np.uint8)
    for s in (-1, 1):
        pts = []
        for t in np.linspace(0, 1, 40):
            pts.append((W / 2 + s * t * 170, 60 + 35 * math.sin(t * math.pi * 0.9) - 70 * max(0.0, t - 0.75)))
        pts = np.array(pts, np.int32)
        for i, (p, q) in enumerate(zip(pts, pts[1:])):
            cv2.line(m, tuple(int(v) for v in p), tuple(int(v) for v in q), 255, int(36 * (1 - i / 40) ** 0.8) + 4, cv2.LINE_AA)
        cv2.circle(m, tuple(int(v) for v in pts[-1]), 10, 255, -1, cv2.LINE_AA)
    out = np.zeros((H, W, 4), np.uint8)
    yy = np.arange(H, dtype=np.float32)[:, None]
    out[..., :3] = np.clip(np.float32([80, 45, 30]) * (0.8 + 0.4 * (1 - yy / H))[..., None], 0, 255)
    out[..., 3] = m
    return out, (0.5, 0.3)


# ------------------------------------------------------------------ the filter
class FaceFX:
    def __init__(self, seed=21):
        rng = np.random.default_rng(seed)
        self.acc = 1                     # start with the Santa hat
        self.changed_at = -10.0
        self.makeup_on = MAKEUP
        self.shape_on = True
        try:
            from stickers import holly
            hol = holly()
        except Exception:
            hol = None
        self.sprites = {
            "santa hat": make_santa_hat(rng),
            "reindeer": make_antlers(rng),
            "tree hat": make_tree_hat(rng),
            "garland crown": make_garland_crown(rng, hol),
            "earmuffs": make_earmuff(rng),
            "mustache": make_mustache(),
        }
        self.red_nose = _sphere_rgba(60, (35, 30, 210))
        self.glitter = [(rng.random(), rng.random(), rng.uniform(0, 6.28)) for _ in range(60)]
        self.freckles = [(int(rng.choice([197, 195, 5, 6, 117, 346, 50, 280, 119, 348])), rng.normal(0, 0.04),
                          rng.normal(0, 0.028), rng.uniform(0.0025, 0.0045)) for _ in range(46)]
        self.face_bits = [  # (landmark, dx, dy, kind, size) in face-width units
            (117, -0.07, 0.01, "flake", 0.032), (346, 0.07, 0.01, "flake", 0.032),
            (117, -0.02, -0.03, "gold", 0.016), (346, 0.03, -0.035, "red", 0.014),
            (50, -0.05, -0.02, "gold", 0.014), (280, 0.05, -0.01, "gold", 0.014),
            (6, 0.0, 0.05, "red", 0.012), (4, 0.0, -0.05, "white", 0.009),
            (105, 0.0, 0.02, "white", 0.01), (334, 0.02, 0.03, "white", 0.012),
            (334, 0.06, 0.06, "red", 0.012), (105, -0.06, 0.07, "white", 0.008),
            (119, 0.0, 0.01, "red", 0.006), (348, -0.02, 0.0, "white", 0.008),
        ]
        self.stars = [(rng.uniform(-0.95, 0.95), rng.uniform(-0.95, 0.75), rng.uniform(0.018, 0.035),
                       [(250, 200, 220), (220, 210, 250), (240, 230, 190), (200, 240, 220), (250, 220, 255)][i % 5],
                       rng.uniform(0, 6.28)) for i in range(26)]
        self.hair_flakes = [(int(rng.choice([10, 338, 297, 332, 109, 67, 103])), rng.normal(0, 0.14),
                             abs(rng.normal(0.07, 0.06)), rng.uniform(0.008, 0.02), rng.uniform(0, 6.28))
                            for _ in range(40)]

    # ---------------------------------------------------------- switching
    def next(self, now):
        self.acc = (self.acc + 1) % len(ACCESSORIES)
        self.changed_at = now

    @property
    def name(self):
        return ACCESSORIES[self.acc]

    # ---------------------------------------------------------- main entry
    def apply(self, frame, pts, now):
        if pts is None:
            return frame
        if self.shape_on:
            out, pts = self._reshape(frame, pts)
        else:
            out = frame.copy()
        fw = float(np.linalg.norm(pts[234] - pts[454])) + 1e-6
        self._skin(out, pts, fw)
        if self.makeup_on:
            self._makeup(out, pts, fw, now)
        self._accessory(out, pts, fw, now)
        return out

    # ---------------------------------------------------------- 1. face shaping
    def _field(self, X, Y, pts):
        """How far the content moves at points X, Y (a smooth sum of small local pushes)."""
        Dx = np.zeros_like(X)
        Dy = np.zeros_like(Y)
        fw = np.linalg.norm(pts[234] - pts[454]) + 1e-6
        fh = np.linalg.norm(pts[10] - pts[152]) + 1e-6
        up = _unit(pts[10] - pts[152])
        right = np.array([-up[1], up[0]])
        mid = (pts[168] + pts[152]) / 2

        def translate(c, v, R):
            w = _falloff((X - c[0]) ** 2 + (Y - c[1]) ** 2, R)
            Dx[...] += v[0] * w
            Dy[...] += v[1] * w

        def scale(c, s, R):
            w = _falloff((X - c[0]) ** 2 + (Y - c[1]) ** 2, R)
            Dx[...] += (X - c[0]) * s * w
            Dy[...] += (Y - c[1]) * s * w

        for eye in (L_EYE, R_EYE):
            c = pts[eye].mean(axis=0)
            ew = np.linalg.norm(pts[eye[0]] - pts[eye[8]])
            if EYE_ENLARGE:
                scale(c, EYE_ENLARGE, ew * 1.05)
            if EYE_LOWER:
                translate(c, -up * EYE_LOWER * fh, ew * 1.25)
        if FACE_SLIM:
            for i in JAW_L + JAW_R:
                side = np.dot(pts[i] - mid, right)
                translate(pts[i], -right * side * FACE_SLIM, fw * 0.2)
        if NOSE_SLIM:
            for a, b in ALAE:
                for i in (a, b):
                    side = np.dot(pts[i] - pts[1], right)
                    translate(pts[i], -right * side * NOSE_SLIM, fw * 0.065)
        if HAIRLINE_LOWER:
            translate(pts[10] + up * fh * 0.12, -up * HAIRLINE_LOWER * fh, fw * 0.42)
        return Dx, Dy

    def _reshape(self, frame, pts):
        H, W = frame.shape[:2]
        fw = np.linalg.norm(pts[234] - pts[454])
        x0, y0 = np.maximum(pts[FACE_OVAL].min(axis=0) - fw * 0.35, 0).astype(int)
        x1, y1 = np.minimum(pts[FACE_OVAL].max(axis=0) + fw * 0.35, [W, H]).astype(int)
        if x1 - x0 < 20 or y1 - y0 < 20:
            return frame.copy(), pts
        q = 2  # compute the warp at half resolution (it's smooth), then scale it up
        gw, gh = (x1 - x0) // q + 1, (y1 - y0) // q + 1
        X, Y = np.meshgrid(np.linspace(x0, x1 - 1, gw, dtype=np.float32),
                           np.linspace(y0, y1 - 1, gh, dtype=np.float32))
        Dx, Dy = self._field(X, Y, pts)
        mapx = cv2.resize(X - Dx, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        mapy = cv2.resize(Y - Dy, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        out = frame.copy()
        out[y0:y1, x0:x1] = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        px, py = pts[:, 0].astype(np.float32).copy(), pts[:, 1].astype(np.float32).copy()
        dx, dy = self._field(px, py, pts)
        return out, pts + np.stack([dx, dy], 1)

    # ---------------------------------------------------------- 2. skin + makeup
    def _roi(self, out, pts, pad):
        H, W = out.shape[:2]
        x0, y0 = np.maximum(pts[FACE_OVAL].min(axis=0) - pad, 0).astype(int)
        x1, y1 = np.minimum(pts[FACE_OVAL].max(axis=0) + pad, [W, H]).astype(int)
        return x0, y0, x1, y1

    def _skin_mask(self, shape, pts, fw, off):
        m = np.zeros(shape, np.uint8)
        cv2.fillPoly(m, [(pts[FACE_OVAL] - off).astype(np.int32)], 255, cv2.LINE_AA)
        for region, k in ((L_EYE, 1.35), (R_EYE, 1.35), (LIPS_OUT, 1.08)):
            poly = pts[region] - off
            c = poly.mean(axis=0)
            cv2.fillPoly(m, [(c + (poly - c) * k).astype(np.int32)], 0, cv2.LINE_AA)
        for brow in (L_BROW, R_BROW):
            cv2.polylines(m, [(pts[brow] - off).astype(np.int32)], False, 0, max(2, int(fw * 0.04)), cv2.LINE_AA)
        return m

    def _skin(self, out, pts, fw):
        x0, y0, x1, y1 = self._roi(out, pts, 8)
        if x1 - x0 < 20 or y1 - y0 < 20:
            return
        off = np.array([x0, y0])
        roi = out[y0:y1, x0:x1]
        mask_u8 = self._skin_mask(roi.shape[:2], pts, fw, off)
        rh, rw = roi.shape[:2]
        small = cv2.resize(roi, (rw // 2, rh // 2), interpolation=cv2.INTER_AREA)
        k = max(3, int(fw * 0.02) | 1)                               # neighbourhood size at half resolution
        if BLEMISH:   # small spots that are darker or redder than the skin around them
            g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            red = cv2.subtract(small[..., 2], small[..., 1])
            g_bg = cv2.medianBlur(g, k)
            red_bg = cv2.medianBlur(red, k)
            spot = (cv2.subtract(g_bg, g) > 9) | (cv2.subtract(red, red_bg) > 8)
            spot = spot & (cv2.resize(mask_u8, (rw // 2, rh // 2), interpolation=cv2.INTER_NEAREST) > 200)
            spot = cv2.dilate(spot.astype(np.uint8), np.ones((3, 3), np.uint8))
            nn, ll, ss, _ = cv2.connectedComponentsWithStats(spot)
            if nn > 1:
                big = np.zeros(nn, bool)
                big[1:] = ss[1:, cv2.CC_STAT_AREA] > (fw * 0.015) ** 2  # only small spots, not features
                spot[big[ll]] = 0
                if spot.any():
                    clean = cv2.resize(cv2.medianBlur(small, k), (rw, rh), interpolation=cv2.INTER_LINEAR)
                    sm = cv2.GaussianBlur(cv2.resize(spot.astype(np.float32), (rw, rh)), (0, 0), 1.5)[..., None]
                    roi[:] = (roi * (1 - sm) + clean * sm).astype(np.uint8)
                    small = cv2.resize(roi, (rw // 2, rh // 2), interpolation=cv2.INTER_AREA)
        # airbrush (edge-preserving smoothing at half size), then brighten to a pinkish fair tone
        smooth = cv2.bilateralFilter(small, 9, 45, 9)
        smooth = cv2.bilateralFilter(smooth, 7, 30, 7)
        smooth = cv2.resize(smooth, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        smooth = smooth * np.float32([1.04, 1.05, 1.08]) + BRIGHTEN
        m = cv2.resize(cv2.GaussianBlur(cv2.resize(mask_u8, (rw // 2, rh // 2)), (0, 0), max(1, fw * 0.015)),
                       (rw, rh)).astype(np.float32) / 255 * SMOOTH
        if FOLDS:   # nasolabial folds: extra smoothing + lift from the nose wing down to the mouth corner
            fold = np.zeros(roi.shape[:2], np.float32)
            up = _unit(pts[10] - pts[152])
            right = np.array([-up[1], up[0]])
            for ala, corner, s in ((129, 61, -1), (358, 291, 1)):
                a = pts[ala] - off + right * s * fw * 0.03
                b = pts[corner] - off + right * s * fw * 0.06 - up * fw * 0.02
                cv2.line(fold, tuple(a.astype(int)), tuple(b.astype(int)), 1.0, max(3, int(fw * 0.07)), cv2.LINE_AA)
            fold = cv2.GaussianBlur(fold, (0, 0), max(1, fw * 0.03)) * FOLDS
            very = cv2.resize(cv2.GaussianBlur(small, (0, 0), max(1, fw * 0.0125)), (rw, rh)).astype(np.float32) + 10
            smooth = smooth * (1 - fold[..., None]) + very * fold[..., None]
            m = np.maximum(m, fold * 0.9)
        roi[:] = np.clip(roi * (1 - m[..., None]) + smooth * m[..., None], 0, 255).astype(np.uint8)

    def _soft_tint(self, out, center, rx, ry, color, k, angle=0.0):
        H, W = out.shape[:2]
        R = int(max(rx, ry) * 2.4)
        x0, y0 = int(max(center[0] - R, 0)), int(max(center[1] - R, 0))
        x1, y1 = int(min(center[0] + R, W)), int(min(center[1] + R, H))
        if x1 <= x0 or y1 <= y0:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        ca, sa = math.cos(angle), math.sin(angle)
        u = (xx - center[0]) * ca + (yy - center[1]) * sa
        v = -(xx - center[0]) * sa + (yy - center[1]) * ca
        a = (np.exp(-((u / rx) ** 2 + (v / ry) ** 2)) * k)[..., None]
        roi = out[y0:y1, x0:x1].astype(np.float32)
        out[y0:y1, x0:x1] = np.clip(roi * (1 - a) + roi * np.float32(color) * a, 0, 255).astype(np.uint8)

    def _glow_spot(self, out, center, r, k, color=(255, 255, 255)):
        H, W = out.shape[:2]
        R = int(r * 2.5) + 1
        x0, y0 = int(max(center[0] - R, 0)), int(max(center[1] - R, 0))
        x1, y1 = int(min(center[0] + R, W)), int(min(center[1] + R, H))
        if x1 <= x0 or y1 <= y0:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        a = (np.exp(-(((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / (r * r + 1e-6))) * k)[..., None]
        roi = out[y0:y1, x0:x1].astype(np.float32)
        out[y0:y1, x0:x1] = np.clip(roi * (1 - a) + np.float32(color) * a, 0, 255).astype(np.uint8)

    @staticmethod
    def _sparkle(out, p, r, col=(255, 255, 255)):
        x, y = int(p[0]), int(p[1])
        r = max(1, int(r))
        cv2.line(out, (x - r, y), (x + r, y), col, 1, cv2.LINE_AA)
        cv2.line(out, (x, y - r), (x, y + r), col, 1, cv2.LINE_AA)
        cv2.circle(out, (x, y), max(1, r // 3), col, -1, cv2.LINE_AA)

    def _makeup(self, out, pts, fw, now):
        up = _unit(pts[10] - pts[152])
        right = np.array([-up[1], up[0]])
        roll = math.atan2(right[1], right[0])
        H, W = out.shape[:2]

        # blush: high on the cheeks, and a soft flush across the nose
        for i, j in ((117, 50), (346, 280)):
            self._soft_tint(out, pts[i] * 0.55 + pts[j] * 0.45, fw * 0.15, fw * 0.1, (0.82, 0.78, 1.06), 0.42, roll)
        self._soft_tint(out, pts[197], fw * 0.08, fw * 0.045, (0.86, 0.82, 1.04), 0.3, roll)

        # eyeshadow: champagne-pink wash on the lid + twinkling glitter
        glitter = out.copy()
        for upper in (L_UPPER, R_UPPER):
            lid = pts[upper]
            ew = np.linalg.norm(lid[0] - lid[-1]) + 1e-6
            top = lid + up * ew * 0.42 * np.sin(np.linspace(0.2, math.pi - 0.2, len(lid)))[:, None]
            poly = np.vstack([lid, top[::-1]]).astype(np.int32)
            bx, by, bw, bh = cv2.boundingRect(poly)
            pad = int(ew * 0.3) + 2
            x0, y0 = max(bx - pad, 0), max(by - pad, 0)
            x1, y1 = min(bx + bw + pad, W), min(by + bh + pad, H)
            if x1 <= x0 or y1 <= y0:
                continue
            m = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillPoly(m, [poly - np.array([x0, y0])], 255, cv2.LINE_AA)
            mm = cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), ew * 0.08)[..., None] * 0.5
            roi = out[y0:y1, x0:x1].astype(np.float32)
            shade = roi * np.float32([0.88, 0.86, 1.0]) + np.float32([10, 14, 22])
            out[y0:y1, x0:x1] = np.clip(roi * (1 - mm) + shade * mm, 0, 255).astype(np.uint8)
            for (u, v, ph) in self.glitter:
                kk = int(u * (len(lid) - 1))
                p = lid[kk] * (1 - v * 0.8) + top[kk] * v * 0.8
                b = 0.5 + 0.5 * math.sin(now * 3 + ph)
                if b > 0.35:
                    cv2.circle(glitter, (int(p[0]), int(p[1])), 1, (int(200 + 55 * b), int(220 + 35 * b), 255), -1, cv2.LINE_AA)
        cv2.addWeighted(glitter, 0.85, out, 0.15, 0, out)

        # liner: brown tapered wing on the upper lid, thin white wing under it, sparkle at the inner corner
        for upper, lower in ((L_UPPER, L_LOWER), (R_UPPER, R_LOWER)):
            lid = pts[upper]
            outer, inner = lid[0], lid[-1]
            ew = np.linalg.norm(outer - inner) + 1e-6
            out_dir = _unit(outer - inner)
            wing = outer + out_dir * ew * 0.28 + up * ew * 0.16
            line = list(lid[:5][::-1]) + [outer, wing]
            for i, (p, q) in enumerate(zip(line, line[1:])):
                th = max(1, int(ew * (0.035 + 0.02 * i / len(line))))
                cv2.line(out, tuple(p.astype(int)), tuple(q.astype(int)), (40, 40, 70), th, cv2.LINE_AA)
            low = pts[lower]
            w0 = low[2] * 0.5 + outer * 0.5 - up * ew * 0.03
            w1 = outer + out_dir * ew * 0.25 + up * ew * 0.06
            cv2.line(out, tuple(w0.astype(int)), tuple(w1.astype(int)), (250, 248, 252), 1, cv2.LINE_AA)
            self._glow_spot(out, inner + out_dir * ew * 0.08 - up * ew * 0.08, ew * 0.08, 0.35, (255, 245, 240))
            for k in range(3):
                if math.sin(now * 2.5 + k * 1.7 + float(outer[0])) > 0.2:
                    self._sparkle(out, low[5 + k % 3] - up * ew * 0.12, ew * 0.05, (230, 245, 255))

        # freckles
        layer = out.copy()
        for i, dx, dy, r in self.freckles:
            p = pts[i] + right * dx * fw - up * dy * fw
            x, y = int(np.clip(p[0], 0, W - 1)), int(np.clip(p[1], 0, H - 1))
            c = out[y, x].astype(np.float32) * np.float32([0.68, 0.7, 0.8])
            cv2.circle(layer, (x, y), max(1, int(r * fw)), tuple(int(v) for v in c), -1, cv2.LINE_AA)
        cv2.addWeighted(layer, 0.6, out, 0.4, 0, out)

        # nose-tip shimmer
        self._glow_spot(out, pts[4] - up * fw * 0.015, fw * 0.035, 0.35)
        if math.sin(now * 1.9) > 0:
            self._sparkle(out, pts[4] - up * fw * 0.02, fw * 0.015)

        # glossy gradient lips: berry in the middle, rosy at the edges, shine on the lower lip
        bx, by, bw, bh = cv2.boundingRect(pts[LIPS_OUT].astype(np.int32))
        pad = int(fw * 0.05) + 2
        x0, y0 = max(bx - pad, 0), max(by - pad, 0)
        x1, y1 = min(bx + bw + pad, W), min(by + bh + pad, H)
        if x1 > x0 and y1 > y0:
            off = np.array([x0, y0])
            mo = np.zeros((y1 - y0, x1 - x0), np.uint8)
            mi = np.zeros_like(mo)
            cv2.fillPoly(mo, [(pts[LIPS_OUT] - off).astype(np.int32)], 255, cv2.LINE_AA)
            cv2.fillPoly(mi, [(pts[LIPS_IN] - off).astype(np.int32)], 255, cv2.LINE_AA)
            lips = cv2.GaussianBlur(cv2.subtract(mo, mi).astype(np.float32) / 255, (0, 0), 1.2)[..., None]
            depth = cv2.GaussianBlur(cv2.dilate(mi, np.ones((5, 5), np.uint8)).astype(np.float32) / 255,
                                     (0, 0), max(1, fw * 0.03))[..., None]
            roi = out[y0:y1, x0:x1].astype(np.float32)
            tint = roi * np.float32([0.75, 0.72, 1.08]) * (1 - depth) + roi * np.float32([0.5, 0.45, 0.95]) * depth
            out[y0:y1, x0:x1] = np.clip(roi * (1 - lips * 0.75) + tint * lips * 0.75, 0, 255).astype(np.uint8)
            self._glow_spot(out, pts[14] * 0.45 + pts[17] * 0.55, fw * 0.03, 0.45)
            self._glow_spot(out, pts[0] * 0.6 + pts[13] * 0.4 + right * fw * 0.03, fw * 0.012, 0.35)

        # little face stickers: snowflakes, gold and red stars
        for i, dx, dy, kind, sz in self.face_bits:
            p = pts[i] + right * dx * fw - up * dy * fw
            r = sz * fw
            if kind == "flake":
                for a in range(6):
                    ang = roll + a * math.pi / 3
                    q = p + np.array([math.cos(ang), math.sin(ang)]) * r
                    cv2.line(out, tuple(p.astype(int)), tuple(q.astype(int)), (252, 250, 250), max(1, int(r * 0.22)), cv2.LINE_AA)
                cv2.circle(out, tuple(p.astype(int)), max(1, int(r * 0.25)), (255, 255, 255), -1, cv2.LINE_AA)
            else:
                col = {"gold": (70, 185, 230), "red": (50, 40, 200), "white": (250, 250, 252)}[kind]
                cv2.fillPoly(out, [_star_pts(p, r, r * 0.32, n=4, rot=roll)], col, cv2.LINE_AA)

    # ---------------------------------------------------------- 3. accessories
    def _accessory(self, out, pts, fw, now):
        name = ACCESSORIES[self.acc]
        if name == "none":
            return
        fh = float(np.linalg.norm(pts[10] - pts[152]))
        up = _unit(pts[10] - pts[152])
        right = np.array([-up[1], up[0]])
        roll = -math.degrees(math.atan2(right[1], right[0]))
        crown = pts[10] + up * fh * 0.1             # hats sit just above the hairline
        if name in ("santa hat", "reindeer", "tree hat", "garland crown"):
            spr, anc = self.sprites[name]
            width = {"santa hat": 1.6, "reindeer": 2.0, "tree hat": 1.55, "garland crown": 1.6}[name] * fw
            pos = crown + up * fh * {"santa hat": -0.06, "reindeer": 0.17, "tree hat": -0.04, "garland crown": 0.05}[name]
            if name == "santa hat":
                pos = pos + right * fw * 0.05
            _place(out, spr, pos, anc, width, roll)
            if name == "reindeer":
                _place(out, self.red_nose, pts[4] - up * fw * 0.01, (0.5, 0.5), fw * 0.22, roll)
        elif name == "earmuffs":
            spr, anc = self.sprites["earmuffs"]
            a, b = pts[234] - right * fw * 0.06, pts[454] + right * fw * 0.06
            top = pts[10] + up * fh * 0.3
            arc = [a * (1 - t) ** 2 + top * 2 * t * (1 - t) + b * t * t for t in np.linspace(0, 1, 30)]
            for p, q in zip(arc, arc[1:]):
                cv2.line(out, tuple(p.astype(int)), tuple(q.astype(int)), (40, 38, 42), max(4, int(fw * 0.05)), cv2.LINE_AA)
            for p in (a, b):
                _place(out, spr, p, anc, fw * 0.5, roll)
        elif name == "mustache":
            spr, anc = self.sprites["mustache"]
            _place(out, spr, pts[2] * 0.4 + pts[0] * 0.6, anc, fw * 0.55, roll)
        elif name == "starry":
            c = pts[168]
            for (sx, sy, sz, col, ph) in self.stars:
                p = c + right * sx * fw * 0.8 + up * (sy * 0.75 + 0.15) * fh
                r = sz * fw * (0.75 + 0.25 * math.sin(now * 2 + ph))
                cv2.fillPoly(out, [_star_pts(p, r, r * 0.45)], col, cv2.LINE_AA)
            for i, sgn in ((117, -1), (346, 1)):   # yellow stars on the cheeks
                for ox, oy, s in ((0.0, 0.0, 0.035), (0.05, 0.06, 0.025), (-0.03, -0.07, 0.022)):
                    p = pts[i] + right * (0.02 * sgn + ox * sgn) * fw + up * oy * fw
                    cv2.fillPoly(out, [_star_pts(p, s * fw, s * fw * 0.45)], (90, 225, 250), cv2.LINE_AA)
        elif name == "snowy hair":
            layer = out.copy()
            for i, dx, dy, r, ph in self.hair_flakes:
                p = pts[i] + right * dx * fw + up * dy * fw
                b = 0.85 + 0.15 * math.sin(now * 1.5 + ph)
                for a in range(6):
                    ang = a * math.pi / 3
                    q = p + np.array([math.cos(ang), math.sin(ang)]) * r * fw
                    cv2.line(layer, tuple(p.astype(int)), tuple(q.astype(int)), (int(255 * b),) * 3, 1, cv2.LINE_AA)
                cv2.circle(layer, tuple(p.astype(int)), max(1, int(r * fw * 0.3)), (250, 248, 245), -1, cv2.LINE_AA)
            cv2.addWeighted(layer, 0.85, out, 0.15, 0, out)
