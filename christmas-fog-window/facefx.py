"""Douyin-style beauty filter + Christmas face filters, all following MediaPipe's 478 face landmarks.

Three layers, applied in this order:
  1. Face shaping   - V-line jaw and chin, slightly bigger eyes set a touch lower, narrower nose wings, lower hairline
  2. Beauty skin    - airbrushed, brighter, slightly cooler (less yellow) skin, softer nasolabial folds (法令纹),
                      blemish removal
     + makeup       - natural: pink blush, rosy lips, softly defined brows and lash line, brighter under-eyes
  3. Accessories    - Santa hat, reindeer, tree hat, garland crown, earmuffs, stars, mustache, snowy hair
                      (switch with thumb + ring finger on the grab hand, or the E key)

Every strength is a setting below: 0 turns that effect off.
"""
import json
import math
import os

import cv2
import numpy as np

# ------------------------------------------------------------------ settings (tuned to Shiqi's face, to match her Douyin filter)
USE_MY_SHAPE = True     # reshape exactly like Shiqi's Douyin filter (measured, see my_face_shape.json)
SHAPE_STRENGTH = 1.0    # 1.0 = same as Douyin
# extra hand tweaks on top (0 = off)
FACE_SLIM = 0.0         # pull the lower jawline inward
CHIN_SLIM = 0.0         # narrower chin
EYE_ENLARGE = 0.0       # bigger eyes
EYE_LOWER = 0.0         # move eyes lower (fraction of face height)
NOSE_SLIM = 0.0         # narrower nose wings (鼻翼)
HAIRLINE_LOWER = 0.0    # lower hairline (fraction of face height)
SMOOTH = 0.75           # skin smoothing (evens out tone, keeps it clear)
DETAIL_KEEP = 0.8       # how much fine skin detail to keep (clearer, not blurry)
FOLDS = 0.8             # soften nasolabial folds (法令纹)
BLEMISH = True          # remove small spots and pimples
IRIS_ENLARGE = 0.14     # circle-lens effect: slightly bigger, softer irises
AEGYO_SAL = 0.0         # extra hand-drawn 卧蚕 (off: the learned Douyin makeup map already has her exact aegyo sal)
LASH_LINE = 0.0         # extra drawn lash line (off: the learned Douyin makeup already darkens the lash line, and a drawn line can look misaligned)
MAKEUP = True           # natural makeup: blush, lip tint, brows, lash line, brighter under-eyes

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
JAW_L = [172, 136, 150]
JAW_R = [397, 365, 379]
CHIN = [149, 176, 148, 152, 377, 400, 378]
L_BROW_FULL = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
R_BROW_FULL = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
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


# ------------------------------------------------------------------ measured face shape
class ShapeProfile:
    """Smooth warp that moves every face landmark the way Shiqi's Douyin filter does.

    my_face_shape.json holds, for each of the 468 landmarks, where it sits on a reference face and how far the
    Douyin filter moved it (both in eye-distance units, x along the eyes, y down the face). We fit a smooth
    radial-basis field to those movements once; each frame we just evaluate it in the current face's frame.
    """
    SIGMA = 0.55          # smoothness of the field (eye-distance units)

    def __init__(self, path):
        d = json.load(open(path))
        pos = np.array(d["ref_positions"], np.float32)
        disp = np.array(d["displacements"], np.float32)
        centers = pos[::3]
        ring = [(math.cos(a) * r, math.sin(a) * r + 0.6) for r in (2.8, 3.6) for a in np.linspace(0, math.tau, 20, endpoint=False)]
        ring = np.array(ring, np.float32)                       # far away: no movement
        self.centers = np.vstack([centers, ring])
        targets = np.vstack([pos, ring])
        values = np.vstack([disp, np.zeros_like(ring)])
        Phi = self._phi(targets, self.centers)
        lam = 1e-3
        self.w = np.linalg.solve(Phi.T @ Phi + lam * np.eye(len(self.centers)), Phi.T @ values).astype(np.float32)

    def _phi(self, X, C):
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        return np.exp(-d2 / (2 * self.SIGMA ** 2)).astype(np.float32)

    @staticmethod
    def frame(pts):
        le, re = pts[[33, 133]].mean(0), pts[[263, 362]].mean(0)
        iod = float(np.linalg.norm(re - le)) + 1e-6
        ex = (re - le) / iod
        ey = np.array([-ex[1], ex[0]], np.float32)
        return (le + re) / 2, ex, ey, iod

    # the field is the same for every frame in face coordinates, so we tabulate it once on a grid
    TAB_MIN, TAB_MAX, TAB_N = (-4.0, -3.4), (4.0, 4.6), 160

    def _table(self):
        if not hasattr(self, "_tab"):
            xs = np.linspace(self.TAB_MIN[0], self.TAB_MAX[0], self.TAB_N, dtype=np.float32)
            ys = np.linspace(self.TAB_MIN[1], self.TAB_MAX[1], self.TAB_N, dtype=np.float32)
            gx, gy = np.meshgrid(xs, ys)
            D = self._phi(np.stack([gx.ravel(), gy.ravel()], 1), self.centers) @ self.w
            self._tab = (D[:, 0].reshape(gx.shape).astype(np.float32), D[:, 1].reshape(gx.shape).astype(np.float32))
        return self._tab

    def field(self, X, Y, pts, strength=1.0):
        org, ex, ey, iod = self.frame(pts)
        dx, dy = X - org[0], Y - org[1]
        lx = (dx * ex[0] + dy * ex[1]) / iod
        ly = (dx * ey[0] + dy * ey[1]) / iod
        n = self.TAB_N - 1
        u = ((lx - self.TAB_MIN[0]) / (self.TAB_MAX[0] - self.TAB_MIN[0]) * n).astype(np.float32)
        v = ((ly - self.TAB_MIN[1]) / (self.TAB_MAX[1] - self.TAB_MIN[1]) * n).astype(np.float32)
        tx, ty = self._table()
        shape = np.shape(X)
        u2, v2 = u.reshape(-1, 1), v.reshape(-1, 1)
        Lx = cv2.remap(tx, u2, v2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0).reshape(shape)
        Ly = cv2.remap(ty, u2, v2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0).reshape(shape)
        s = iod * strength
        return (Lx * ex[0] + Ly * ey[0]) * s, (Lx * ex[1] + Ly * ey[1]) * s


# ------------------------------------------------------------------ learned makeup
class MakeupMap:
    """Shiqi's Douyin makeup, learned from her before/after video.

    my_makeup.png stores, for every spot of a reference face, how much Douyin multiplies the color there
    (blush gradient, lip tint, aegyo-sal highlight, brightening, soft shading). Each frame we line the
    reference face up with the live face using the landmarks and multiply the colors.
    """
    def __init__(self, png, meta):
        d = json.load(open(meta))
        img = cv2.imread(png, cv2.IMREAD_UNCHANGED)
        self.R = img.astype(np.float32) / d["scale"]
        self.ref = np.array(d["ref_landmarks"], np.float32)
        self.g = np.float32(d["global_ratio_bgr"])
        x = np.arange(256, dtype=np.float32)
        self.lut = np.stack([np.clip(x * gr, 0, 255) for gr in self.g], 1).astype(np.uint8).reshape(256, 1, 3)
        self.sub = np.array(sorted(set(range(0, 468, 6)) | set(FACE_OVAL) | {33, 133, 263, 362, 1, 13, 14, 61, 291}))
        le, re = self.ref[[33, 133]].mean(0), self.ref[[263, 362]].mean(0)
        self.ref_iod = float(np.linalg.norm(re - le))

    @staticmethod
    def _similarity(src, dst):
        """Best scale+rotation+shift mapping src points onto dst points."""
        ms, md = src.mean(0), dst.mean(0)
        a, b = src - ms, dst - md
        n = (a ** 2).sum()
        c = (a * b).sum() / n
        s = (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum() / n
        M = np.array([[c, -s], [s, c]], np.float32)
        return M, md - ms @ M.T

    def apply(self, out, pts):
        H, W = out.shape[:2]
        cur = pts[self.sub].astype(np.float32)
        ref = self.ref[self.sub]
        M, t = self._similarity(cur, ref)
        sim = cur @ M.T + t
        # smooth correction so every landmark lands exactly on its reference spot
        sig = self.ref_iod * 0.45
        d2 = ((sim[:, None] - sim[None]) ** 2).sum(-1)
        Phi = np.exp(-d2 / (2 * sig * sig))
        w = np.linalg.solve(Phi + 1e-2 * np.eye(len(sim)), ref - sim)
        fw = float(np.linalg.norm(pts[234] - pts[454]))
        x0, y0 = np.maximum(pts[:468].min(0) - fw * 0.45, 0).astype(int)
        x1, y1 = np.minimum(pts[:468].max(0) + fw * 0.45, [W, H]).astype(int)
        if x1 - x0 < 16 or y1 - y0 < 16:
            return
        q = max(6, int(fw / 28))                     # the map is smooth: line it up on a coarse grid
        gx, gy = np.meshgrid(np.arange(x0, x1 + q, q, dtype=np.float32), np.arange(y0, y1 + q, q, dtype=np.float32))
        g = np.stack([gx.ravel(), gy.ravel()], 1) @ M.T + t
        g = g + np.exp(-((g[:, None] - sim[None]) ** 2).sum(-1) / (2 * sig * sig)) @ w
        u = g[:, 0].reshape(gx.shape).astype(np.float32)
        v = g[:, 1].reshape(gx.shape).astype(np.float32)
        ratio = cv2.remap(self.R, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=tuple(float(c) for c in self.g))
        gh, gw = ratio.shape[:2]
        ratio = cv2.resize(ratio, ((gw - 1) * q, (gh - 1) * q), interpolation=cv2.INTER_LINEAR)[:y1 - y0, :x1 - x0]
        roi = out[y0:y1, x0:x1].astype(np.float32)   # (a copy, taken before the overall brightening)
        out[:] = cv2.LUT(out, self.lut)            # whole picture: Douyin's overall brightening
        out[y0:y1, x0:x1] = np.clip(roi * ratio, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ the filter
class FaceFX:
    def __init__(self, seed=21):
        rng = np.random.default_rng(seed)
        self.acc = 0                     # start with just the beauty filter (no accessory)
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
        prof = os.path.join(os.path.dirname(os.path.abspath(__file__)), "my_face_shape.json")
        self.profile = ShapeProfile(prof) if USE_MY_SHAPE and os.path.exists(prof) else None
        here = os.path.dirname(os.path.abspath(__file__))
        png, meta = os.path.join(here, "my_makeup.png"), os.path.join(here, "my_makeup.json")
        self.makeup_map = MakeupMap(png, meta) if os.path.exists(png) and os.path.exists(meta) else None
        x = np.arange(256, dtype=np.float32)
        lift = lambda gain, add: np.clip(x * gain + add * (1 - x / 255) ** 0.5, 0, 255)
        self.lut = np.stack([lift(1.04, 6), lift(1.05, 7), lift(1.03, 6)], axis=1).astype(np.uint8).reshape(256, 1, 3)
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
        lut = self.makeup_map.lut if self.makeup_map is not None else self.lut
        if pts is None:
            return cv2.LUT(frame, lut)
        if self.makeup_map is None:
            frame = cv2.LUT(frame, self.lut)      # whole picture: a little brighter and less yellow
        if self.shape_on:
            out, pts = self._reshape(frame, pts)
        else:
            out = frame.copy()
        fw = float(np.linalg.norm(pts[234] - pts[454])) + 1e-6
        self._skin(out, pts, fw)
        if self.makeup_on:
            if self.makeup_map is not None:
                self.makeup_map.apply(out, pts)
            else:
                self._makeup(out, pts, fw, now)
            self._eyes(out, pts)
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

        if self.profile is not None:
            pDx, pDy = self.profile.field(X, Y, pts, SHAPE_STRENGTH)
            Dx += pDx
            Dy += pDy

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
        if CHIN_SLIM:
            for i in CHIN:
                side = np.dot(pts[i] - mid, right)
                translate(pts[i], -right * side * CHIN_SLIM + up * fh * 0.015 * CHIN_SLIM / 0.12, fw * 0.16)
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
        x0, y0 = np.maximum(pts[FACE_OVAL].min(axis=0) - fw * 0.8, 0).astype(int)
        x1, y1 = np.minimum(pts[FACE_OVAL].max(axis=0) + fw * 0.8, [W, H]).astype(int)
        if x1 - x0 < 20 or y1 - y0 < 20:
            return frame.copy(), pts
        q = max(3, int(fw / 60))   # compute the warp at low resolution (it's smooth), then scale it up
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
        x0, y0, x1, y1 = self._roi(out, pts, int(fw * 0.2))   # room for the soft edge, so there's no seam
        if x1 - x0 < 20 or y1 - y0 < 20:
            return
        off = np.array([x0, y0])
        roi = out[y0:y1, x0:x1]
        mask_u8 = self._skin_mask(roi.shape[:2], pts, fw, off)
        rh, rw = roi.shape[:2]
        s = min(0.5, 220.0 / fw)                                     # work at a fixed size: face ~220 px wide
        sw, sh = max(8, int(rw * s)), max(8, int(rh * s))
        small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_AREA)
        fws = fw * s
        k = max(3, int(fws * 0.04) | 1)
        if BLEMISH:   # small spots that are darker or redder than the skin around them
            g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            red = cv2.subtract(small[..., 2], small[..., 1])
            g_bg = cv2.medianBlur(g, k)
            red_bg = cv2.medianBlur(red, k)
            spot = (cv2.subtract(g_bg, g) > 6) | (cv2.subtract(red, red_bg) > 5)
            spot = spot & (cv2.resize(mask_u8, (sw, sh), interpolation=cv2.INTER_NEAREST) > 200)
            spot = cv2.dilate(spot.astype(np.uint8), np.ones((3, 3), np.uint8))
            nn, ll, ss, _ = cv2.connectedComponentsWithStats(spot)
            if nn > 1:
                big = np.zeros(nn, bool)
                big[1:] = ss[1:, cv2.CC_STAT_AREA] > (fws * 0.04) ** 2  # only small spots, not features
                spot[big[ll]] = 0
                if spot.any():
                    clean = cv2.resize(cv2.medianBlur(small, k), (rw, rh), interpolation=cv2.INTER_LINEAR)
                    sm = cv2.GaussianBlur(cv2.resize(spot.astype(np.float32), (rw, rh)), (0, 0), 1.5)[..., None]
                    roi[:] = (roi * (1 - sm) + clean * sm).astype(np.uint8)
                    small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_AREA)
        # airbrush (edge-preserving smoothing at half size), then brighten to a pinkish fair tone
        smooth = cv2.bilateralFilter(small, 9, 45, 9)
        smooth = cv2.bilateralFilter(smooth, 7, 30, 7)
        smooth = cv2.resize(smooth, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        # keep the fine detail (pores, lashes, hair) so the skin stays clear, only even out the unevenness
        rf = roi.astype(np.float32)
        detail = rf - cv2.GaussianBlur(rf, (0, 0), max(0.8, fw * 0.004))
        smooth = smooth + detail * DETAIL_KEEP
        m = cv2.resize(cv2.GaussianBlur(cv2.resize(mask_u8, (sw, sh)), (0, 0), max(1, fws * 0.06)),
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
            very = cv2.resize(cv2.GaussianBlur(small, (0, 0), max(1, fws * 0.025)), (rw, rh)).astype(np.float32) + 6
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
        """Natural Douyin-style makeup: pink blush, rosy lips, softly defined brows and lash line, bright under-eyes."""
        up = _unit(pts[10] - pts[152])
        right = np.array([-up[1], up[0]])
        roll = math.atan2(right[1], right[0])
        H, W = out.shape[:2]

        # blush on the apples of the cheeks, a little toward the outside
        for i, j, s in ((117, 50, -1), (346, 280, 1)):
            c = pts[i] * 0.45 + pts[j] * 0.55 + right * s * fw * 0.03
            self._soft_tint(out, c, fw * 0.14, fw * 0.1, (0.86, 0.8, 1.06), 0.5, roll)

        # brighter, smoother under-eyes
        for lower in (L_LOWER, R_LOWER):
            low = pts[lower]
            ew = np.linalg.norm(low[0] - low[-1]) + 1e-6
            self._soft_tint(out, low.mean(axis=0) - up * ew * 0.28, ew * 0.42, ew * 0.16, (1.08, 1.07, 1.05), 0.6, roll)

        # brows: a bit darker and more even
        for brow in (L_BROW_FULL, R_BROW_FULL):
            poly = pts[brow].astype(np.int32)
            bx, by, bw, bh = cv2.boundingRect(poly)
            pad = 6
            x0, y0, x1, y1 = max(bx - pad, 0), max(by - pad, 0), min(bx + bw + pad, W), min(by + bh + pad, H)
            if x1 <= x0 or y1 <= y0:
                continue
            m = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillPoly(m, [poly - np.array([x0, y0])], 255, cv2.LINE_AA)
            a = cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), 2.0)[..., None] * 0.35
            roi = out[y0:y1, x0:x1].astype(np.float32)
            out[y0:y1, x0:x1] = np.clip(roi * (1 - a) + roi * np.float32([0.7, 0.72, 0.76]) * a, 0, 255).astype(np.uint8)

        # lash line: a soft, thin darkening along the upper lid (no wing)
        layer = out.copy()
        for upper in (L_UPPER, R_UPPER):
            lid = pts[upper].astype(np.int32)
            ew = np.linalg.norm(pts[upper][0] - pts[upper][-1]) + 1e-6
            cv2.polylines(layer, [lid], False, (35, 30, 35), max(1, int(ew * 0.045)), cv2.LINE_AA)
        x0, y0, x1, y1 = self._roi(out, pts, 0)
        soft = cv2.GaussianBlur(layer[y0:y1, x0:x1], (0, 0), 0.8)
        out[y0:y1, x0:x1] = cv2.addWeighted(soft, 0.45, out[y0:y1, x0:x1], 0.55, 0)

        # rosy lip tint, a little deeper toward the middle, soft sheen on the lower lip
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
            lips = cv2.GaussianBlur(cv2.subtract(mo, mi).astype(np.float32) / 255, (0, 0), 1.5)[..., None]
            depth = cv2.GaussianBlur(cv2.dilate(mi, np.ones((5, 5), np.uint8)).astype(np.float32) / 255,
                                     (0, 0), max(1, fw * 0.03))[..., None]
            roi = out[y0:y1, x0:x1].astype(np.float32)
            tint = roi * np.float32([0.86, 0.8, 1.08]) * (1 - depth) + roi * np.float32([0.74, 0.68, 1.04]) * depth
            out[y0:y1, x0:x1] = np.clip(roi * (1 - lips * 0.6) + tint * lips * 0.6, 0, 255).astype(np.uint8)
            self._glow_spot(out, pts[14] * 0.45 + pts[17] * 0.55, fw * 0.025, 0.18)

    # ---------------------------------------------------------- eyes: bigger irises, aegyo sal, lash line
    @staticmethod
    def _iris(out, pts, eye):
        """Iris center and radius: from the iris landmarks when the model gives them, else the darkest spot."""
        poly = pts[eye]
        ew = float(np.linalg.norm(poly[0] - poly[8])) + 1e-6
        iris_idx = 468 if eye is L_EYE else 473
        if len(pts) > 477 and np.any(pts[iris_idx] != 0):
            ring = pts[iris_idx + 1:iris_idx + 5]
            c = pts[iris_idx]
            r = float(np.linalg.norm(ring - c, axis=1).mean())
            return c, r if r > 1 else ew * 0.2
        x0, y0, bw, bh = cv2.boundingRect(poly.astype(np.int32))
        if bw < 4 or bh < 4:
            return poly.mean(0), ew * 0.2
        g = cv2.cvtColor(out[y0:y0 + bh, x0:x0 + bw], cv2.COLOR_BGR2GRAY).astype(np.float32)
        m = np.zeros((bh, bw), np.uint8)
        cv2.fillPoly(m, [(poly - [x0, y0]).astype(np.int32)], 1)
        g = cv2.GaussianBlur(g, (0, 0), max(1.0, ew * 0.08))
        g[m == 0] = 255
        y, x = np.unravel_index(np.argmin(g), g.shape)
        return np.array([x0 + x, y0 + y], np.float32), ew * 0.21

    def _eyes(self, out, pts):
        H, W = out.shape[:2]
        up = _unit(pts[10] - pts[152])
        for eye, lower, upper in ((L_EYE, L_LOWER, L_UPPER), (R_EYE, R_LOWER, R_UPPER)):
            poly = pts[eye]
            ew = float(np.linalg.norm(poly[0] - poly[8])) + 1e-6
            pad = int(ew * 0.9)
            c0 = poly.mean(0)
            x0, y0 = int(max(c0[0] - pad, 0)), int(max(c0[1] - pad, 0))
            x1, y1 = int(min(c0[0] + pad, W)), int(min(c0[1] + pad, H))
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            off = np.array([x0, y0], np.float32)
            roi = out[y0:y1, x0:x1].astype(np.float32)
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)

            # circle-lens irises: magnify the iris inside the eye opening, soften and deepen its rim a little
            if IRIS_ENLARGE:
                c, r = self._iris(out, pts, eye)
                R = r * 1.9
                d2 = (xx - c[0]) ** 2 + (yy - c[1]) ** 2
                w = _falloff(d2, R) * IRIS_ENLARGE
                mapx = (xx - (xx - c[0]) * w - x0).astype(np.float32)
                mapy = (yy - (yy - c[1]) * w - y0).astype(np.float32)
                big = cv2.remap(out[y0:y1, x0:x1], mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)
                em = np.zeros((y1 - y0, x1 - x0), np.uint8)
                cv2.fillPoly(em, [(poly - off).astype(np.int32)], 255, cv2.LINE_AA)
                em = cv2.GaussianBlur(em.astype(np.float32) / 255, (0, 0), max(1.0, ew * 0.02))[..., None]
                rr = np.sqrt(d2)[..., None]
                limbal = np.exp(-((rr - r * (1 + IRIS_ENLARGE)) / (r * 0.18)) ** 2) * 0.25
                big = big * (1 - limbal)
                roi = roi * (1 - em) + big * em

            # aegyo sal: a soft highlighted puff hugging the lower lashes, with a faint shadow underneath
            if AEGYO_SAL:
                low = pts[lower]
                band = np.zeros((y1 - y0, x1 - x0), np.float32)
                shade = np.zeros_like(band)
                n = len(low)
                for i in range(n - 1):
                    t = i / (n - 1)
                    k = math.sin(math.pi * min(1.0, t * 1.15)) ** 0.7        # strongest under the middle
                    p = low[i] - up * ew * 0.09 - off
                    q = low[i + 1] - up * ew * 0.09 - off
                    cv2.line(band, tuple(p.astype(int)), tuple(q.astype(int)), float(k), max(2, int(ew * 0.11)), cv2.LINE_AA)
                    p2 = low[i] - up * ew * 0.2 - off
                    q2 = low[i + 1] - up * ew * 0.2 - off
                    cv2.line(shade, tuple(p2.astype(int)), tuple(q2.astype(int)), float(k), max(1, int(ew * 0.05)), cv2.LINE_AA)
                band = cv2.GaussianBlur(band, (0, 0), ew * 0.035)[..., None] * AEGYO_SAL
                shade = cv2.GaussianBlur(shade, (0, 0), ew * 0.03)[..., None] * AEGYO_SAL
                roi = roi * (1 + 0.1 * band) + 10 * band * np.float32([1.0, 1.0, 1.05])
                roi = roi * (1 - 0.08 * shade * np.float32([1.0, 1.02, 0.98]))

            # thin, soft lash line along the upper lid
            if LASH_LINE:
                line = np.zeros((y1 - y0, x1 - x0), np.float32)
                lid = (pts[upper] - off).astype(np.int32)
                cv2.polylines(line, [lid], False, 1.0, max(1, int(ew * 0.022)), cv2.LINE_AA)
                line = cv2.GaussianBlur(line, (0, 0), max(0.6, ew * 0.008))[..., None] * 0.35 * LASH_LINE
                roi = roi * (1 - line) + np.float32([30, 28, 34]) * line
            out[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)

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
