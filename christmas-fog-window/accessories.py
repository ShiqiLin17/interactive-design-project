"""Watercolor-style Christmas accessories (in the style of Shiqi's sticker sheet), painted in code.

Watercolor look: soft layered washes, darker pigment pooling at the edges, paper granulation,
irregular soft edges, and fine brush strokes for fur, needles and details.
Each sprite is a BGRA image plus an anchor point; they're cached in .cache/ after the first run.
"""
import math
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".cache")
VERSION = 4


# ------------------------------------------------------------------ basic painting tools
def noise(h, w, cells, rng, octaves=3):
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        c = max(2, int(cells * 2 ** o))
        n = rng.random((c, max(2, int(c * w / h)))).astype(np.float32)
        out += cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC) * amp
        total += amp
        amp *= 0.5
    return out / total


def over(dst, src):
    """Alpha-composite src over dst (float32 BGRA, alpha 0..1)."""
    a = src[..., 3:]
    out_a = a + dst[..., 3:] * (1 - a)
    rgb = (src[..., :3] * a + dst[..., :3] * dst[..., 3:] * (1 - a)) / np.maximum(out_a, 1e-4)
    return np.concatenate([rgb, out_a], -1)


def canvas(h, w):
    return np.zeros((h, w, 4), np.float32)


def wash(mask, color, rng, var=0.14, pool=0.35, grain=0.1, soft=2.0, light=(0.0, 0.0), shade=0.0, opacity=1.0):
    """One watercolor wash over a 0..1 mask: uneven pigment, dark edges, paper grain, soft edge."""
    h, w = mask.shape
    m8 = (mask > 0.5).astype(np.uint8)
    # irregular edge: wobble the boundary with noise
    edge_n = noise(h, w, max(4, h // 40), rng)
    sd = cv2.distanceTransform(m8, cv2.DIST_L2, 5) - cv2.distanceTransform(1 - m8, cv2.DIST_L2, 5)
    a = np.clip((sd + (edge_n - 0.5) * 6 + soft) / (2 * soft), 0, 1)
    inside = np.clip(sd, 0, None)
    col = np.float32(color)[None, None, :] * np.ones((h, w, 1), np.float32)
    blot = noise(h, w, max(3, h // 60), rng)
    col *= (1 + var * (blot - 0.5) * 2)[..., None]
    # simple lighting gradient (light = direction the light comes from, in -1..1)
    if shade:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        g = (light[0] * (xx / w - 0.5) + light[1] * (yy / h - 0.5)) * 2
        col *= (1 - shade * np.clip(-g, -1, 1) * 0.5)[..., None]
    # pigment pools at the edge of a wash -> darker rim
    rim = np.exp(-inside / 3.5)
    col *= (1 - pool * rim)[..., None]
    gr = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 0.7)
    col *= (1 - grain * (gr - 0.5) * 2)[..., None]
    out = np.zeros((h, w, 4), np.float32)
    out[..., :3] = np.clip(col, 0, 255)
    out[..., 3] = a * opacity
    return out


def strokes(h, w, pts, dirs, color_fn, rng, length=10, width=1, alpha=0.6, curve=0.2, blur=0.5):
    """Many short brush strokes (fur, needles). pts: (N,2), dirs: (N,2) unit directions."""
    prem = np.zeros((h, w, 3), np.float32)     # color * alpha (premultiplied), so blurring keeps colors clean
    a_img = np.zeros((h, w), np.float32)
    for (x, y), (dx, dy) in zip(pts, dirs):
        L = length * rng.uniform(0.6, 1.2)
        bend = rng.normal(0, curve)
        c = np.float32(color_fn())
        k = float(rng.uniform(0.5, 1.0) * alpha)
        px, py = x, y
        for s in range(3):
            ang = math.atan2(dy, dx) + bend * s
            nx, ny = px + math.cos(ang) * L / 3, py + math.sin(ang) * L / 3
            kk = k * (1 - 0.25 * s / 3)
            p0, p1 = (int(px), int(py)), (int(nx), int(ny))
            cv2.line(prem, p0, p1, tuple(float(v) * kk for v in c), width, cv2.LINE_AA)
            cv2.line(a_img, p0, p1, kk, width, cv2.LINE_AA)
            px, py = nx, ny
    if blur:
        prem = cv2.GaussianBlur(prem, (0, 0), blur)
        a_img = cv2.GaussianBlur(a_img, (0, 0), blur)
    layer = np.zeros((h, w, 4), np.float32)
    layer[..., :3] = np.clip(prem / np.maximum(a_img[..., None], 1e-4), 0, 255)
    layer[..., 3] = np.clip(a_img, 0, 1)
    return layer


def sample_in(mask, n, rng):
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0:
        return np.zeros((0, 2), np.float32)
    idx = rng.integers(0, len(xs), n)
    return np.stack([xs[idx] + rng.random(n), ys[idx] + rng.random(n)], 1).astype(np.float32)


def ball(h, w, cx, cy, r, color, rng, highlight=True):
    """A watercolor ornament ball / berry: wash + darker lower side + white highlight."""
    m = np.zeros((h, w), np.float32)
    cv2.circle(m, (int(cx), int(cy)), int(r), 1.0, -1, cv2.LINE_AA)
    lay = wash(m, color, rng, var=0.1, pool=0.4, grain=0.08, soft=1.2, light=(-0.6, -0.7), shade=0.6)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((xx - cx + r * 0.1) ** 2 + (yy - cy + r * 0.1) ** 2) / max(r, 1)
    lay[..., :3] *= (1.05 - 0.35 * np.clip(d, 0, 1) ** 2)[..., None]
    if highlight:
        hl = np.zeros((h, w), np.float32)
        cv2.ellipse(hl, (int(cx - r * 0.38), int(cy - r * 0.4)), (max(1, int(r * 0.22)), max(1, int(r * 0.14))), -35, 0, 360, 1.0, -1, cv2.LINE_AA)
        hl = cv2.GaussianBlur(hl, (0, 0), max(0.6, r * 0.06)) * 0.85 * m
        lay[..., :3] = lay[..., :3] * (1 - hl[..., None]) + 250 * hl[..., None]
    return lay


def paper(h, w, rng, k=0.05):
    g = noise(h, w, h // 8, rng, octaves=2)
    return 1 - k * (g - 0.5) * 2


def outline(img, k=0.35, width=2.0):
    """Soft darker rim around the whole painting, like the ink/pigment edge on the sticker art."""
    a = img[..., 3]
    inner = cv2.erode((a > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=int(width))
    rim = cv2.GaussianBlur(((a > 0.5) & (inner == 0)).astype(np.float32), (0, 0), width * 0.6)
    img[..., :3] *= (1 - k * np.clip(rim, 0, 1))[..., None]
    return img


def snow_dabs(img, mask_edge_pts, rng, size=10):
    """White watercolor snow resting on top edges."""
    H, W = img.shape[:2]
    m = np.zeros((H, W), np.float32)
    for x, y in mask_edge_pts:
        cv2.ellipse(m, (int(x), int(y)), (int(size * rng.uniform(1.2, 2.4)), int(size * rng.uniform(0.6, 1.0))),
                    float(rng.uniform(-15, 15)), 180, 360, 1.0, -1, cv2.LINE_AA)
    m = cv2.GaussianBlur(m, (0, 0), 1.5)
    lay = wash(m, (246, 246, 250), rng, var=0.05, pool=0.12, grain=0.05, soft=1.5)
    sh = np.zeros((H, W), np.float32)
    for x, y in mask_edge_pts:
        cv2.ellipse(sh, (int(x), int(y + size * 0.5)), (int(size * 1.4), int(size * 0.35)), 0, 0, 180, 1.0, -1, cv2.LINE_AA)
    sh = cv2.GaussianBlur(sh, (0, 0), 2) * (m > 0.3)
    lay[..., :3] *= (1 - 0.12 * sh)[..., None]
    return over(img, lay)


# ------------------------------------------------------------------ the accessories
def santa_hat(rng):
    W, H = 900, 760
    img = canvas(H, W)
    # cone: from a wide base over the head, rising and flopping over to the right
    body = np.zeros((H, W), np.float32)
    pts = np.array([(110, 600), (170, 400), (300, 230), (470, 140), (620, 130), (740, 210), (790, 330),
                    (770, 430), (720, 360), (650, 300), (600, 320), (620, 450), (760, 600)], np.int32)
    cv2.fillPoly(body, [pts], 1.0, cv2.LINE_AA)
    body = cv2.GaussianBlur(body, (0, 0), 4)
    red = wash(body, (58, 60, 180), rng, var=0.18, pool=0.3, grain=0.12, soft=3, light=(-0.8, -0.6), shade=0.7)
    img = over(img, red)
    # second, darker glaze in the folds
    fold = np.zeros((H, W), np.float32)
    cv2.ellipse(fold, (640, 330), (60, 110), -30, 0, 360, 1.0, -1, cv2.LINE_AA)
    cv2.ellipse(fold, (560, 520), (80, 70), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    cv2.line(fold, (330, 560), (420, 260), 1.0, 26, cv2.LINE_AA)
    fold = cv2.GaussianBlur(fold, (0, 0), 18) * body
    img = over(img, wash(fold, (42, 38, 128), rng, var=0.2, pool=0.15, soft=6, opacity=0.55))
    # velvety fur strokes following the hat toward the tip
    tip = np.array([760, 400], np.float32)
    P = sample_in(body > 0.6, 4200, rng)
    D = tip - P
    D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-6
    D = D * 0.6 + rng.normal(0, 0.45, D.shape)
    D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-6
    shades = [(66, 68, 195), (45, 42, 150), (88, 92, 210), (40, 36, 125)]
    img = over(img, strokes(H, W, P, D, lambda: shades[int(rng.integers(0, 4))], rng, length=16, width=2, alpha=0.4, curve=0.15, blur=0.7))
    # fluffy white trim and pom-pom: many soft blobs, blue-grey shadows, fur at the edges
    def fluff(mask_shape_fn, n_blobs, n_fur):
        nonlocal img
        m = np.zeros((H, W), np.float32)
        mask_shape_fn(m)
        m = cv2.GaussianBlur(m, (0, 0), 3)
        base = wash(m, (232, 238, 244), rng, var=0.06, pool=0.18, grain=0.06, soft=4, light=(-0.5, -0.9), shade=0.5)
        img = over(img, base)
        sh = np.zeros((H, W), np.float32)
        P2 = sample_in(m > 0.5, n_blobs, rng)
        for x, y in P2:
            cv2.circle(sh, (int(x), int(y + 9)), int(rng.uniform(8, 18)), float(rng.uniform(0.2, 0.6)), -1, cv2.LINE_AA)
        sh = cv2.GaussianBlur(sh, (0, 0), 5) * m
        lay = canvas(H, W)
        lay[..., :3] = (205, 192, 182)
        lay[..., 3] = np.clip(sh, 0, 0.6)
        img = over(img, lay)
        P3 = sample_in(m > 0.3, n_fur, rng)
        D3 = rng.normal(0, 1, P3.shape)
        D3 /= np.linalg.norm(D3, axis=1, keepdims=True) + 1e-6
        fur_cols = [(250, 250, 252), (235, 232, 230), (215, 205, 198)]
        img = over(img, strokes(H, W, P3, D3, lambda: fur_cols[int(rng.integers(0, 3))], rng,
                                length=12, width=2, alpha=0.55, curve=0.4, blur=0.6))
    fluff(lambda m: cv2.ellipse(m, (440, 600), (355, 78), -3, 0, 360, 1.0, -1, cv2.LINE_AA), 260, 3200)
    fluff(lambda m: cv2.circle(m, (775, 470), 74, 1.0, -1, cv2.LINE_AA), 90, 1300)
    # a little holly sprig on the trim
    img = holly_on(img, 250, 560, 1.0, rng)
    img = outline(img, 0.3, 2)
    img[..., :3] *= paper(H, W, rng)[..., None]
    return img, (0.49, 0.8)


def holly_on(img, cx, cy, s, rng):
    H, W = img.shape[:2]
    for ang in (-35, 25, 160):
        m = np.zeros((H, W), np.float32)
        a = math.radians(ang)
        L, Wd = 70 * s, 22 * s
        pts = []
        for i in range(15):
            t = i / 14
            r = Wd * math.sin(math.pi * t) * (1.2 if i % 2 == 0 else 0.8)
            pts.append((t * L, -r))
        for i in range(14, -1, -1):
            t = i / 14
            r = Wd * math.sin(math.pi * t) * (1.2 if i % 2 == 0 else 0.8)
            pts.append((t * L, r))
        poly = np.array([(cx + px * math.cos(a) - py * math.sin(a), cy + px * math.sin(a) + py * math.cos(a)) for px, py in pts], np.int32)
        cv2.fillPoly(m, [poly], 1.0, cv2.LINE_AA)
        img = over(img, wash(m, (52, 96, 40), rng, var=0.2, pool=0.45, grain=0.1, soft=1.5))
        vein = np.zeros((H, W), np.float32)
        cv2.line(vein, (int(cx), int(cy)), (int(cx + L * 0.9 * math.cos(a)), int(cy + L * 0.9 * math.sin(a))), 0.6, 2, cv2.LINE_AA)
        lay = canvas(H, W)
        lay[..., :3] = (110, 160, 120)
        lay[..., 3] = cv2.GaussianBlur(vein, (0, 0), 0.7)
        img = over(img, lay)
    for dx, dy in ((-8, -6), (10, -4), (0, 10)):
        img = over(img, ball(H, W, cx + dx * s, cy + dy * s, 11 * s, (40, 35, 185), rng))
    return img


def antlers(rng):
    W, H = 1100, 640
    img = canvas(H, W)
    m = np.zeros((H, W), np.float32)
    branches = []
    for s in (-1, 1):
        cx = W / 2 + s * 230
        main = [(cx, 560), (cx + s * 20, 430), (cx + s * 45, 300), (cx + s * 60, 170), (cx + s * 50, 70)]
        branches.append((main, 60))
        branches.append(([(cx + s * 22, 420), (cx + s * 110, 360), (cx + s * 185, 300), (cx + s * 210, 230)], 44))
        branches.append(([(cx + s * 45, 300), (cx - s * 25, 220), (cx - s * 60, 140)], 40))
        branches.append(([(cx + s * 130, 345), (cx + s * 150, 260), (cx + s * 145, 190)], 34))
        branches.append(([(cx + s * 58, 190), (cx + s * 140, 120), (cx + s * 175, 60)], 34))
    for pts, t in branches:
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            cv2.line(m, (int(x0), int(y0)), (int(x1), int(y1)), 1.0, t, cv2.LINE_AA)
        cv2.circle(m, (int(pts[-1][0]), int(pts[-1][1])), t // 2, 1.0, -1, cv2.LINE_AA)
    m = cv2.GaussianBlur(m, (0, 0), 3)
    img = over(img, wash(m, (75, 120, 165), rng, var=0.2, pool=0.5, grain=0.12, soft=2.5, light=(-0.7, -0.5), shade=0.5))
    # plush texture strokes along each branch + lighter highlight down one side
    for pts, t in branches:
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            n = int(math.hypot(x1 - x0, y1 - y0) / 2)
            d = np.array([x1 - x0, y1 - y0], np.float32)
            d /= np.linalg.norm(d) + 1e-6
            nrm = np.array([-d[1], d[0]])
            P = np.array([(x0 + (x1 - x0) * u, y0 + (y1 - y0) * u) for u in rng.random(n)], np.float32)
            P += nrm[None] * rng.normal(0, t * 0.3, (n, 1))
            D = np.repeat(d[None], n, 0) + rng.normal(0, 0.3, (n, 2))
            D /= np.linalg.norm(D, axis=1, keepdims=True)
            cols = [(100, 148, 190), (55, 88, 125), (130, 175, 210), (80, 120, 160)]
            img = over(img, strokes(H, W, P, D, lambda: cols[int(rng.integers(0, 3))], rng, length=10, width=2, alpha=0.45, blur=0.6))
    # red velvet headband with a holly sprig and two bells
    band = np.zeros((H, W), np.float32)
    cv2.ellipse(band, (W // 2, 820), (420, 290), 0, 198, 342, 1.0, 40, cv2.LINE_AA)
    band = cv2.GaussianBlur(band, (0, 0), 2)
    img = over(img, wash(band, (50, 45, 190), rng, var=0.15, pool=0.4, grain=0.1, soft=2, light=(0, -1), shade=0.5))
    img = holly_on(img, W / 2 - 250, 590, 0.9, rng)
    img = holly_on(img, W / 2 + 250, 590, 0.9, rng)
    tips = [pts[-1] for pts, t in branches]
    img = snow_dabs(img, [(x, y - 8) for x, y in tips], rng, size=12)
    img = outline(img, 0.35, 2)
    img[..., :3] *= paper(H, W, rng)[..., None]
    return img, (0.5, 0.88)


def red_nose(rng):
    S = 220
    img = over(canvas(S, S), ball(S, S, S / 2, S / 2, 92, (45, 40, 205), rng))
    return img, (0.5, 0.5)


def tree_hat(rng):
    W, H = 820, 980
    img = canvas(H, W)
    tiers = [(130, 300, 150), (260, 470, 230), (410, 650, 300), (570, 830, 360)]  # (top y, bottom y, half width)
    greens = [(80, 125, 75), (70, 115, 66), (62, 106, 58), (55, 98, 52)]
    for k, (y0, y1, hw) in enumerate(tiers):
        m = np.zeros((H, W), np.float32)
        poly = [(W / 2, y0 - 30)]
        n = 9
        for i in range(n + 1):           # scalloped bottom edge
            x = W / 2 + hw - 2 * hw * i / n
            poly.append((x, y1 + (14 if i % 2 else -6)))
        cv2.fillPoly(m, [np.array(poly, np.int32)], 1.0, cv2.LINE_AA)
        m = cv2.GaussianBlur(m, (0, 0), 3)
        img = over(img, wash(m, greens[k], rng, var=0.22, pool=0.35, grain=0.12, soft=3, light=(-0.6, -0.8), shade=0.6))
        # needles: radiating down/outward from the center line
        P = sample_in(m > 0.5, 1500, rng)
        D = np.stack([(P[:, 0] - W / 2) / (hw + 1e-6), np.ones(len(P)) * 1.2], 1) + rng.normal(0, 0.35, P.shape)
        D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-6
        cols = [(62, 108, 55), (45, 82, 40), (92, 142, 82), (110, 158, 100)]
        img = over(img, strokes(H, W, P, D, lambda: cols[int(rng.integers(0, 4))], rng, length=18, width=2, alpha=0.55, curve=0.1, blur=0.5))
    # snow resting along the bottom edge of each tier
    for (y0, y1, hw) in tiers:
        pts = [(W / 2 + hw * (1 - 2 * t) * 0.95, y1 - 4 + rng.normal(0, 3)) for t in np.linspace(0.05, 0.95, 9)]
        img = snow_dabs(img, pts, rng, size=13)
    # pearl garland swags
    for k, (y0, y1, hw) in enumerate(tiers[1:]):
        yb = (y0 + y1) / 2
        for i in range(26):
            t = i / 25
            x = W / 2 - hw * 0.8 + t * hw * 1.6
            y = yb - 40 + 70 * t + 25 * math.sin(t * math.pi)
            img = over(img, ball(H, W, x, y, 7, (225, 232, 238), rng))
    # ornaments and little bows
    spots = [(330, 260), (470, 330), (300, 420), (520, 470), (400, 560), (250, 590), (560, 640), (330, 720),
             (480, 760), (620, 790), (220, 780)]
    cols = [(55, 55, 178), (80, 175, 215), (55, 55, 178), (215, 205, 195)]
    for i, (x, y) in enumerate(spots):
        img = over(img, ball(H, W, x, y, 17 if i % 3 else 21, cols[i % 4], rng))
    for (x, y) in ((420, 400), (300, 640), (560, 560)):
        bow = np.zeros((H, W), np.float32)
        cv2.ellipse(bow, (x - 14, y), (16, 10), -20, 0, 360, 1.0, -1, cv2.LINE_AA)
        cv2.ellipse(bow, (x + 14, y), (16, 10), 20, 0, 360, 1.0, -1, cv2.LINE_AA)
        cv2.line(bow, (x, y), (x - 10, y + 26), 1.0, 6, cv2.LINE_AA)
        cv2.line(bow, (x, y), (x + 10, y + 26), 1.0, 6, cv2.LINE_AA)
        img = over(img, wash(bow, (50, 45, 190), rng, var=0.15, pool=0.45, soft=1.2))
    # gold star on top
    star = np.zeros((H, W), np.float32)
    c = (W / 2, 70)
    pts = [(c[0] + (62 if i % 2 == 0 else 26) * math.cos(-math.pi / 2 + i * math.pi / 5),
            c[1] + (62 if i % 2 == 0 else 26) * math.sin(-math.pi / 2 + i * math.pi / 5)) for i in range(10)]
    cv2.fillPoly(star, [np.array(pts, np.int32)], 1.0, cv2.LINE_AA)
    img = over(img, wash(star, (90, 190, 228), rng, var=0.15, pool=0.4, soft=1.5, light=(-0.7, -0.7), shade=0.6))
    img = outline(img, 0.3, 2)
    img[..., :3] *= paper(H, W, rng)[..., None]
    return img, (0.5, 0.9)


def garland_crown(rng):
    W, H = 1100, 460
    img = canvas(H, W)
    path = [(W / 2 + math.cos(a) * 470, 410 - math.sin(a) * 175) for a in np.linspace(math.radians(8), math.radians(172), 90)]
    base = np.zeros((H, W), np.float32)
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        cv2.line(base, (int(x0), int(y0)), (int(x1), int(y1)), 1.0, 105, cv2.LINE_AA)
    base = cv2.GaussianBlur(base, (0, 0), 5)
    img = over(img, wash(base, (55, 100, 45), rng, var=0.25, pool=0.3, grain=0.12, soft=6, opacity=0.9))
    P = sample_in(base > 0.4, 8000, rng)
    D = rng.normal(0, 1, P.shape)
    D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-6
    cols = [(62, 108, 55), (40, 78, 36), (92, 142, 82), (112, 160, 102)]
    img = over(img, strokes(H, W, P, D, lambda: cols[int(rng.integers(0, 4))], rng, length=20, width=2, alpha=0.6, curve=0.1, blur=0.5))
    # pinecones
    for k in (12, 45, 78):
        x, y = path[k]
        m = np.zeros((H, W), np.float32)
        cv2.ellipse(m, (int(x), int(y)), (22, 32), 15, 0, 360, 1.0, -1, cv2.LINE_AA)
        img = over(img, wash(m, (60, 95, 140), rng, var=0.15, pool=0.45, soft=1.5))
        sc = canvas(H, W)
        for j in range(6):
            cv2.ellipse(sc, (int(x), int(y - 24 + j * 9)), (18 - abs(j - 2) * 2, 5), 15, 0, 180, (40, 60, 95, 0.8), 2, cv2.LINE_AA)
        img = over(img, sc)
    # red berries and a bow in the middle
    for _ in range(26):
        x, y = path[int(rng.integers(3, len(path) - 3))]
        img = over(img, ball(H, W, x + rng.normal(0, 12), y + rng.normal(0, 10), 8, (40, 35, 185), rng))
    bx, by = path[45]
    bow = np.zeros((H, W), np.float32)
    cv2.ellipse(bow, (int(bx - 30), int(by)), (34, 20), -18, 0, 360, 1.0, -1, cv2.LINE_AA)
    cv2.ellipse(bow, (int(bx + 30), int(by)), (34, 20), 18, 0, 360, 1.0, -1, cv2.LINE_AA)
    cv2.line(bow, (int(bx), int(by)), (int(bx - 22), int(by + 55)), 1.0, 12, cv2.LINE_AA)
    cv2.line(bow, (int(bx), int(by)), (int(bx + 22), int(by + 55)), 1.0, 12, cv2.LINE_AA)
    img = over(img, wash(bow, (50, 45, 195), rng, var=0.18, pool=0.45, soft=1.5, light=(-0.6, -0.7), shade=0.5))
    img = over(img, ball(H, W, bx, by, 12, (45, 40, 170), rng))
    for k in (25, 65):        # gold bells
        x, y = path[k]
        m = np.zeros((H, W), np.float32)
        poly = np.array([(x - 10, y - 26), (x + 10, y - 26), (x + 26, y + 14), (x - 26, y + 14)], np.int32)
        cv2.fillPoly(m, [poly], 1.0, cv2.LINE_AA)
        cv2.ellipse(m, (int(x), int(y - 26)), (12, 10), 0, 180, 360, 1.0, -1, cv2.LINE_AA)
        img = over(img, wash(m, (70, 170, 215), rng, var=0.15, pool=0.5, soft=1.2, light=(-0.8, -0.4), shade=0.7))
        img = over(img, ball(H, W, x, y + 16, 6, (60, 130, 175), rng))
    img = snow_dabs(img, [path[k] for k in range(4, len(path) - 4, 7)], rng, size=12)
    img = outline(img, 0.3, 2)
    img[..., :3] *= paper(H, W, rng)[..., None]
    return img, (0.5, 0.86)


# ------------------------------------------------------------------ build + cache
def _to_u8(img):
    out = np.zeros(img.shape, np.uint8)
    out[..., :3] = np.clip(img[..., :3], 0, 255)
    out[..., 3] = np.clip(img[..., 3] * 255, 0, 255)
    return out


def build_all(seed=7):
    makers = {"santa hat": santa_hat, "reindeer": antlers, "red nose": red_nose,
              "tree hat": tree_hat, "garland crown": garland_crown}
    os.makedirs(CACHE, exist_ok=True)
    out = {}
    for i, (name, fn) in enumerate(makers.items()):
        path = os.path.join(CACHE, f"acc_v{VERSION}_{name.replace(' ', '_')}.png")
        anchors = {"santa hat": (0.49, 0.8), "reindeer": (0.5, 0.88), "red nose": (0.5, 0.5),
                   "tree hat": (0.5, 0.9), "garland crown": (0.5, 0.86)}
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED) if os.path.exists(path) else None
        if img is None:
            img, _ = fn(np.random.default_rng(seed + i))
            img = _to_u8(img)
            cv2.imwrite(path, img)
        out[name] = (img, anchors[name])
    return out
