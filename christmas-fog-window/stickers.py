"""Elegant, realistic Christmas ornaments, rendered with light and shading in code (no image files).

Each one is drawn at 2x size with lighting, then scaled down for smooth edges.
Returns BGRA images with soft transparent edges (no outlines).
"""
import cv2
import numpy as np

S = 2                 # supersampling
SIZE = 220 * S        # canvas
L = np.array([-0.45, -0.6, 0.66], np.float32)
L /= np.linalg.norm(L)
HV = L + np.array([0, 0, 1], np.float32)
HV /= np.linalg.norm(HV)


def _grid():
    y, x = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    return x, y


def _finish(rgb, alpha):
    """Combine float BGR (0..255) + alpha (0..1) at 2x, downsample to final BGRA."""
    out = np.zeros((SIZE, SIZE, 4), np.float32)
    out[..., :3] = np.clip(rgb, 0, 255) * alpha[..., None]   # premultiplied for clean downsampling
    out[..., 3] = np.clip(alpha, 0, 1)
    out = cv2.resize(out, (SIZE // S, SIZE // S), interpolation=cv2.INTER_AREA)
    a = out[..., 3:]
    out[..., :3] = out[..., :3] / np.maximum(a, 1e-4)
    res = np.zeros(out.shape, np.uint8)
    res[..., :3] = np.clip(out[..., :3], 0, 255)
    res[..., 3] = np.clip(out[..., 3] * 255, 0, 255)
    return res


def _sphere(x, y, cx, cy, R, base, gloss=1.0):
    """Shaded glossy sphere. Returns (rgb, alpha)."""
    nx, ny = (x - cx) / R, (y - cy) / R
    r2 = nx * nx + ny * ny
    inside = r2 < 1
    nz = np.sqrt(np.clip(1 - r2, 0, 1))
    diff = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, 1)
    spec = np.clip(nx * HV[0] + ny * HV[1] + nz * HV[2], 0, 1)
    fres = (1 - nz) ** 2.5
    base = np.array(base, np.float32)
    rgb = base * (0.18 + 0.95 * diff[..., None])
    rgb += fres[..., None] * np.float32([235, 205, 180]) * 0.35            # cool sky reflection on the rim
    warm = np.clip((nx * 0.6 + ny * 0.8), 0, 1) ** 3 * (1 - nz) * 1.4        # warm firelight from bottom right
    rgb += warm[..., None] * np.float32([60, 150, 255]) * 0.55
    rgb += (spec ** 90)[..., None] * 255 * gloss
    # soft window-shaped reflection near the top left
    wx, wy = (nx + 0.38) / 0.2, (ny + 0.42) / 0.27
    win = np.clip(1 - (np.abs(wx) ** 4 + np.abs(wy) ** 4), 0, 1) ** 0.6
    rgb += win[..., None] * 255 * 0.55 * gloss
    edge = np.clip((1 - np.sqrt(r2)) * R / 1.5, 0, 1)
    return rgb, edge * inside


def bauble():
    x, y = _grid()
    cx, cy, R = SIZE / 2, SIZE * 0.56, SIZE * 0.36
    rgb, a = _sphere(x, y, cx, cy, R, (38, 22, 150))   # deep crimson glass
    # brushed-gold cap
    cw, ch = R * 0.36, R * 0.24
    x1, x2, y1, y2 = cx - cw / 2, cx + cw / 2, cy - R - ch * 0.55, cy - R + ch * 0.55
    capm = ((x >= x1) & (x <= x2) & (y >= y1) & (y <= y2)).astype(np.float32)
    u = np.clip((x - x1) / max(cw, 1), 0, 1)
    gold = np.float32([50, 160, 215])
    shade = 0.55 + 0.6 * np.clip(np.sin(u * np.pi), 0, 1) ** 0.7 + 0.12 * np.sin(u * 60)
    caprgb = gold[None, None, :] * shade[..., None] + (np.abs(u - 0.32) < 0.06)[..., None] * 80
    for k in range(4):  # little ridges
        yy = y1 + (y2 - y1) * (k + 0.5) / 4
        capm_r = (np.abs(y - yy) < 1.2 * S) & (capm > 0)
        caprgb[capm_r] *= 0.75
    rgb = np.where(capm[..., None] > 0, caprgb, rgb)
    a = np.maximum(a, capm)
    # wire loop
    loop = np.zeros((SIZE, SIZE), np.float32)
    cv2.ellipse(loop, (int(cx), int(y1 - 9 * S)), (int(7 * S), int(9 * S)), 0, 0, 360, 1.0, int(2.2 * S), cv2.LINE_AA)
    rgb = rgb * (1 - loop[..., None]) + np.float32([90, 190, 235]) * loop[..., None]
    a = np.maximum(a, loop)
    return _finish(rgb, a)


def snowflake():
    canvas = np.zeros((SIZE, SIZE), np.float32)
    c = np.array([SIZE / 2, SIZE / 2])
    Rm = SIZE * 0.44

    def seg(p, q, th):
        cv2.line(canvas, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), 1.0, max(1, int(th)), cv2.LINE_AA)

    for i in range(6):
        a = i * np.pi / 3 - np.pi / 2
        d = np.array([np.cos(a), np.sin(a)])
        seg(c, c + d * Rm, 5 * S)
        for t, ln, th in ((0.3, 0.22, 3.6), (0.52, 0.27, 3.2), (0.72, 0.2, 2.8), (0.88, 0.1, 2.2)):
            b = c + d * Rm * t
            for s in (-1, 1):
                a2 = a + s * np.pi / 3
                seg(b, b + np.array([np.cos(a2), np.sin(a2)]) * Rm * ln, th * S)
        tip = c + d * Rm
        cv2.circle(canvas, (int(tip[0]), int(tip[1])), int(3.5 * S), 1.0, -1, cv2.LINE_AA)
    hexa = [c + np.array([np.cos(i * np.pi / 3), np.sin(i * np.pi / 3)]) * Rm * 0.14 for i in range(6)]
    cv2.polylines(canvas, [np.array(hexa, np.int32)], True, 1.0, int(2.5 * S), cv2.LINE_AA)
    cv2.circle(canvas, (int(c[0]), int(c[1])), int(5 * S), 1.0, -1, cv2.LINE_AA)
    # bevelled ice: height from blurred shape -> lighting
    hgt = cv2.GaussianBlur(canvas, (0, 0), 2.2 * S)
    gx = cv2.Sobel(hgt, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(hgt, cv2.CV_32F, 0, 1, ksize=3)
    n = np.dstack([-gx, -gy, np.full_like(gx, 0.35)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    diff = np.clip(n @ L, 0, 1)
    spec = np.clip(n @ HV, 0, 1) ** 40
    base = np.float32([250, 238, 222])                   # frosty silver-white with a hint of blue
    rgb = base * (0.62 + 0.42 * diff[..., None]) + spec[..., None] * 140
    rng = np.random.default_rng(3)
    glint = (rng.random((SIZE, SIZE)) > 0.996) * canvas
    rgb += cv2.GaussianBlur(glint.astype(np.float32), (0, 0), 0.8)[..., None] * 600
    halo = cv2.GaussianBlur(canvas, (0, 0), 6 * S) * 0.25
    alpha = np.maximum(canvas, halo)
    return _finish(rgb, alpha)


def star():
    rgb = np.zeros((SIZE, SIZE, 3), np.float32)
    alpha = np.zeros((SIZE, SIZE), np.float32)
    c = np.array([SIZE / 2, SIZE / 2 + 6 * S])
    Ro, Ri = SIZE * 0.44, SIZE * 0.19
    pts = []
    for i in range(10):
        r = Ro if i % 2 == 0 else Ri
        a = -np.pi / 2 + i * np.pi / 5
        pts.append(c + np.array([np.cos(a), np.sin(a)]) * r)
    light = np.array([-0.5, -0.8])
    for i in range(10):  # each arm = two facets, lit by how they face the light
        p, q = pts[i], pts[(i + 1) % 10]
        mid = (p + q) / 2 - c
        k = 0.5 + 0.5 * np.dot(mid / (np.linalg.norm(mid) + 1e-6), light)
        col = np.float32([30, 120, 190]) + np.float32([110, 120, 65]) * k
        m = np.zeros((SIZE, SIZE), np.float32)
        cv2.fillPoly(m, [np.array([c, p, q], np.int32)], 1.0, cv2.LINE_AA)
        # gentle gradient toward the tip
        rgb = rgb * (1 - m[..., None]) + col * m[..., None]
        alpha = np.maximum(alpha, m)
    x, y = _grid()
    d = np.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2) / Ro
    rgb *= (1.08 - 0.25 * d)[..., None]
    for p in pts[::2]:  # bright ridge lines from center to tips
        line = np.zeros((SIZE, SIZE), np.float32)
        cv2.line(line, (int(c[0]), int(c[1])), (int(p[0]), int(p[1])), 1.0, int(1.4 * S), cv2.LINE_AA)
        rgb += line[..., None] * np.float32([120, 200, 235]) * 0.45
    glint = np.zeros((SIZE, SIZE), np.float32)
    g = pts[8] * 0.55 + c * 0.45
    cv2.circle(glint, (int(g[0]), int(g[1])), int(3 * S), 1.0, -1, cv2.LINE_AA)
    glint = cv2.GaussianBlur(glint, (0, 0), 3 * S)
    rgb += glint[..., None] * 400
    return _finish(rgb, alpha)


def holly():
    rgb = np.zeros((SIZE, SIZE, 3), np.float32)
    alpha = np.zeros((SIZE, SIZE), np.float32)
    x, y = _grid()

    def leaf(cx, cy, length, width, angle):
        nonlocal rgb, alpha
        pts = []
        n = 7
        for i in range(n * 2 + 1):  # spiky holly outline
            t = i / (n * 2)
            along = (t - 0.5) * length
            w = width * np.sin(np.pi * t) ** 0.8
            w *= 1.18 if i % 2 == 0 else 0.82
            pts.append((along, -w))
        for i in range(n * 2, -1, -1):
            t = i / (n * 2)
            along = (t - 0.5) * length
            w = width * np.sin(np.pi * t) ** 0.8
            w *= 1.18 if i % 2 == 0 else 0.82
            pts.append((along, w))
        ca, sa = np.cos(angle), np.sin(angle)
        poly = np.array([(cx + px * ca - py * sa, cy + px * sa + py * ca) for px, py in pts], np.int32)
        m = np.zeros((SIZE, SIZE), np.float32)
        cv2.fillPoly(m, [poly], 1.0, cv2.LINE_AA)
        # leaf lighting: each half tilts a different way, glossy sheen along the middle
        u = (x - cx) * ca + (y - cy) * sa
        v = -(x - cx) * sa + (y - cy) * ca
        side = np.tanh(v / (width * 0.5))
        k = 0.55 + 0.35 * side * np.sign(-sa + 0.2) + 0.15 * (1 - np.abs(u) / (length / 2))
        col = np.float32([40, 88, 22]) * np.clip(k, 0.3, 1.3)[..., None]
        sheen = np.exp(-((v + width * 0.25) / (width * 0.25)) ** 2) * (np.abs(u) < length * 0.45)
        col += sheen[..., None] * np.float32([150, 190, 150]) * 0.35
        vein = (np.abs(v) < 1.2 * S) & (np.abs(u) < length * 0.48)
        col[vein] = col[vein] * 0.6 + np.float32([90, 150, 110]) * 0.4
        rgb = rgb * (1 - m[..., None]) + col * m[..., None]
        alpha = np.maximum(alpha, m)

    leaf(SIZE * 0.36, SIZE * 0.42, SIZE * 0.56, SIZE * 0.13, np.radians(-35))
    leaf(SIZE * 0.64, SIZE * 0.43, SIZE * 0.56, SIZE * 0.13, np.radians(35))
    leaf(SIZE * 0.5, SIZE * 0.66, SIZE * 0.46, SIZE * 0.11, np.radians(92))
    for (bx, by) in ((0.44, 0.52), (0.56, 0.53), (0.5, 0.42)):
        b, a = _sphere(x, y, SIZE * bx, SIZE * by, SIZE * 0.075, (30, 25, 175), gloss=0.9)
        rgb = rgb * (1 - a[..., None]) + b * a[..., None]
        alpha = np.maximum(alpha, a)
    return _finish(rgb, alpha)


# which finger touches your thumb picks the ornament
STICKERS = {
    "bauble": bauble,        # thumb + index
    "snowflake": snowflake,  # thumb + middle
    "star": star,            # thumb + ring
    "holly": holly,          # thumb + pinky
}


def build_all():
    return {name: fn() for name, fn in STICKERS.items()}
