"""The Christmas window: night outside, your reflection in the glass, and a cozy decorated frame."""
import math
import random

import cv2
import numpy as np

AA = cv2.LINE_AA
# warm bulb colors (BGR)
LIGHT_COLORS = [(70, 80, 255), (90, 230, 120), (60, 200, 255), (255, 170, 90), (200, 230, 255)]


def _noise(h, w, cells_y, cells_x, rng, sigma=0):
    n = rng.random((max(2, cells_y), max(2, cells_x))).astype(np.float32)
    n = cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
    if sigma:
        n = cv2.GaussianBlur(n, (0, 0), sigma)
    return n


def wood_texture(h, w, rng, vertical=False):
    """Procedural wood grain (BGR uint8)."""
    if vertical:
        return np.ascontiguousarray(wood_texture(w, h, rng).transpose(1, 0, 2))
    warp = _noise(h, w, h // 6, w // 120, rng) * 30 + _noise(h, w, h // 2, w // 30, rng) * 6
    y = np.arange(h, dtype=np.float32)[:, None]
    rings = np.sin((y * 0.55 + warp) * 0.9)
    rings = np.sign(rings) * np.abs(rings) ** 0.6
    fine = _noise(h, w, h, w // 6, rng) - 0.5
    pores = (rng.random((h, w)) > 0.985).astype(np.float32)
    pores = cv2.GaussianBlur(cv2.resize(pores, (w, h)), (0, 0), 0.7)
    base = np.array([38, 74, 124], np.float32)       # warm honey-brown
    dark = np.array([22, 44, 78], np.float32)
    t = (rings[..., None] * 0.5 + 0.5) * 0.55 + fine[..., None] * 0.35
    tex = dark + (base - dark) * np.clip(t + 0.35, 0, 1.2) - pores[..., None] * 25
    return np.clip(tex, 0, 255).astype(np.uint8)


class ChristmasWindow:
    def __init__(self, w, h, seed=3):
        self.w, self.h = w, h
        rnd = random.Random(seed)
        rng = np.random.default_rng(seed)
        self.side = int(w * 0.05)
        self.top = int(h * 0.08)
        self.sill = int(h * 0.12)
        self.bar = max(12, int(w * 0.013))
        self.glass = (self.side, self.top, w - self.side, h - self.sill)  # x1, y1, x2, y2
        gx1, gy1, gx2, gy2 = self.glass
        self.transom_y = gy1 + int((gy2 - gy1) * 0.24)
        self.night = self._night_scene(rnd)
        self.streaks = self._glass_streaks()
        self.overlay, self.lights, self.candles = self._frame(rnd, rng)
        self.ov_rgb = np.ascontiguousarray(self.overlay[..., :3])
        self.ov_a = self.overlay[..., 3].astype(np.float32) / 255
        self.ov_inv = 1 - self.ov_a
        self.night_part = (self.night * 0.40 + self.streaks).astype(np.float32)
        # vignette + warm light from the candles at the bottom
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        d = np.sqrt(((xx - w / 2) / (w * 0.62)) ** 2 + ((yy - h * 0.45) / (h * 0.7)) ** 2)
        vig = np.clip(1.08 - d ** 2.2 * 0.55, 0.45, 1.0)
        warm = np.stack([vig * 0.96, vig * 0.99, vig * 1.04], -1)
        self.grade = np.clip(warm * 255, 0, 255).astype(np.uint8)

    # ------------------------------------------------ outside + glass
    def _night_scene(self, rnd):
        w, h = self.w, self.h
        y = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
        top = np.array([48, 22, 10], np.float32)
        bottom = np.array([95, 55, 32], np.float32)
        img = np.repeat(top + (bottom - top) * y, w, axis=1)
        bokeh = np.zeros((h, w, 3), np.float32)
        for _ in range(38):  # out-of-focus lights from houses across the street
            c = rnd.choice(LIGHT_COLORS + [(110, 190, 255)] * 4)
            x, yy = rnd.randint(0, w), rnd.randint(int(h * 0.4), h)
            r = rnd.randint(8, 24)
            k = rnd.uniform(0.15, 0.5)
            cv2.circle(bokeh, (x, yy), r, tuple(v * k for v in c), -1, AA)
            cv2.circle(bokeh, (x, yy), r, tuple(v * k * 1.3 for v in c), 2, AA)  # bokeh rim
        bokeh = cv2.GaussianBlur(bokeh, (0, 0), 3)
        return np.clip(img + bokeh, 0, 255).astype(np.float32)

    def _glass_streaks(self):
        w, h = self.w, self.h
        s = np.zeros((h, w), np.float32)
        for x0, width, k in [(0.16, 0.07, 1.0), (0.25, 0.018, 0.8), (0.66, 0.09, 0.7), (0.78, 0.022, 0.9)]:
            pts = np.array([[x0 * w, 0], [(x0 + width) * w, 0],
                            [(x0 + width - 0.3) * w, h], [(x0 - 0.3) * w, h]], np.int32)
            cv2.fillPoly(s, [pts], k, AA)
        return cv2.GaussianBlur(s, (0, 0), 30)[..., None] * np.array([16, 14, 12], np.float32)

    def reflection(self, cam):
        """Make the camera image look like your reflection in a window at night (slightly cool)."""
        out = cv2.scaleAdd(cam.astype(np.float32), 0.74, self.night_part)
        return cv2.convertScaleAbs(out)

    # ------------------------------------------------ frame + decorations
    def _frame(self, rnd, rng):
        w, h = self.w, self.h
        gx1, gy1, gx2, gy2 = self.glass
        b = self.bar
        ov = np.zeros((h, w, 4), np.float32)

        frame_mask = np.full((h, w), 255, np.uint8)
        cv2.rectangle(frame_mask, (gx1, gy1), (gx2, gy2), 0, -1)
        cv2.rectangle(frame_mask, (gx1, self.transom_y - b // 2), (gx2, self.transom_y + b // 2), 255, -1)
        mull_x = [gx1 + (gx2 - gx1) * k // 3 for k in (1, 2)]
        for x in mull_x:
            cv2.rectangle(frame_mask, (x - b // 2, gy1), (x + b // 2, self.transom_y), 255, -1)

        # frost creeping in from the corners of each pane
        frost = np.zeros((h, w), np.float32)
        xs = [gx1] + mull_x + [gx2]
        panes = [(xs[i], gy1, xs[i + 1], self.transom_y) for i in range(3)] + [(gx1, self.transom_y, gx2, gy2)]
        for (x1, y1, x2, y2) in panes:
            r = int(min(x2 - x1, y2 - y1) * 0.32)
            for cx, cy in ((x1, y1), (x2, y1), (x1, y2), (x2, y2)):
                cv2.circle(frost, (cx, cy), r, 1.0, -1, AA)
            cv2.rectangle(frost, (x1, y2 - 6), (x2, y2), 0.6, -1)
        crystals = _noise(h, w, h // 3, w // 3, rng) * 0.6 + _noise(h, w, h // 12, w // 12, rng) * 0.4
        frost = cv2.GaussianBlur(frost, (0, 0), 28) * (0.55 + crystals * 0.9)
        frost = np.clip(frost * 1.1 - 0.12, 0, 0.8)
        ov[..., :3] = (252, 247, 240)
        ov[..., 3] = frost

        # wood
        horiz = wood_texture(h, w, rng).astype(np.float32)
        vert = wood_texture(h, w, rng, vertical=True).astype(np.float32)
        vmask = np.zeros((h, w), np.float32)
        cv2.rectangle(vmask, (0, gy1), (gx1, gy2), 1, -1)
        cv2.rectangle(vmask, (gx2, gy1), (w, gy2), 1, -1)
        for x in mull_x:
            cv2.rectangle(vmask, (x - b // 2, gy1), (x + b // 2, self.transom_y - b // 2), 1, -1)
        wood = horiz * (1 - vmask[..., None]) + vert * vmask[..., None]
        # lighting: soft highlight along the top/left of every piece, shadow along the inner glass edge
        inner = (255 - frame_mask)
        dist = cv2.distanceTransform(frame_mask, cv2.DIST_L2, 5)
        ao = np.clip(dist / 14, 0, 1) ** 0.7                      # darker right next to the glass
        light = cv2.GaussianBlur(frame_mask.astype(np.float32) / 255, (0, 0), 3)
        dx = cv2.Sobel(light, cv2.CV_32F, 1, 0, ksize=5)
        dy = cv2.Sobel(light, cv2.CV_32F, 0, 1, ksize=5)
        bevel = np.clip(-(dx + dy) * 0.6, -1, 1)
        wood = wood * (0.55 + 0.45 * ao)[..., None] + bevel[..., None] * 40
        wood[gy2:] *= 1.12  # the sill catches the candle light
        wood[:gy1] *= 0.92
        sel = frame_mask > 0
        ov[sel, :3] = np.clip(wood[sel], 0, 255)
        ov[sel, 3] = 1.0
        # thin dark gap where the glass meets the wood
        gap = cv2.dilate(inner, np.ones((5, 5), np.uint8)) & frame_mask
        ov[gap > 0, :3] *= 0.45

        # soft snow drifts
        def drift(y, x1, x2, height, seed_off):
            m = np.zeros((h, w), np.float32)
            x = x1
            while x < x2:
                rx = rnd.randint(26, 60)
                ry = rnd.randint(int(height * 0.5), height)
                cv2.ellipse(m, (x, y), (rx, ry), 0, 180, 360, 1.0, -1, AA)
                x += rnd.randint(18, 40)
            cv2.rectangle(m, (x1, y), (x2, y + int(height * 0.6)), 1.0, -1)
            m = cv2.GaussianBlur(m, (0, 0), 3)
            m = np.clip((m - 0.35) * 3, 0, 1)
            d = cv2.distanceTransform((m > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
            top_lit = np.clip(d / 10, 0, 1)
            yy = np.arange(h, dtype=np.float32)[:, None]
            front = np.clip((yy - y) / (height * 0.6), 0, 1)       # front face is in shade
            shadow = np.array([222, 205, 196], np.float32)
            bright = np.array([253, 251, 249], np.float32)
            k = np.clip(top_lit * (1 - 0.65 * front), 0, 1)[..., None]
            col = shadow + (bright - shadow) * k
            sparkle = (rng.random((h, w)) > 0.993).astype(np.float32)
            col += cv2.GaussianBlur(sparkle, (0, 0), 0.6)[..., None] * 160
            sel = m > 0
            a = m[sel][..., None]
            ov[sel, :3] = ov[sel, :3] * (1 - a) + np.clip(col[sel], 0, 255) * a
            ov[sel, 3] = np.maximum(ov[sel, 3], m[sel])

        drift(gy2 + 6, gx1 - 14, gx2 + 14, 26, 0)
        drift(self.transom_y - b // 2 + 3, gx1, gx2, 13, 1)

        # pine garland: swags across the top and hanging down both sides
        path = []
        n_swags = 4
        sag = h * 0.065
        for i in range(n_swags):
            xa = gx1 + (gx2 - gx1) * i / n_swags
            xb = gx1 + (gx2 - gx1) * (i + 1) / n_swags
            for t in np.linspace(0, 1, 70, endpoint=False):
                path.append((xa + (xb - xa) * t, gy1 + 2 + math.sin(t * math.pi) * sag))
        for x in (gx1 + 4, gx2 - 4):
            for t in np.linspace(0, 1, 110):
                path.append((x + math.sin(t * 22) * 5, gy1 + t * (gy2 - gy1) * 0.78))
        greens = [(28, 62, 20), (36, 82, 28), (48, 102, 38), (62, 122, 52), (85, 145, 75)]
        layer_specs = [(0, 7, 2, 18, 1.4), (1, 6, 2, 16, 0.0), (2, 5, 2, 14, 0.0), (3, 4, 1, 12, 0.0), (4, 3, 1, 9, 0.0)]
        for gi, per, thick, L, blur in layer_specs:  # back (dark, blurry) -> front (light, crisp)
            lay = np.zeros((h, w, 4), np.float32)
            for (x, y) in path:
                for _ in range(per):
                    a = rnd.uniform(0, math.tau)
                    ln = rnd.uniform(L * 0.6, L)
                    ox, oy = rnd.uniform(-5, 5), rnd.uniform(-5, 5)
                    p0 = (int(x + ox), int(y + oy))
                    p1 = (int(x + ox + math.cos(a) * ln), int(y + oy + math.sin(a) * ln * 0.8))
                    col = tuple(c + rnd.randint(-8, 8) for c in greens[gi])
                    cv2.line(lay, p0, p1, (*col, 1.0), thick, AA)
            if blur:
                lay = cv2.GaussianBlur(lay, (0, 0), blur)
            a = lay[..., 3:]
            ov[..., :3] = ov[..., :3] * (1 - a) + lay[..., :3] * a
            ov[..., 3:] = np.maximum(ov[..., 3:], a)

        # pine cones + berry clusters tucked into the garland
        for k, (x, y) in enumerate(path[::23]):
            x, y = int(x), int(y)
            if k % 2 == 0:
                cv2.ellipse(ov, (x, y + 6), (8, 13), rnd.uniform(-30, 30), 0, 360, (30, 55, 95, 1), -1, AA)
                for j in range(5):
                    cv2.ellipse(ov, (x, y + j * 4), (7 - j // 2, 2), 0, 0, 180, (50, 85, 130, 1), 1, AA)
            else:
                for _ in range(3):
                    bx, by = x + rnd.randint(-7, 7), y + rnd.randint(-5, 7)
                    cv2.circle(ov, (bx, by), 5, (30, 30, 185, 1), -1, AA)
                    cv2.circle(ov, (bx - 2, by - 2), 1, (220, 220, 255, 1), -1, AA)

        # velvet bows on the top corners
        for x in (gx1, gx2):
            y = gy1 + 2
            for s in (-1, 1):
                cv2.ellipse(ov, (x + s * 24, y), (26, 15), s * 18, 0, 360, (40, 35, 170, 1), -1, AA)
                cv2.ellipse(ov, (x + s * 24, y - 3), (18, 8), s * 18, 0, 360, (70, 65, 210, 1), -1, AA)
                cv2.line(ov, (x, y), (x + s * 16, y + 50), (35, 30, 150, 1), 9, AA)
                cv2.line(ov, (x + s * 2, y), (x + s * 14, y + 46), (60, 55, 195, 1), 3, AA)
            cv2.circle(ov, (x, y), 11, (35, 30, 155, 1), -1, AA)
            cv2.circle(ov, (x - 3, y - 3), 4, (90, 90, 225, 1), -1, AA)

        # glass ornaments hanging from each swag
        for i in range(n_swags):
            x = int(gx1 + (gx2 - gx1) * (i + 0.5) / n_swags)
            y0 = int(gy1 + 2 + sag)
            y1 = y0 + 26 + (i % 2) * 22
            cv2.line(ov, (x, y0), (x, y1), (60, 160, 200, 1), 1, AA)
            col = np.array([(45, 40, 200), (40, 175, 235), (180, 95, 35), (60, 140, 45)][i % 4], np.float32)
            r = 15
            for rr in range(r, 0, -1):  # radial shading for a glassy ball
                k = (rr / r) ** 0.8
                c = col * (1.25 - 0.55 * k)
                cv2.circle(ov, (x + int((1 - k) * -4), y1 + r - 2 + int((1 - k) * -4)), rr,
                           (*[float(v) for v in np.clip(c, 0, 255)], 1.0), -1, AA)
            cv2.circle(ov, (x - 5, y1 + r - 8), 3, (255, 255, 255, 1), -1, AA)
            cv2.rectangle(ov, (x - 4, y1 - 4), (x + 4, y1 + 1), (60, 175, 215, 1), -1)

        # string lights following the garland
        lights = []
        for k, (x, y) in enumerate(path[::9]):
            lights.append((int(x + rnd.uniform(-4, 4)), int(y + rnd.uniform(4, 10)),
                           LIGHT_COLORS[k % len(LIGHT_COLORS)], rnd.uniform(0, 6.28), rnd.uniform(0.8, 2.2),
                           rnd.uniform(-35, 35)))

        # red pillar candles standing in the snow on the sill
        candles = []
        for x, hgt, r in ((gx1 + 70, 78, 15), (gx1 + 108, 52, 13), (gx2 - 90, 64, 15)):
            y = gy2 + 12
            for i in range(-r, r + 1):  # rounded wax shading
                k = 1 - (i / (r + 1)) ** 2
                c = (40 + 30 * k, 40 + 30 * k, 150 + 70 * k, 1.0)
                cv2.line(ov, (x + i, y - hgt), (x + i, y), c, 1)
            cv2.line(ov, (x - r // 2, y - hgt + 4), (x - r // 2, y - 6), (110, 110, 235, 1.0), 2, AA)  # shine
            cv2.ellipse(ov, (x, y - hgt), (r, 4), 0, 0, 360, (60, 60, 185, 1.0), -1, AA)
            cv2.ellipse(ov, (x, y - hgt), (r - 3, 2), 0, 0, 360, (45, 45, 150, 1.0), -1, AA)
            cv2.line(ov, (x, y - hgt), (x, y - hgt - 7), (25, 25, 25, 1.0), 2, AA)
            cv2.ellipse(ov, (x, y), (r + 8, 6), 0, 0, 360, (250, 248, 246, 1.0), -1, AA)  # snow at the base
            candles.append((x, y - hgt - 13))
        out = np.zeros((h, w, 4), np.uint8)
        out[..., :3] = np.clip(ov[..., :3], 0, 255)
        out[..., 3] = np.clip(ov[..., 3] * 255, 0, 255)
        return out, lights, candles

    def draw_frame(self, out, t):
        out[:] = cv2.blendLinear(out, self.ov_rgb, self.ov_inv, self.ov_a)
        q = 4  # glow is drawn at 1/4 size, then scaled up
        tight = np.zeros((self.h // q, self.w // q, 3), np.uint8)
        wide = np.zeros_like(tight)
        for (x, y, c, ph, sp, ang) in self.lights:
            k = 0.7 + 0.3 * math.sin(t * sp + ph)
            cv2.circle(tight, (x // q, y // q), 3, tuple(int(v * k) for v in c), -1, AA)
            cv2.circle(wide, (x // q, y // q), 5, tuple(int(v * k * 0.5) for v in c), -1, AA)
        for (x, y) in self.candles:
            f = 1 + 0.12 * math.sin(t * 11 + x) + 0.08 * math.sin(t * 6.7 + x * 0.3)
            cv2.circle(tight, (x // q, y // q), int(6 * f), (60, 170, 255), -1, AA)
            cv2.circle(wide, (x // q, (y + 10) // q), int(22 * f), (20, 55, 110), -1, AA)
        glow = cv2.add(cv2.GaussianBlur(tight, (0, 0), 1.6), cv2.GaussianBlur(wide, (0, 0), 7))
        glow = cv2.resize(glow, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        cv2.add(out, glow, out)
        # bulbs
        for (x, y, c, ph, sp, ang) in self.lights:
            k = 0.7 + 0.3 * math.sin(t * sp + ph)
            dx, dy = math.sin(math.radians(ang)), math.cos(math.radians(ang))
            cv2.line(out, (x, y), (int(x - dx * 5), int(y - dy * 5)), (25, 55, 25), 4, AA)  # socket
            col = tuple(int(min(255, 60 + v * 0.85 * k)) for v in c)
            cv2.ellipse(out, (int(x + dx * 4), int(y + dy * 4)), (4, 7), -ang, 0, 360, col, -1, AA)
            core = tuple(int(min(255, 150 + v * 0.4 * k)) for v in c)
            cv2.ellipse(out, (int(x + dx * 4), int(y + dy * 4)), (2, 3), -ang, 0, 360, core, -1, AA)
        # flames
        for (x, y) in self.candles:
            f = 1 + 0.12 * math.sin(t * 11 + x)
            sway = int(1.5 * math.sin(t * 5 + x))
            cv2.ellipse(out, (x + sway, y), (5, int(12 * f)), 0, 0, 360, (70, 190, 255), -1, AA)
            cv2.ellipse(out, (x + sway, y + 3), (3, int(7 * f)), 0, 0, 360, (190, 240, 255), -1, AA)
            cv2.ellipse(out, (x, y + 8), (2, 3), 0, 0, 360, (255, 170, 120), -1, AA)
        # color grade: cozy vignette
        cv2.multiply(out, self.grade, out, scale=1 / 255)
