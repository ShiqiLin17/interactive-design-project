"""Water droplets on the glass.

Each droplet works like a tiny lens: it shows an upside-down, sharper view of what's behind the glass,
with a bright highlight on top and a dark rim. Some drops slide down and leave a clear trail in the fog.
"""
import math
import random

import cv2
import numpy as np

LIGHT = np.array([-0.45, -0.75, 0.5], np.float32)
LIGHT /= np.linalg.norm(LIGHT)
HALF = LIGHT + np.array([0, 0, 1], np.float32)
HALF /= np.linalg.norm(HALF)


def _shade(height, scale):
    """Turn a droplet height field into (multiply, add, alpha) layers."""
    gx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3) / 8
    nz = np.full_like(height, 1.0 / scale)
    norm = np.sqrt(gx * gx + gy * gy + nz * nz) + 1e-6
    nx, ny, nz = -gx / norm, -gy / norm, nz / norm
    inside = (height > 0).astype(np.float32)
    rim = (1 - nz) ** 1.3 * inside                         # steep edges are dark
    spec = np.clip(nx * HALF[0] + ny * HALF[1] + nz * HALF[2], 0, 1) ** 60 * inside
    caustic = np.clip(ny, 0, 1) ** 2 * inside * (1 - rim)  # light focused at the bottom of the drop
    mul = (1.02 - 0.42 * rim + 0.3 * caustic) * inside
    return mul, spec, caustic


class Droplets:
    def __init__(self, w, h, warm_map, seed=11, count=None):
        self.w, self.h = w, h
        rng = np.random.default_rng(seed)
        self.rnd = random.Random(seed)
        count = count or int(w * h / 380)

        height = np.zeros((h, w), np.float32)
        alpha = np.zeros((h, w), np.float32)
        mx, my = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        map_x, map_y = mx.copy(), my.copy()

        for _ in range(count):
            y = h * rng.beta(1.7, 1.1)                  # more (and bigger) drops lower on the glass
            x = rng.uniform(0, w)
            r = float(np.clip(rng.lognormal(math.log(0.9 + 2.8 * y / h), 0.55), 0.6, 11))
            rx, ry = r * rng.uniform(0.85, 1.05), r * rng.uniform(1.0, 1.3)
            cy = y
            if r > 5 and rng.random() < 0.35:            # some heavy drops sag into a teardrop
                ry *= 1.5
            x0, x1 = int(max(x - rx - 2, 0)), int(min(x + rx + 3, w))
            y0, y1 = int(max(cy - ry - 2, 0)), int(min(cy + ry + 3, h))
            if x0 >= x1 or y0 >= y1:
                continue
            xx, yy = mx[y0:y1, x0:x1], my[y0:y1, x0:x1]
            sag = np.clip((yy - cy) / ry, 0, 1) * 0.25    # flatter top, fuller bottom
            d = ((xx - x) / (rx * (1 + sag))) ** 2 + ((yy - cy) / ry) ** 2
            dome = np.sqrt(np.clip(1 - d, 0, 1)) * r
            sel = dome > height[y0:y1, x0:x1]
            height[y0:y1, x0:x1][sel] = dome[sel]
            a = np.clip((1 - d) * r * 1.5, 0, 1)
            alpha[y0:y1, x0:x1] = np.maximum(alpha[y0:y1, x0:x1], a)
            s = 2.2                                       # lens strength: shows a wider, flipped view
            map_x[y0:y1, x0:x1][sel] = (x - (xx - x) * s)[sel]
            map_y[y0:y1, x0:x1][sel] = (cy - (yy - cy) * s - r * 1.5)[sel]

        mul, spec, caustic = _shade(height, 1.6)
        warm = warm_map[..., None]
        hl = np.float32([255, 248, 240]) * (1 - 0.3 * warm) + np.float32([120, 200, 255]) * 0.5 * warm
        self.mul = np.clip(np.repeat(mul[..., None], 3, axis=2) * 160, 0, 255).astype(np.uint8)
        self.add = np.clip(spec[..., None] * hl * 1.1 + caustic[..., None] * 30, 0, 255).astype(np.uint8)
        self.alpha = alpha
        self.map_x, self.map_y = map_x, map_y
        self.runners = []

    # ---------------------------------------------------------------- sliding drops
    def update_runners(self, fog, glass, dt):
        x1, y1, x2, y2 = glass
        if len(self.runners) < 7 and fog.coverage() > 0.25 and self.rnd.random() < dt * 0.9:
            x = self.rnd.uniform(x1 + 20, x2 - 20)
            y = self.rnd.uniform(y1 + 20, y1 + (y2 - y1) * 0.6)
            if fog.value_at(x, y) > 0.6:
                self.runners.append({"x": x, "y": y, "r": self.rnd.uniform(3.5, 6.5), "v": 0.0,
                                     "stop": self.rnd.uniform(y + 80, y2 + 40), "ph": self.rnd.uniform(0, 6)})
        alive = []
        for d in self.runners:
            # stick-slip: drops hesitate, then rush down
            d["v"] = max(0.0, min(160.0, d["v"] + self.rnd.uniform(-90, 130) * dt * 3))
            px, py = d["x"], d["y"]
            d["y"] += d["v"] * dt
            d["x"] += math.sin(d["y"] * 0.03 + d["ph"]) * 0.12
            fog.wipe((px, py), (d["x"], d["y"]), d["r"] * 0.9)
            if d["y"] < min(d["stop"], y2 - 4):
                alive.append(d)
        self.runners = alive

    def _draw_runner(self, out, src, d):
        r = d["r"]
        rx, ry = r, r * 1.35
        x, y = d["x"], d["y"]
        x0, x1 = int(x - rx - 3), int(x + rx + 4)
        y0, y1 = int(y - ry - 3), int(y + ry + 4)
        if x0 < 0 or y0 < 0 or x1 > self.w or y1 > self.h:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dd = ((xx - x) / rx) ** 2 + ((yy - y) / ry) ** 2
        dome = np.sqrt(np.clip(1 - dd, 0, 1)) * r
        mul, spec, caustic = _shade(dome, 1.6)
        a = np.clip((1 - dd) * r * 1.5, 0, 1)[..., None]
        sx = np.clip(x - (xx - x) * 2.2, 0, self.w - 1)
        sy = np.clip(y - (yy - y) * 2.2 - r * 1.5, 0, self.h - 1)
        refr = cv2.remap(src, sx, sy, cv2.INTER_LINEAR).astype(np.float32)
        col = refr * mul[..., None] + spec[..., None] * 280 + caustic[..., None] * 30
        roi = out[y0:y1, x0:x1].astype(np.float32)
        out[y0:y1, x0:x1] = np.clip(roi * (1 - a) + col * a, 0, 255).astype(np.uint8)

    # ---------------------------------------------------------------- draw
    def apply(self, fogged, clear, fog_m):
        """fogged: the glass with fog. clear: the sharp view behind the glass. fog_m: fog amount (0..1)."""
        src = cv2.addWeighted(clear, 0.62, fogged, 0.38, 0)   # drops on fogged glass look a little milky
        refr = cv2.remap(src, self.map_x, self.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        col = cv2.add(cv2.multiply(refr, self.mul, scale=1 / 160), self.add)
        a = self.alpha * (0.3 + 0.7 * fog_m)               # wiping clears most drops away
        out = cv2.blendLinear(fogged, col, 1 - a, a)
        for d in self.runners:
            self._draw_runner(out, clear, d)
        return out
