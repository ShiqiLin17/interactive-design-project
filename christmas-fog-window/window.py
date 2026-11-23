"""A realistic winter window with no frame: frost creeping in from the edges, snow piled on the sill
outside, snow falling, warm firelight from the left and cool blue light from outside."""
import math
import random

import cv2
import numpy as np

AA = cv2.LINE_AA
WARM = np.array([70, 150, 255], np.float32)       # firelight (BGR)


def flicker(t):
    return (0.86 + 0.07 * math.sin(t * 7.1) + 0.05 * math.sin(t * 12.7 + 1.3)
            + 0.03 * math.sin(t * 23.3 + 0.4) + 0.025 * math.sin(t * 3.1))


def _noise(h, w, ch, cw, rng):
    n = rng.random((max(2, ch), max(2, cw))).astype(np.float32)
    return cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)


class RealWindow:
    def __init__(self, w, h, seed=5):
        self.w, self.h = w, h
        self.rnd = random.Random(seed)
        rng = np.random.default_rng(seed)

        # warm light from a fireplace off-screen at the bottom left
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        d = np.sqrt(((xx + 0.18 * w) / w) ** 2 + ((yy - 0.95 * h) / h) ** 2 * 1.4)
        self.warm = np.clip(np.exp(-d * 1.9) * 1.6, 0, 1.2).astype(np.float32)
        self.cool = np.clip(1.0 - self.warm * 0.6, 0.3, 1.0).astype(np.float32)

        self._build_frame(rng)
        self._build_outside(rng)

    def _build_frame(self, rng):
        """No frame: the glass fills the screen. Frost creeps in from the edges, snow piles at the bottom."""
        w, h = self.w, self.h
        glass = np.full((h, w), 255, np.uint8)
        cv2.rectangle(glass, (0, 0), (w - 1, h - 1), 0, 1)
        self.glass_mask = np.ones((h, w), bool)

        # frost grains, thickest at the edges and corners
        gdist = cv2.distanceTransform(glass, cv2.DIST_L2, 5)
        grains = rng.random((h, w)).astype(np.float32)
        clumps = _noise(h, w, h // 14, w // 14, rng)
        fine = _noise(h, w, h // 4, w // 4, rng)
        reach = w * 0.035 * (0.5 + clumps)
        prob = np.exp(-gdist / reach)
        frost_a = (grains < prob * 0.9).astype(np.float32) * 0.9
        frost_a = np.maximum(frost_a, np.clip(prob * 1.4 - 0.55, 0, 1) * (0.6 + 0.4 * fine))
        frost_a = cv2.GaussianBlur(frost_a, (0, 0), 0.6)

        # snow piled on the sill outside, along the bottom of the glass (slightly out of focus)
        snow_a = np.zeros((h, w), np.float32)
        hgt = h * 0.1
        xs = np.arange(w)
        ph = rng.random() * 6
        top = h - hgt * (0.65 + 0.3 * np.sin(xs * 0.013 + ph) * np.sin(xs * 0.0041 + 2)
                         + 0.06 * _noise(1, w, 1, w // 6, rng)[0])
        for x, ty in zip(xs, top):
            snow_a[int(ty):, x] = 1.0
        inside = cv2.distanceTransform((snow_a > 0).astype(np.uint8), cv2.DIST_L2, 5)
        snow_a = cv2.GaussianBlur(snow_a, (0, 0), 2.0)
        self.snow_top = int(top.min())
        # granular snow texture: bright crust on top, blue-grey grains and shadow lower down
        lit = np.clip(inside / (hgt * 0.35), 0, 1)
        grain = cv2.GaussianBlur((grains > 0.55).astype(np.float32), (0, 0), 0.8)
        lumps = _noise(h, w, h // 6, w // 6, rng)
        snow_shade = (1.03 - 0.2 * lit) * (0.9 + 0.1 * grain) * (0.93 + 0.07 * lumps)

        cool = self.cool[..., None]
        warm = self.warm[..., None]
        ice = np.zeros((h, w, 3), np.float32)
        ice[:] = (252, 246, 240)
        ice = ice * (0.8 + 0.2 * cool)
        sel = snow_a > 0.01
        ice[sel] = np.float32([255, 250, 246]) * snow_shade[sel][..., None] * np.float32([1.0, 0.98, 0.95])  # bluish shadow in the drift
        ice += WARM * 0.35 * warm
        ice += (np.random.default_rng(9).random((h, w, 1)) > 0.992) * 60
        # soft cool shadow on the glass just above the drift, so the snow's edge reads clearly
        above = np.zeros((h, w), np.float32)
        for x, ty in zip(xs, top):
            above[max(int(ty) - int(hgt * 0.6), 0):int(ty), x] = 1.0
        above = cv2.GaussianBlur(above, (0, 0), hgt * 0.25) * (1 - snow_a)
        ice[above > 0.01] = (ice[above > 0.01] * 0 + np.float32([200, 185, 172]))
        a = np.clip(np.maximum(np.maximum(frost_a * 0.85, snow_a), above * 0.07), 0, 1)
        self.ice_rgb = np.clip(ice, 0, 255).astype(np.uint8)
        self.ice_a = a.astype(np.float32)
        self.ice_inv = (1 - self.ice_a).astype(np.float32)

        self.glass = (0, 0, w, int(h - hgt))

        # warm reflection of the fireplace on the glass
        spr = np.zeros((h // 2, w // 2), np.float32)
        cx, cy = int(w * 0.1 / 2), int(h * 0.66 / 2)
        for i in range(7):
            cv2.ellipse(spr, (cx + int(rng.normal(0, 6)), cy - i * 9), (24 - i * 3, 34 - i * 3), 0, 0, 360,
                        0.35 + i * 0.05, -1, AA)
        spr = cv2.GaussianBlur(spr, (0, 0), 14)
        self.fire_reflection = (cv2.resize(spr, (w, h))[..., None] * WARM * 0.55).astype(np.float32)

        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        d = np.sqrt(((xx - w * 0.5) / (w * 0.7)) ** 2 + ((yy - h * 0.45) / (h * 0.75)) ** 2)
        v = np.clip(1.05 - d ** 2.4 * 0.32, 0.65, 1)
        self.vignette = np.clip(np.stack([v, v, v], -1) * 255, 0, 255).astype(np.uint8)

    # ------------------------------------------------ the world outside
    def _build_outside(self, rng):
        w, h = self.w, self.h
        warm = self.warm[..., None]
        self.cam_cool = np.clip(np.ones((h, w, 3), np.float32) * np.float32([1.08, 0.99, 0.86]) * 0.86 * 128,
                                0, 255).astype(np.uint8)
        self.cam_warm = np.clip(warm * np.float32([0.10, 0.32, 0.62]) * 255, 0, 255).astype(np.uint8)
        self.cam_lift = (22, 14, 6, 0)
        self.fairy = []
        for _ in range(46):
            self.fairy.append((self.rnd.uniform(0.25, 1.0) * w, self.rnd.uniform(0.05, 0.5) * h,
                               self.rnd.uniform(0, 6.28), self.rnd.uniform(0.6, 1.8), self.rnd.uniform(0.5, 1)))
        self.far = [self._flake(True, self.rnd.uniform(0, h)) for _ in range(170)]
        self.near = [self._flake(False, self.rnd.uniform(0, h)) for _ in range(16)]

    def _flake(self, far, y=None):
        r = self.rnd.choice([1, 1, 1, 2, 2]) if far else self.rnd.uniform(9, 22)
        speed = self.rnd.uniform(20, 45) if far else self.rnd.uniform(50, 95)
        return [self.rnd.uniform(-20, self.w + 20), -30 if y is None else y, r, speed, self.rnd.uniform(0, 6.28)]

    def outside(self, cam, t, dt):
        """Your camera image, graded like a cold winter evening with firelight on you."""
        f = flicker(t)
        gray = cv2.cvtColor(cv2.cvtColor(cam, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        c = cv2.addWeighted(cam, 0.72, gray, 0.28, 0)
        cool = cv2.multiply(c, self.cam_cool, scale=1 / 128)
        warm = cv2.multiply(c, self.cam_warm, scale=1 / 255)
        out = cv2.addWeighted(cool, 1.0, warm, f, 0)
        out = cv2.add(out, self.cam_lift)

        q = 4
        glow = np.zeros((self.h // q, self.w // q, 3), np.uint8)
        for (x, y, ph, sp, k) in self.fairy:
            b = k * (0.65 + 0.35 * math.sin(t * sp + ph))
            cv2.circle(glow, (int(x / q), int(y / q)), 2, (int(60 * b), int(170 * b), int(255 * b)), -1, AA)
        for fl in self.near:
            fl[1] += fl[3] * dt
            fl[0] += math.sin(t * 0.8 + fl[4]) * 18 * dt
            if fl[1] > self.h + 40:
                fl[:] = self._flake(False)
            cv2.circle(glow, (int(fl[0] / q), int(fl[1] / q)), max(1, int(fl[2] / q)), (120, 118, 115), -1, AA)
        glow = cv2.GaussianBlur(glow, (0, 0), 1.6)
        glow = cv2.resize(glow, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        cv2.add(out, glow, out)
        for fl in self.far:
            fl[1] += fl[3] * dt
            fl[0] += math.sin(t * 1.2 + fl[4]) * 10 * dt
            if fl[1] > self.h + 10:
                fl[:] = self._flake(True)
            cv2.circle(out, (int(fl[0]), int(fl[1])), fl[2], (245, 240, 236), -1, AA)
        return out

    def fog_tint(self, t, shape):
        """Color of the fog: cool white, glowing warm near the fire."""
        h, w = shape
        if getattr(self, "_tint_shape", None) != shape:
            warm = cv2.resize(self.warm, (w, h), interpolation=cv2.INTER_AREA)[..., None]
            self._tint_base = np.float32([252, 240, 226]) * (1 - 0.15 * warm)
            self._tint_fire = warm * WARM * 0.8
            self._tint_shape = shape
        return self._tint_base + self._tint_fire * flicker(t)

    # ------------------------------------------------ on the glass + frame
    def glass_fx(self, out, t):
        k = 0.6 + 0.6 * (flicker(t) - 0.8)
        cv2.add(out, cv2.convertScaleAbs(self.fire_reflection, alpha=k), out)

    def draw_ice(self, out):
        out[:] = cv2.blendLinear(out, self.ice_rgb, self.ice_inv, self.ice_a)

    def draw_frame(self, out, t):
        cv2.multiply(out, self.vignette, out, scale=1 / 255)
