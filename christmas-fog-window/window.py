"""A realistic winter window: dark painted frame seen at a slight angle, frost creeping in from the
edges, snow piled on the outside sill, snow falling outside, warm firelight from the left and cool
blue light from outside."""
import math
import random

import cv2
import numpy as np

AA = cv2.LINE_AA
WARM = np.array([70, 150, 255], np.float32)       # firelight (BGR)
COOL = np.array([255, 215, 175], np.float32)      # winter daylight (BGR)

# window layout in "window space" (0..1), before the perspective warp
SILL_TOP = 0.875
PANES = [(0.055, 0.05, 0.70, 0.135), (0.722, 0.05, 0.98, 0.135),
         (0.055, 0.158, 0.70, 0.835), (0.722, 0.158, 0.98, 0.835)]


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

        # perspective: window space -> screen (left side closer, right side cropped and receding)
        self.WW, self.WH = int(w * 1.1), int(h * 1.1)
        src = np.float32([[0, 0], [self.WW, 0], [self.WW, self.WH], [0, self.WH]])
        dst = np.float32([[-0.03 * w, -0.05 * h], [1.24 * w, 0.03 * h],
                          [1.24 * w, 0.95 * h], [-0.03 * w, 1.07 * h]])
        self.H = cv2.getPerspectiveTransform(src, dst)

        self._build_frame(rng)
        self._build_outside(rng)

    # ------------------------------------------------ helpers
    def _warp(self, img):
        return cv2.warpPerspective(img, self.H, (self.w, self.h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def _px(self, fx, fy):
        return int(fx * self.WW), int(fy * self.WH)

    # ------------------------------------------------ frame, frost, sill snow
    def _build_frame(self, rng):
        WW, WH = self.WW, self.WH
        glass = np.zeros((WH, WW), np.uint8)
        for (x1, y1, x2, y2) in PANES:
            cv2.rectangle(glass, self._px(x1, y1), self._px(x2, y2), 255, -1)
        frame = 255 - glass
        sill = np.zeros((WH, WW), np.uint8)
        cv2.rectangle(sill, self._px(0, SILL_TOP), self._px(1, 1), 255, -1)

        # painted charcoal frame with a little brush texture
        albedo = np.zeros((WH, WW, 3), np.float32)
        albedo[:] = (58, 60, 66)
        tex = (_noise(WH, WW, WH // 3, WW // 60, rng) - 0.5) * 10 + (rng.random((WH, WW)) - 0.5) * 4
        albedo += tex[..., None]
        albedo[sill > 0] *= 1.15

        # shape lighting: bevels catch light, corners next to the glass are in shadow
        dist = cv2.distanceTransform(frame, cv2.DIST_L2, 5)
        ao = 0.45 + 0.55 * np.clip(dist / (WW * 0.012), 0, 1) ** 0.8
        ring = ((dist > 1.5) & (dist < 4.5) & (frame > 0)).astype(np.float32)
        ring = cv2.GaussianBlur(ring, (0, 0), 1.0)
        top_edge = np.zeros((WH, WW), np.float32)
        cv2.line(top_edge, self._px(0, SILL_TOP), self._px(1, SILL_TOP), 1.0, 3)
        top_edge = cv2.GaussianBlur(top_edge, (0, 0), 1.5)
        gap = ((dist > 0) & (dist <= 1.6)).astype(np.float32)

        # frost creeping in from the pane edges
        gdist = cv2.distanceTransform(glass, cv2.DIST_L2, 5)
        grains = rng.random((WH, WW)).astype(np.float32)
        clumps = _noise(WH, WW, WH // 14, WW // 14, rng)
        fine = _noise(WH, WW, WH // 4, WW // 4, rng)
        reach = WW * 0.035 * (0.5 + clumps)
        prob = np.exp(-gdist / reach) * (glass > 0)
        frost_a = (grains < prob * 0.9).astype(np.float32) * 0.9
        frost_a = np.maximum(frost_a, np.clip(prob * 1.4 - 0.55, 0, 1) * (0.6 + 0.4 * fine))
        frost_a = cv2.GaussianBlur(frost_a, (0, 0), 0.6)

        # snow piled on the outside sill of each pane
        snow_a = np.zeros((WH, WW), np.float32)
        for (x1, y1, x2, y2) in PANES:
            px1, py1 = self._px(x1, y1)
            px2, py2 = self._px(x2, y2)
            hgt = (py2 - py1) * (0.07 if y2 - y1 > 0.3 else 0.12)
            xs = np.arange(px1, px2)
            ph = rng.random() * 6
            top = py2 - hgt * (0.65 + 0.35 * np.sin(xs * 0.021 + ph) * np.sin(xs * 0.0063 + 2)
                               + 0.12 * rng.random(len(xs)))
            for x, ty in zip(xs, top):
                snow_a[int(ty):py2, x] = 1.0
        snow_edge = cv2.GaussianBlur(snow_a, (0, 0), 2.0)
        speck = (grains > 0.55).astype(np.float32)
        snow_a = np.clip(snow_edge * (0.82 + 0.18 * speck), 0, 1)

        # ---- warp everything onto the screen
        W = self._warp
        self.glass_mask = W(glass) > 127
        frame_s = W(frame).astype(np.float32) / 255
        albedo_s = W(albedo)
        ao_s, ring_s, edge_s, gap_s = W(ao), W(ring), W(top_edge), W(gap)
        frost_s, snow_s = W(frost_a), W(snow_a)

        cool = self.cool[..., None]
        warm = self.warm[..., None]
        base = albedo_s * ao_s[..., None] * (0.55 * cool * COOL / 255 + 0.2)
        base += ring_s[..., None] * (COOL * 0.35 * cool + WARM * 0.5 * warm)
        base += edge_s[..., None] * (COOL * 0.25 + WARM * 0.6 * warm)
        base *= (1 - gap_s[..., None] * 0.7)
        fire = albedo_s * ao_s[..., None] * WARM / 255 * warm * 1.3
        fire += ring_s[..., None] * WARM * warm * 0.4
        self.frame_base = base.astype(np.float32)
        self.frame_fire = fire.astype(np.float32)
        self.frame_a = np.clip(frame_s, 0, 1).astype(np.float32)
        self.frame_inv = (1 - self.frame_a).astype(np.float32)

        ice = np.zeros((self.h, self.w, 3), np.float32)
        ice[:] = (252, 246, 240)
        ice = ice * (0.82 + 0.18 * cool) + WARM * 0.35 * warm
        ice += (np.random.default_rng(9).random((self.h, self.w, 1)) > 0.992) * 60
        a = np.clip(np.maximum(frost_s * 0.85, snow_s), 0, 1)
        self.ice_rgb = np.clip(ice, 0, 255).astype(np.uint8)
        self.ice_a = a.astype(np.float32)
        self.ice_inv = (1 - self.ice_a).astype(np.float32)

        big = np.zeros((WH, WW), np.uint8)
        x1, y1, x2, y2 = PANES[2]
        cv2.rectangle(big, self._px(x1, y1), self._px(x2, y2), 255, -1)
        bx, by, bw, bh = cv2.boundingRect((W(big) > 127).astype(np.uint8))
        self.glass = (bx, by, min(bx + bw, self.w), min(by + bh, self.h))

        # warm reflection of the fireplace on the glass
        spr = np.zeros((self.h // 2, self.w // 2), np.float32)
        cx, cy = int(self.w * 0.12 / 2), int(self.h * 0.66 / 2)
        for i in range(7):
            cv2.ellipse(spr, (cx + int(rng.normal(0, 6)), cy - i * 9), (24 - i * 3, 34 - i * 3), 0, 0, 360,
                        0.35 + i * 0.05, -1, AA)
        spr = cv2.GaussianBlur(spr, (0, 0), 14)
        spr = cv2.resize(spr, (self.w, self.h))[..., None] * WARM * 0.55
        self.fire_reflection = (spr * self.glass_mask[..., None]).astype(np.float32)

        yy, xx = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        d = np.sqrt(((xx - self.w * 0.5) / (self.w * 0.7)) ** 2 + ((yy - self.h * 0.45) / (self.h * 0.75)) ** 2)
        v = np.clip(1.06 - d ** 2.4 * 0.5, 0.5, 1)
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
        lit = cv2.scaleAdd(self.frame_fire, flicker(t), self.frame_base)
        out[:] = cv2.blendLinear(out, cv2.convertScaleAbs(lit), self.frame_inv, self.frame_a)
        cv2.multiply(out, self.vignette, out, scale=1 / 255)
