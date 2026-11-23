"""Subtle, realistic winter face filters that follow MediaPipe's 478 face landmarks.

Press E to switch: natural -> soft glow -> rosy cheeks -> frosted lashes -> snow in hair -> freckles -> winter glam
"""
import math

import cv2
import numpy as np

FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
             152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LEFT_EYE = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
RIGHT_EYE = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]
LIPS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
UPPER_LIDS = [246, 161, 160, 159, 158, 157, 173, 466, 388, 387, 386, 385, 384, 398]
BROWS = [70, 63, 105, 66, 107, 300, 293, 334, 296, 336]
CHEEKS = (205, 425)
HAIRLINE = [10, 338, 297, 332, 284, 251, 109, 67, 103, 54, 21]
FRECKLE_BASE = [6, 197, 195, 5, 205, 425, 50, 280, 123, 352]

MODES = [
    ("natural", set()),
    ("soft glow", {"glow"}),
    ("rosy cheeks", {"glow", "rosy"}),
    ("frosted lashes", {"glow", "frost"}),
    ("snow in hair", {"glow", "hair"}),
    ("freckles", {"glow", "freckles"}),
    ("winter glam", {"glow", "rosy", "frost", "hair", "freckles"}),
]


class FaceFX:
    def __init__(self, seed=21):
        rng = np.random.default_rng(seed)
        self.mode = 1
        self.changed_at = -10.0
        # face-relative random positions (in units of face width) so details stick to the face
        self.lash_bits = [(i, rng.normal(0, 0.012), -abs(rng.normal(0.01, 0.01)), rng.uniform(0.003, 0.007))
                          for i in UPPER_LIDS for _ in range(3)]
        self.brow_bits = [(i, rng.normal(0, 0.018), rng.normal(-0.01, 0.012), rng.uniform(0.003, 0.006))
                          for i in BROWS for _ in range(3)]
        self.hair_flakes = [(int(rng.choice(HAIRLINE)), rng.normal(0, 0.12), -abs(rng.normal(0.06, 0.06)),
                             rng.uniform(0.008, 0.02), rng.uniform(0, 6.28)) for _ in range(38)]
        self.freckles = [(int(rng.choice(FRECKLE_BASE)), rng.normal(0, 0.05), rng.normal(0, 0.035),
                          rng.uniform(0.0025, 0.005), rng.uniform(0.35, 0.8)) for _ in range(70)]
        self.shimmer = [(int(rng.choice([50, 280, 123, 352])), rng.normal(0, 0.03), rng.normal(0, 0.02),
                         rng.uniform(0, 6.28)) for _ in range(10)]

    def next(self, now):
        self.mode = (self.mode + 1) % len(MODES)
        self.changed_at = now

    @property
    def name(self):
        return MODES[self.mode][0]

    # ------------------------------------------------------------------
    def apply(self, frame, pts, now):
        fx = MODES[self.mode][1]
        if pts is None or not fx:
            return frame
        out = frame.copy()
        fw = float(np.linalg.norm(pts[234] - pts[454])) + 1e-6
        if "glow" in fx:
            self._glow(out, pts, fw)
        if "rosy" in fx:
            self._rosy(out, pts, fw)
        if "freckles" in fx:
            self._freckles(out, pts, fw, now)
        if "frost" in fx:
            self._frost(out, pts, fw, now)
        if "hair" in fx:
            self._hair(out, pts, fw, now)
        return out

    def _skin_mask(self, shape, pts, fw, x0, y0):
        h, w = shape
        m = np.zeros((h, w), np.uint8)
        off = np.array([x0, y0])
        cv2.fillPoly(m, [(pts[FACE_OVAL] - off).astype(np.int32)], 255, cv2.LINE_AA)
        for region in (LEFT_EYE, RIGHT_EYE, LIPS):
            poly = pts[region] - off
            c = poly.mean(axis=0)
            poly = c + (poly - c) * 1.25
            cv2.fillPoly(m, [poly.astype(np.int32)], 0, cv2.LINE_AA)
        for brow in (BROWS[:5], BROWS[5:]):
            cv2.polylines(m, [(pts[brow] - off).astype(np.int32)], False, 0, max(2, int(fw * 0.035)), cv2.LINE_AA)
        return cv2.GaussianBlur(m, (0, 0), max(2, fw * 0.03)).astype(np.float32) / 255

    def _glow(self, out, pts, fw):
        """Smooth skin a little, lift it, and warm it slightly. Keeps eyes, brows and lips sharp."""
        H, W = out.shape[:2]
        x0, y0 = np.maximum(pts[FACE_OVAL].min(axis=0) - 10, 0).astype(int)
        x1, y1 = np.minimum(pts[FACE_OVAL].max(axis=0) + 10, [W, H]).astype(int)
        if x1 - x0 < 20 or y1 - y0 < 20:
            return
        roi = out[y0:y1, x0:x1]
        small = cv2.resize(roi, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        smooth = cv2.bilateralFilter(small, 7, 28, 7)
        smooth = cv2.resize(smooth, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        smooth = smooth * np.float32([1.02, 1.04, 1.07]) + 6
        m = self._skin_mask((y1 - y0, x1 - x0), pts, fw, x0, y0)[..., None] * 0.65
        out[y0:y1, x0:x1] = np.clip(roi * (1 - m) + smooth * m, 0, 255).astype(np.uint8)

    def _rosy(self, out, pts, fw):
        """Soft cold-weather blush on the cheeks and a touch on the nose tip."""
        H, W = out.shape[:2]
        for idx, r, k in ((CHEEKS[0], 0.17, 0.22), (CHEEKS[1], 0.17, 0.22), (4, 0.07, 0.12)):
            c = pts[idx]
            R = int(fw * r * 2.6)
            x0, y0 = int(max(c[0] - R, 0)), int(max(c[1] - R, 0))
            x1, y1 = int(min(c[0] + R, W)), int(min(c[1] + R, H))
            if x1 <= x0 or y1 <= y0:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            a = np.exp(-(((xx - c[0]) / (fw * r)) ** 2 + ((yy - c[1]) / (fw * r * 0.75)) ** 2)) * k
            roi = out[y0:y1, x0:x1].astype(np.float32)
            tint = roi * np.float32([0.82, 0.78, 1.05])            # deepen toward rose
            out[y0:y1, x0:x1] = np.clip(roi * (1 - a[..., None]) + tint * a[..., None], 0, 255).astype(np.uint8)

    def _crystal(self, img, p, r, bright=1.0):
        x, y = int(p[0]), int(p[1])
        r = max(1, int(r))
        col = (int(255 * bright), int(250 * bright), int(240 * bright))
        cv2.line(img, (x - r, y), (x + r, y), col, 1, cv2.LINE_AA)
        cv2.line(img, (x, y - r), (x, y + r), col, 1, cv2.LINE_AA)
        if r > 1:
            d = max(1, int(r * 0.6))
            cv2.line(img, (x - d, y - d), (x + d, y + d), col, 1, cv2.LINE_AA)
            cv2.line(img, (x - d, y + d), (x + d, y - d), col, 1, cv2.LINE_AA)
        cv2.circle(img, (x, y), 1, (255, 255, 255), -1, cv2.LINE_AA)

    def _frost(self, out, pts, fw, now):
        """Tiny ice crystals caught on the upper lashes and brows."""
        layer = out.copy()
        for i, dx, dy, r in self.lash_bits + self.brow_bits:
            p = pts[i] + np.array([dx, dy]) * fw
            self._crystal(layer, p, r * fw, 0.95)
        cv2.addWeighted(layer, 0.85, out, 0.15, 0, out)

    def _hair(self, out, pts, fw, now):
        """Snowflakes resting in the hair along the hairline, slowly twinkling."""
        layer = out.copy()
        up = pts[10] - pts[152]
        up = up / (np.linalg.norm(up) + 1e-6)
        side = np.array([-up[1], up[0]])
        for i, dx, dy, r, ph in self.hair_flakes:
            p = pts[i] + side * dx * fw - up * dy * fw
            b = 0.85 + 0.15 * math.sin(now * 1.5 + ph)
            self._crystal(layer, p, r * fw, b)
            cv2.circle(layer, (int(p[0]), int(p[1])), max(1, int(r * fw * 0.35)), (250, 248, 245), -1, cv2.LINE_AA)
        cv2.addWeighted(layer, 0.8, out, 0.2, 0, out)

    def _freckles(self, out, pts, fw, now):
        """Soft sun-kissed freckles across the nose and cheeks, plus a little shimmer on the cheekbones."""
        H, W = out.shape[:2]
        x0, y0 = np.maximum(pts[FACE_OVAL].min(axis=0) - 5, 0).astype(int)
        x1, y1 = np.minimum(pts[FACE_OVAL].max(axis=0) + 5, [W, H]).astype(int)
        if x1 - x0 < 10 or y1 - y0 < 10:
            return
        layer = out.copy()
        for i, dx, dy, r, k in self.freckles:
            p = pts[i] + np.array([dx, dy]) * fw
            c = out[int(np.clip(p[1], 0, out.shape[0] - 1)), int(np.clip(p[0], 0, out.shape[1] - 1))].astype(np.float32)
            col = tuple(int(v) for v in c * np.float32([0.72, 0.74, 0.82]))
            cv2.circle(layer, (int(p[0]), int(p[1])), max(1, int(r * fw)), col, -1, cv2.LINE_AA)
        soft = cv2.GaussianBlur(layer[y0:y1, x0:x1], (3, 3), 0.6)
        out[y0:y1, x0:x1] = cv2.addWeighted(soft, 0.55, out[y0:y1, x0:x1], 0.45, 0)
        for i, dx, dy, ph in self.shimmer:
            p = pts[i] + np.array([dx, dy]) * fw
            b = max(0.0, math.sin(now * 2.2 + ph))
            if b > 0.5:
                self._crystal(out, p, fw * 0.008 * b, b)
