"""Nice-looking text using system fonts (falls back gracefully on any computer)."""
import os
from functools import lru_cache

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
except ImportError:  # pillow missing: text just won't show
    Image = None

FONTS = {
    "script": [
        "/System/Library/Fonts/Supplemental/SnellRoundhand.ttc",
        "/System/Library/Fonts/Supplemental/Zapfino.ttf",
        "C:/Windows/Fonts/segoesc.ttf",
        "/System/Library/Fonts/Supplemental/Georgia Italic.ttf",
        "C:/Windows/Fonts/georgiai.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
    ],
    "sans": [
        "/System/Library/Fonts/Avenir Next.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ],
}


@lru_cache(maxsize=16)
def _font(style, size):
    for path in FONTS[style]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


@lru_cache(maxsize=128)
def render(text, size=20, color=(255, 255, 255), style="sans", glow=0.6):
    """Return a BGRA numpy image of the text with a soft dark glow behind it."""
    if Image is None:
        return None
    font = _font(style, size)
    l, t, r, b = font.getbbox(text)
    pad = size // 2 + 4
    W, H = r - l + pad * 2, b - t + pad * 2
    shadow = Image.new("L", (W, H), 0)
    ImageDraw.Draw(shadow).text((pad - l, pad - t), text, font=font, fill=255)
    shadow = shadow.filter(ImageFilter.GaussianBlur(size / 6))
    fg = Image.new("L", (W, H), 0)
    ImageDraw.Draw(fg).text((pad - l, pad - t), text, font=font, fill=255)
    a_fg = np.asarray(fg, np.float32) / 255
    a_sh = np.asarray(shadow, np.float32) / 255 * glow
    alpha = a_fg + a_sh * (1 - a_fg)
    rgb = np.zeros((H, W, 3), np.float32)
    rgb[:] = np.array(color[::-1], np.float32)  # color given as RGB, image is BGR
    rgb *= (a_fg / np.maximum(alpha, 1e-6))[..., None]
    out = np.zeros((H, W, 4), np.uint8)
    out[..., :3] = np.clip(rgb + (20 * (1 - a_fg))[..., None], 0, 255)
    out[..., 3] = np.clip(alpha * 255, 0, 255)
    return out
