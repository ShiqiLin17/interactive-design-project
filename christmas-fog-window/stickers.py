"""Christmas stickers, drawn with code (no image files needed).

Each sticker is an RGBA image with a white die-cut border, like a real sticker.
"""
import cv2
import numpy as np

SIZE = 200          # drawing canvas (pixels)
BORDER = 9          # white sticker border thickness

# BGR + alpha
RED = (40, 40, 210, 255)
DARK_RED = (30, 25, 150, 255)
GREEN = (60, 140, 40, 255)
DARK_GREEN = (40, 95, 25, 255)
GOLD = (40, 190, 245, 255)
WHITE = (255, 255, 255, 255)
SNOW = (250, 245, 240, 255)
SHADE = (225, 215, 205, 255)
BLACK = (35, 30, 30, 255)
BROWN = (40, 70, 115, 255)
ORANGE = (30, 130, 250, 255)

AA = cv2.LINE_AA


def _canvas():
    return np.zeros((SIZE, SIZE, 4), np.uint8)


def _poly(img, pts, color):
    cv2.fillPoly(img, [np.array(pts, np.int32)], color, AA)


def _circle(img, c, r, color, thickness=-1):
    cv2.circle(img, (int(c[0]), int(c[1])), int(r), color, thickness, AA)


def draw_tree():
    img = _canvas()
    cv2.rectangle(img, (88, 150), (112, 182), BROWN, -1)
    _poly(img, [(100, 60), (40, 160), (160, 160)], DARK_GREEN)
    _poly(img, [(100, 42), (52, 120), (148, 120)], GREEN)
    _poly(img, [(100, 26), (64, 82), (136, 82)], GREEN)
    # garland
    cv2.line(img, (66, 104), (132, 88), GOLD, 4, AA)
    cv2.line(img, (56, 142), (142, 122), GOLD, 4, AA)
    # ornaments
    for (x, y), col in [((80, 96), RED), ((118, 70), GOLD), ((124, 112), RED),
                        ((72, 136), GOLD), ((108, 140), RED), ((140, 150), GOLD),
                        ((96, 112), (200, 120, 60, 255))]:
        _circle(img, (x, y), 6, col)
        _circle(img, (x - 2, y - 2), 2, WHITE)
    # star
    star = []
    for i in range(10):
        r = 17 if i % 2 == 0 else 7
        a = -np.pi / 2 + i * np.pi / 5
        star.append((100 + r * np.cos(a), 24 + r * np.sin(a)))
    _poly(img, star, GOLD)
    return img


def draw_snowman():
    img = _canvas()
    _circle(img, (100, 145), 42, SNOW)
    _circle(img, (100, 90), 32, SNOW)
    _circle(img, (100, 145), 42, SHADE, 2)
    _circle(img, (100, 90), 32, SHADE, 2)
    # hat
    cv2.rectangle(img, (70, 56), (130, 64), BLACK, -1)
    cv2.rectangle(img, (80, 18), (120, 58), BLACK, -1)
    cv2.rectangle(img, (80, 46), (120, 54), RED, -1)
    # face
    _circle(img, (89, 84), 4, BLACK)
    _circle(img, (111, 84), 4, BLACK)
    _poly(img, [(100, 92), (100, 100), (128, 97)], ORANGE)
    for i, x in enumerate(range(86, 116, 7)):
        _circle(img, (x, 107 + (2 if i in (1, 2, 3) else 0)), 2, BLACK)
    # scarf
    cv2.ellipse(img, (100, 118), (30, 8), 0, 0, 360, RED, -1, AA)
    cv2.rectangle(img, (110, 118), (124, 150), RED, -1)
    cv2.line(img, (110, 132), (124, 132), DARK_RED, 3, AA)
    # buttons
    for y in (135, 152, 169):
        _circle(img, (100, y), 4, BLACK)
    return img


def draw_gift():
    img = _canvas()
    cv2.rectangle(img, (40, 90), (160, 180), RED, -1)
    cv2.rectangle(img, (32, 70), (168, 96), DARK_RED, -1)
    cv2.rectangle(img, (90, 70), (110, 180), GOLD, -1)
    cv2.rectangle(img, (40, 128), (160, 140), GOLD, -1)
    # bow
    cv2.ellipse(img, (78, 58), (24, 14), -20, 0, 360, GOLD, -1, AA)
    cv2.ellipse(img, (122, 58), (24, 14), 20, 0, 360, GOLD, -1, AA)
    cv2.ellipse(img, (78, 58), (12, 6), -20, 0, 360, (20, 140, 200, 255), -1, AA)
    cv2.ellipse(img, (122, 58), (12, 6), 20, 0, 360, (20, 140, 200, 255), -1, AA)
    _circle(img, (100, 62), 11, GOLD)
    # polka dots
    for x, y in [(58, 110), (140, 112), (62, 160), (138, 162)]:
        _circle(img, (x, y), 5, (90, 90, 235, 255))
    return img


def draw_candy_cane():
    img = _canvas()
    mask = np.zeros((SIZE, SIZE), np.uint8)
    cv2.ellipse(mask, (100, 60), (32, 32), 0, 180, 360, 255, 22, AA)
    cv2.line(mask, (132, 60), (132, 180), 255, 22, AA)
    cv2.line(mask, (68, 60), (68, 78), 255, 22, AA)
    stripes = np.zeros_like(img)
    stripes[:] = WHITE
    for k in range(-SIZE, 2 * SIZE, 26):
        cv2.line(stripes, (k, 0), (k + SIZE, SIZE), RED, 11, AA)
    img[mask > 0] = stripes[mask > 0]
    img[..., 3] = mask
    # small holly bow
    cv2.ellipse(img, (148, 110), (14, 7), 30, 0, 360, GREEN, -1, AA)
    cv2.ellipse(img, (118, 110), (14, 7), -30, 0, 360, GREEN, -1, AA)
    _circle(img, (133, 108), 6, RED)
    return img


def add_border(img, border=BORDER):
    """Pad and add a white die-cut outline around everything drawn."""
    pad = border + 6
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    alpha = img[..., 3]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * border + 1, 2 * border + 1))
    outline = cv2.dilate(alpha, k)
    outline = cv2.GaussianBlur(outline, (3, 3), 0)
    out = np.zeros_like(img)
    out[..., :3] = 255
    out[..., 3] = outline
    a = alpha[..., None].astype(np.float32) / 255
    out[..., :3] = (img[..., :3] * a + out[..., :3] * (1 - a)).astype(np.uint8)
    return out


# order matters: which finger touches your thumb picks the sticker
STICKERS = {
    "tree": draw_tree,        # thumb + index
    "snowman": draw_snowman,  # thumb + middle
    "gift": draw_gift,        # thumb + ring
    "candy cane": draw_candy_cane,  # thumb + pinky
}


def build_all():
    return {name: add_border(fn()) for name, fn in STICKERS.items()}
