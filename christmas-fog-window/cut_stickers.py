"""Cut individual stickers out of a sticker sheet image (white background) into transparent PNGs.

Usage:
    python cut_stickers.py path/to/sheet.jpg
Saves them into the my_stickers/ folder (kept out of git). main.py uses that folder automatically.
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def cut(sheet_path, out_dir, min_area_frac=0.004, bg_level=246):
    img = cv2.imread(sheet_path)
    if img is None:
        sys.exit(f"Couldn't read {sheet_path}")
    h, w = img.shape[:2]
    # background = near-white pixels connected to the border (so white snow inside a sticker is kept)
    near_white = (img.min(axis=2) >= bg_level).astype(np.uint8)
    flood = near_white.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    for x, y in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if flood[y, x]:
            cv2.floodFill(flood, mask, (x, y), 2)
    bg = (flood == 2).astype(np.uint8)
    # big enclosed white areas (like the middle of a wreath) are background too
    n0, lab0, st0, _ = cv2.connectedComponentsWithStats(near_white)
    for i in range(1, n0):
        if st0[i, cv2.CC_STAT_AREA] > 0.0015 * w * h:
            bg[lab0 == i] = 1
    fg = 1 - bg
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # group nearby pieces of the same sticker
    grouped = cv2.dilate(fg, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(grouped)
    boxes = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_area_frac * w * h or bh < h * 0.05:
            continue
        boxes.append((y, x, bw, bh, i))
    boxes.sort(key=lambda b: (round(b[0] / (h * 0.08)), b[1]))  # reading order
    # split pieces where two stickers barely touch: shrink the shape until it breaks into two big
    # parts, then give every pixel to the nearer part
    pieces = []   # (y, x, bh, bw, mask_in_box)
    for (y, x, bw, bh, i) in boxes:
        m = ((labels[y:y + bh, x:x + bw] == i) & (fg[y:y + bh, x:x + bw] > 0)).astype(np.uint8)
        parts = None
        er = m.copy()
        for _ in range(25):
            er = cv2.erode(er, np.ones((3, 3), np.uint8))
            nn, ll, ss, _ = cv2.connectedComponentsWithStats(er)
            big = [j for j in range(1, nn) if ss[j, cv2.CC_STAT_AREA] > 0.12 * m.sum()]
            if len(big) >= 2:
                parts = (ll, big)
                break
        if parts is None:
            pieces.append((y, x, bh, bw, m))
            continue
        ll, big = parts
        dists = [cv2.distanceTransform((ll != j).astype(np.uint8), cv2.DIST_L2, 5) for j in big]
        owner = np.argmin(np.stack(dists), axis=0)
        for k in range(len(big)):
            sub = (m > 0) & (owner == k)
            ys, xs = np.where(sub)
            pieces.append((y, x, bh, bw, sub.astype(np.uint8)))
    pieces.sort(key=lambda p: (round((p[0] + np.where(p[4])[0].min()) / (h * 0.08)), p[1] + np.where(p[4])[1].min()))
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for k, (by, bx, bh, bw, m) in enumerate(pieces):
        ys, xs = np.where(m)
        pad = 6
        y, x = by + ys.min(), bx + xs.min()
        bh, bw = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        x0, y0 = max(x - pad, 0), max(y - pad, 0)
        x1, y1 = min(x + bw + pad, w), min(y + bh + pad, h)
        piece_mask = np.zeros((y1 - y0, x1 - x0), np.float32)
        full = np.zeros((h, w), np.uint8)
        full[by:by + m.shape[0], bx:bx + m.shape[1]] = m
        piece_mask[:] = full[y0:y1, x0:x1]
        # keep only the main shape (drops stray specks like a watermark)
        nn, ll, ss, _ = cv2.connectedComponentsWithStats((piece_mask > 0).astype(np.uint8))
        if nn > 2:
            big = ss[1:, cv2.CC_STAT_AREA].max()
            for j in range(1, nn):
                if ss[j, cv2.CC_STAT_AREA] < big * 0.02:
                    piece_mask[ll == j] = 0
        # soften the edge: fade out pixels that are almost background white
        crop = img[y0:y1, x0:x1].astype(np.float32)
        whiteness = np.clip((crop.min(axis=2) - 225) / (bg_level - 225), 0, 1)
        edge = cv2.GaussianBlur(piece_mask, (0, 0), 1.2)
        alpha = np.clip(edge * (1 - 0.6 * whiteness * (edge < 0.99)), 0, 1)
        alpha = cv2.GaussianBlur(alpha, (0, 0), 0.6)
        out = np.zeros((y1 - y0, x1 - x0, 4), np.uint8)
        out[..., :3] = crop.astype(np.uint8)
        out[..., 3] = (alpha * 255).astype(np.uint8)
        path = os.path.join(out_dir, f"{k + 1:02d}.png")
        cv2.imwrite(path, out)
        saved.append(path)
    return saved


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    files = cut(sys.argv[1], os.path.join(HERE, "my_stickers"))
    print(f"Saved {len(files)} stickers to my_stickers/")
