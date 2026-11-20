"""
Pre-place: propose a mark for every colony that looks like the ones already
marked, so a dense plate becomes "correct these" rather than "click 400 times".

It learns from the marks already on the plate (the model's prefill, a saved
label, or a few colonies you click first) instead of guessing what a colony
looks like - plates vary too much (white on blue, orange on clear, dark on
yellow, printed grids, pen writing) for one fixed rule:

1. Colony colour = the colour inside the example marks; agar colour = the most
   common colour in a ring just outside them. Every pixel is scored by how far
   it lies from the agar *towards the colony colour* (0 = agar, 1 = colony),
   minus a penalty for colours off that line (ink, grid lines, shadows).
2. The dish = the agar-coloured region with its holes (the colonies) filled in.
3. Printed grid lines are painted out first so their crossings aren't blobs.
4. Multi-scale difference-of-Gaussians on the score, at scales matching the
   examples' sizes; a local maximum is a candidate colony of radius ~1.4s.
   The bar it must clear is set from how strongly the examples themselves
   respond, and elongated responses (lines, writing) are rejected with SIFT's
   edge test.

Everything it proposes is meant to be checked: the labeller draws these marks
dotted until you've saved the plate.
"""
import cv2
import numpy as np

from detector import _line_kernel

_LINE_ANGLES = (0, 30, 60, 90, 120, 150)
MIN_EXAMPLES = 3


def _remove_grid(work):
    """Inpaint long thin dark lines (a printed counting grid). Found at half size for speed."""
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(small, (5, 5), 0)
    dark = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 15, 6)
    line_len = max(int(0.06 * min(small.shape)), 15)
    grid = np.zeros_like(dark)
    for a in _LINE_ANGLES:
        grid = cv2.bitwise_or(grid, cv2.morphologyEx(dark, cv2.MORPH_OPEN, _line_kernel(line_len, a)))
    if (grid > 0).mean() < 0.002:
        return work, None
    grid = cv2.resize(grid, (work.shape[1], work.shape[0]), interpolation=cv2.INTER_NEAREST)
    grid = cv2.dilate(grid, np.ones((3, 3), np.uint8), iterations=2)
    return cv2.inpaint(work, grid, inpaintRadius=5, flags=cv2.INPAINT_TELEA), grid > 0


def _dish_from_rim(work, ex):
    """Fallback dish: the smallest strong circle (the dish rim) that contains the examples."""
    h, w = work.shape[:2]
    small = cv2.medianBlur(cv2.resize(cv2.cvtColor(work, cv2.COLOR_BGR2GRAY), None, fx=0.5, fy=0.5,
                                      interpolation=cv2.INTER_AREA), 5)
    m = min(small.shape)
    circles = cv2.HoughCircles(small, cv2.HOUGH_GRADIENT, dp=1.5, minDist=20, param1=80, param2=35,
                               minRadius=int(0.25 * m), maxRadius=int(0.55 * m))
    best = None
    if circles is not None:
        for cx, cy, r in circles[0]:
            cx, cy, r = cx * 2, cy * 2, r * 2
            inside = np.mean([(x - cx) ** 2 + (y - cy) ** 2 < (0.95 * r) ** 2 for x, y, _ in ex])
            if inside >= 0.9 and (best is None or r < best[2]):
                best = (cx, cy, r)
    if best is None:
        return np.ones((h, w), bool)
    yy, xx = np.mgrid[0:h, 0:w]
    return (xx - best[0]) ** 2 + (yy - best[1]) ** 2 < (0.93 * best[2]) ** 2


def _mode_colour(px):
    """Most common colour (coarse 3-D histogram peak) of an (N, 3) Lab array."""
    q = (px // 5).astype(np.int64)
    keys = q[:, 0] * 1_000_000 + q[:, 1] * 1000 + q[:, 2]
    vals, counts = np.unique(keys, return_counts=True)
    return px[keys == vals[np.argmax(counts)]].mean(axis=0)


def propose_colonies(work, examples, max_marks=3000, sized=None):
    """
    work: BGR image at the labeller's working size.
    examples: [(x, y, r), ...] marks already on the plate - used to learn what a
    colony looks like here, and never duplicated.
    sized: optional [bool, ...] - which examples have a real (measured or
    hand-set) size, used to calibrate the proposed sizes.
    Returns (marks, info); marks = [{"x", "y", "r"}, ...] in work pixels.
    """
    ex = [(float(x), float(y), max(float(r), 2.0)) for x, y, r in examples]
    sized = list(sized) if sized is not None else [True] * len(ex)
    if len(ex) < MIN_EXAMPLES:
        return [], {"reason": f"need at least {MIN_EXAMPLES} marked colonies to learn from"}
    h, w = work.shape[:2]
    clean, grid_mask = _remove_grid(work)
    had_grid = grid_mask is not None
    lab = cv2.cvtColor(cv2.GaussianBlur(clean, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]

    # ---- learn colony and agar colours from the examples ----
    inside_any = np.zeros((h, w), bool)
    col_samples = []
    for x, y, r in ex:
        x0, x1, y0, y1 = int(max(x - r, 0)), int(min(x + r + 1, w)), int(max(y - r, 0)), int(min(y + r + 1, h))
        if x1 <= x0 or y1 <= y0:
            continue
        d2 = (xx[y0:y1, x0:x1] - x) ** 2 + (yy[y0:y1, x0:x1] - y) ** 2
        core = d2 <= (0.5 * r) ** 2
        if core.any():
            col_samples.append(np.median(lab[y0:y1, x0:x1][core], axis=0))
        inside_any[y0:y1, x0:x1] |= d2 <= (1.15 * r) ** 2
    if len(col_samples) < MIN_EXAMPLES:
        return [], {"reason": "examples fall outside the image"}
    c_col = np.median(np.array(col_samples), axis=0)

    ring = np.zeros((h, w), bool)
    for x, y, r in ex:
        R1, R2 = 1.5 * r + 2, 2.6 * r + 4
        x0, x1, y0, y1 = int(max(x - R2, 0)), int(min(x + R2 + 1, w)), int(max(y - R2, 0)), int(min(y + R2 + 1, h))
        d2 = (xx[y0:y1, x0:x1] - x) ** 2 + (yy[y0:y1, x0:x1] - y) ** 2
        ring[y0:y1, x0:x1] |= (d2 >= R1 ** 2) & (d2 <= R2 ** 2)
    ring &= ~inside_any
    ring_px = lab[ring]
    if ring_px.shape[0] < 200:
        return [], {"reason": "not enough bare agar around the marked colonies"}
    # Drop ring pixels that are themselves colony-coloured (unmarked neighbours) before taking the mode.
    far = np.linalg.norm(ring_px - c_col, axis=1) > 6
    c_ag = _mode_colour(ring_px[far] if far.sum() > 200 else ring_px)

    axis = c_col - c_ag
    L = float(np.linalg.norm(axis))
    if L < 12:
        return [], {"reason": "the marked colonies are too close in colour to the agar for pre-place to tell them apart"}
    diff = lab - c_ag
    t = (diff @ axis) / (L * L)                                  # 0 = agar, 1 = colony colour
    off = np.linalg.norm(diff - t[..., None] * axis, axis=2) / L  # distance off the agar->colony line
    score = np.clip(t - 0.8 * off, -1.0, 1.5).astype(np.float32)

    # ---- the dish: agar-coloured region with its holes (colonies) filled ----
    agar_like = (np.linalg.norm(diff, axis=2) < max(10.0, 0.45 * L)).astype(np.uint8) * 255
    agar_like = cv2.morphologyEx(agar_like, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(agar_like, 8)
    if n > 1:
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        region = (lbl == big).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dish = np.zeros((h, w), np.uint8)
        cv2.drawContours(dish, cnts, -1, 255, -1)               # fills the colony holes
        dish = cv2.erode(dish, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
        dish = dish > 0
        # The examples must be inside it; if most aren't, the colour guess failed.
        if np.mean([dish[int(min(y, h - 1)), int(min(x, w - 1))] for x, y, _ in ex]) < 0.6:
            dish = _dish_from_rim(work, ex)
    else:
        dish = _dish_from_rim(work, ex)
    score[~dish] = 0

    # ---- multi-scale blob response at the examples' sizes ----
    rs = np.array([r for _, _, r in ex])
    rmin, rmax = max(1.5, 0.55 * np.percentile(rs, 5)), 1.6 * np.percentile(rs, 95)
    sigmas = []
    s = rmin / 1.42
    while s <= rmax / 1.42 * 1.0001:
        sigmas.append(s)
        s *= 1.25
    sigmas = sigmas or [rmin / 1.42]
    stack = np.stack([cv2.GaussianBlur(score, (0, 0), s) - cv2.GaussianBlur(score, (0, 0), 1.6 * s) for s in sigmas])
    best = stack.argmax(axis=0)
    M = stack.max(axis=0)

    # Bar to clear: a fraction of how strongly the examples themselves respond.
    # Size calibration: the blob scale tends to read touching colonies small, so
    # compare it with the examples whose size was actually measured or set by hand.
    ex_resp, ratios = [], []
    for (x, y, r), was_sized in zip(ex, sized):
        xi, yi = int(round(x)), int(round(y))
        rad = max(2, int(0.3 * r))
        y0, x0 = max(yi - rad, 0), max(xi - rad, 0)
        win = M[y0:yi + rad + 1, x0:xi + rad + 1]
        if win.size:
            ex_resp.append(float(win.max()))
            if was_sized:
                wy, wx = np.unravel_index(int(np.argmax(win)), win.shape)
                ratios.append(r / (sigmas[best[y0 + wy, x0 + wx]] * 1.42))
    thr = max(0.02, 0.35 * float(np.median(ex_resp)))
    size_ratio = float(np.clip(np.median(ratios), 0.8, 1.8)) if len(ratios) >= 2 else 1.0

    peaks = (M == cv2.dilate(M, np.ones((5, 5), np.uint8))) & (M > thr) & dish
    ys, xs = np.nonzero(peaks)
    cands = []
    for x, y in zip(xs, ys):
        k = best[y, x]
        R_s = stack[k]
        x0, x1, y0, y1 = max(x - 1, 0), min(x + 1, w - 1), max(y - 1, 0), min(y + 1, h - 1)
        dxx = R_s[y, x1] + R_s[y, x0] - 2 * R_s[y, x]
        dyy = R_s[y1, x] + R_s[y0, x] - 2 * R_s[y, x]
        dxy = (R_s[y1, x1] - R_s[y1, x0] - R_s[y0, x1] + R_s[y0, x0]) / 4
        tr, det = dxx + dyy, dxx * dyy - dxy * dxy
        if det <= 0 or tr * tr / det > 12.1:                    # a ridge (line, writing), not a blob
            continue
        if t[y, x] < 0.35:                                       # centre not colony-coloured enough
            continue
        if had_grid and grid_mask[y, x]:                          # sits on a grid line: usually a crossing
            continue
        cands.append((float(M[y, x]), int(x), int(y), float(sigmas[k] * 1.42)))

    cands.sort(reverse=True)
    kept = []
    for m, x, y, r in cands:
        if any((x - kx) ** 2 + (y - ky) ** 2 < (0.8 * min(r, kr)) ** 2 for _, kx, ky, kr in kept):
            continue
        kept.append((m, x, y, r))
        if len(kept) >= max_marks:
            break

    out = []
    for _, x, y, r in kept:
        if any((x - exx) ** 2 + (y - eyy) ** 2 < (0.7 * max(min(r, er), 4)) ** 2 for exx, eyy, er in ex):
            continue
        out.append({"x": x, "y": y, "r": round(r * size_ratio, 1)})
    return out, {"examples": len(ex), "grid_removed": bool(had_grid), "found": len(kept), "new": len(out),
                 "colour_contrast": round(L, 1), "size_ratio": round(size_ratio, 2),
                 "scales_px": [round(float(s) * 1.42, 1) for s in sigmas]}
