"""
Classical computer-vision colony detection (fallback engine).

Used when no trained model is present. Detects colonies by thresholding,
with two modes chosen automatically from the plate's appearance, and a
dedicated path for gridded counting plates (grid removal by inpainting).

Returns colony coordinates in the processed image's pixel space:
  { "colonies":[{x,y,r}], "count", "width", "height", "mode", "dish_found" }
"""
import cv2
import numpy as np

from grid_detect import detect_colonies_gridded

WORK_EDGE = 1400


def _has_grid(gray, dish_mask):
    """Detect a printed counting grid: long, thin, straight, dark lines."""
    w = gray.shape[1]
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    dark = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 25, 8)
    dark = cv2.bitwise_and(dark, dish_mask)
    line_len = int(w * 0.08)
    horiz = cv2.morphologyEx(dark, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (line_len, 1)))
    vert = cv2.morphologyEx(dark, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_len)))
    grid = cv2.bitwise_or(horiz, vert)
    dish_area = max(int((dish_mask > 0).sum()), 1)
    return (grid > 0).sum() / dish_area > 0.0025


def _resize_for_work(img):
    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge <= WORK_EDGE:
        return img, 1.0
    scale = WORK_EDGE / long_edge
    return cv2.resize(img, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_AREA), scale


def _find_dish(gray):
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    _, bright = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    cnts, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    if not cnts:
        mask[:] = 255
        return mask, False
    big = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(big) < 0.15 * gray.size:
        mask[:] = 255
        return mask, False
    (cx, cy), rad = cv2.minEnclosingCircle(big)
    cv2.circle(mask, (int(cx), int(cy)), int(rad * 0.90), 255, -1)
    return mask, True


def _auto_mode(gray, dish_mask, hsv):
    inside_gray = gray[dish_mask > 0]
    inside_sat = hsv[:, :, 1][dish_mask > 0]
    if inside_gray.size == 0:
        return "dark_on_light"
    if np.median(inside_sat) > 80 or np.median(inside_gray) < 90:
        return "bright_on_dark"
    return "dark_on_light"


def _threshold(img, gray, hsv, dish_mask, mode):
    if mode == "bright_on_dark":
        H, S, V = cv2.split(hsv)
        inside_v = V[dish_mask > 0]
        inside_s = S[dish_mask > 0]
        v_cut = max(np.percentile(inside_v, 75), 180)
        s_cut = min(np.percentile(inside_s, 20), np.median(inside_s) * 0.6)
        m = ((V > v_cut) & (S < s_cut)).astype(np.uint8) * 255
    else:
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, m = cv2.threshold(blur, 0, 255,
                             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    m = cv2.bitwise_and(m, dish_mask)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return m


def _split_touching(contour, min_area, max_splits=4):
    """
    Try to split one blob into multiple colonies using convexity defects:
    where round colonies touch, the merged outline dents sharply inward at
    the "waist" between them - a real corner, not just the normal pixel-
    level jaggedness of a single colony's edge. This is the same shape
    signature a human eye uses to tell "two circles overlapping" from "one
    blob" at a glance, just computed directly off the contour instead of
    walking tangents by hand: cv2.convexityDefects already reports exactly
    where a contour caves inward away from its own convex hull, and how
    deep each cave is.

    Recurses on each half after a successful split, since 3+ merged
    colonies need more than one cut. Falls back to returning the blob
    whole whenever no confident split is found (a single real colony's
    defects, if any, are shallow and get filtered out here) - so this can
    only ever recover colonies that detect_colonies() used to drop
    entirely for looking non-circular or oversized, never make things
    worse for a blob that was already a single clean colony.
    """
    area = cv2.contourArea(contour)
    if area < min_area * 2 or max_splits <= 0:
        return [contour]

    # Raw pixel contours are jagged enough that convexity defects would
    # fire on every few-pixel bump; simplifying first keeps only real
    # corners, matching "the tangent changing more than the usual variance".
    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
    if len(approx) < 4:
        return [contour]
    try:
        hull_idx = np.sort(cv2.convexHull(approx, returnPoints=False).flatten())
        defects = cv2.convexityDefects(approx, hull_idx)
    except cv2.error:
        return [contour]
    if defects is None:
        return [contour]

    # Scale the "how deep is a real notch" threshold to this blob's own
    # size, so it adapts to both large and small colonies rather than one
    # fixed pixel value.
    equiv_r = (area / np.pi) ** 0.5
    depth_thresh = max(0.15 * equiv_r, 2.0)
    # OpenCV 4 returns defects as (N, 1, 4), OpenCV 5 as (N, 4); reshape
    # so the unpack below works on either.
    far_points = [tuple(approx[far_idx][0])
                  for _, _, far_idx, depth in defects.reshape(-1, 4)
                  if depth / 256.0 >= depth_thresh]
    if len(far_points) < 2:
        return [contour]

    # Cut the blob's own mask along a line between each pair of nearby
    # concave points - one notch on each side of a neck between two
    # touching colonies. Pairing nearest-first naturally groups the two
    # notches bounding the same neck before considering any others.
    x, y, w, h = cv2.boundingRect(contour)
    pad = 2
    mask = np.zeros((h + pad * 2, w + pad * 2), np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1, offset=(pad - x, pad - y))

    pts = [(px_ - x + pad, py_ - y + pad) for px_, py_ in far_points]
    while len(pts) >= 2:
        best = min(
            ((np.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]), i, j)
             for i in range(len(pts)) for j in range(i + 1, len(pts))),
            key=lambda t: t[0],
        )
        dist, i, j = best
        if dist > equiv_r * 1.5:
            break  # remaining concave points don't pair up - stop cutting
        cv2.line(mask, pts[i], pts[j], 0, 2)
        pts = [p for k, p in enumerate(pts) if k not in (i, j)]

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pieces = [c for c in cnts if cv2.contourArea(c) >= min_area * 0.5]
    if len(pieces) < 2:
        return [contour]

    result = []
    for c in pieces:
        c = c.copy()
        c[:, :, 0] += x - pad
        c[:, :, 1] += y - pad
        result.extend(_split_touching(c, min_area, max_splits - 1))
    return result


def _measure_piece_at(piece, px, py, min_area, min_circularity, max_r):
    """Radius of `piece` if it's a plausible single colony under (px, py)."""
    parea = cv2.contourArea(piece)
    if parea < min_area:
        return None
    perim = cv2.arcLength(piece, True)
    if perim == 0:
        return None
    if 4 * np.pi * parea / (perim * perim) < min_circularity:
        return None
    # Allow the click to sit a couple of pixels outside the fitted outline:
    # a fingertip rarely lands dead-centre, and the threshold can shave a
    # pixel or two off a colony's true edge.
    if cv2.pointPolygonTest(piece, (float(px), float(py)), True) < -2.0:
        return None
    _, r = cv2.minEnclosingCircle(piece)
    if r >= max_r:
        return None
    return float(r)


def _watershed_pieces(mask, image_bgr):
    """Split a foreground mask into per-colony masks with watershed."""
    sure_bg = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    sure_fg = np.zeros_like(mask)
    if dist.max() > 0:
        _, sure_fg = cv2.threshold(dist, 0.40 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image_bgr.copy(), markers)
    return [np.uint8(markers == mid) * 255
            for mid in np.unique(markers) if mid > 1]


_LINE_ANGLES = (0, 30, 60, 90, 120, 150)


def _line_kernel(length, angle_deg):
    """A 1-pixel-wide straight line structuring element at the given angle."""
    k = np.zeros((length, length), np.uint8)
    c = (length - 1) / 2.0
    dx = np.cos(np.deg2rad(angle_deg)) * c
    dy = np.sin(np.deg2rad(angle_deg)) * c
    cv2.line(k, (int(round(c - dx)), int(round(c - dy))),
             (int(round(c + dx)), int(round(c + dy))), 1, 1)
    return k


def _inpaint_grid_lines(crop):
    """
    Paint out any printed grid lines crossing the crop, the same way the
    whole-plate gridded path does, so a colony sitting on a line isn't
    merged with it into one long non-round blob. Lines are long, thin and
    dark: an opening with a kernel longer than any colony keeps only them.
    Returns the crop untouched when no line is found, which keeps plain
    plates on the cheap path.
    """
    h, w = crop.shape[:2]
    line_len = max(int(0.3 * min(h, w)), 15)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    dark = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 25, 8)
    # Plates are rarely photographed square to the grid, so look for lines
    # at several angles, not just horizontal and vertical.
    grid = np.zeros_like(dark)
    for angle in _LINE_ANGLES:
        grid = cv2.bitwise_or(grid, cv2.morphologyEx(dark, cv2.MORPH_OPEN,
                                                     _line_kernel(line_len, angle)))
    if not grid.any():
        return crop
    grid = cv2.dilate(grid, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.inpaint(crop, grid, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


def _agar_distance_mask(crop):
    """
    Foreground mask by colour distance from the agar around the crop's
    border (median Lab colour), Otsu-thresholded. Returns the 8-bit
    distance map too, so callers can re-cut it at a different level.
    """
    h, w = crop.shape[:2]
    blur = cv2.GaussianBlur(crop, (3, 3), 0)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = max(2, min(h, w) // 8)
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(lab - bg, axis=2)
    d8 = np.clip(dist * (255.0 / max(float(dist.max()), 1e-6)), 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return d8, mask


def _fit_circle(contour):
    """
    Least-squares circle through a contour's points (Kasa fit). For a
    piece cut off a merged blob this recovers the colony's real radius
    from its remaining outer arc, where the minimum enclosing circle
    would be pulled outward by the straight cut edge. Falls back to the
    enclosing circle if the fit is degenerate.
    """
    pts = contour.reshape(-1, 2).astype(np.float64)
    if len(pts) < 3:
        (x, y), r = cv2.minEnclosingCircle(contour)
        return float(x), float(y), float(r)
    A = np.column_stack([2 * pts[:, 0], 2 * pts[:, 1], np.ones(len(pts))])
    b = (pts ** 2).sum(axis=1)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        (x, y), r = cv2.minEnclosingCircle(contour)
        return float(x), float(y), float(r)
    cx, cy, c = sol
    r2 = c + cx * cx + cy * cy
    (ex, ey), er = cv2.minEnclosingCircle(contour)
    if not np.isfinite(r2) or r2 <= 0:
        return float(ex), float(ey), float(er)
    r = float(np.sqrt(r2))
    # A fit that drifts far from the piece is a sign of a degenerate arc.
    if r > 1.5 * er or np.hypot(cx - ex, cy - ey) > er:
        return float(ex), float(ey), float(er)
    return float(cx), float(cy), r


def _disc_mean_lab(lab, cx, cy, r):
    h, w = lab.shape[:2]
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, (int(round(cx)), int(round(cy))), max(int(round(r)), 1), 255, -1)
    sel = lab[m > 0]
    return sel.mean(axis=0) if len(sel) else None


def pieces_like_anchor(crop, anchor, pieces, max_rel_diff=0.5):
    """
    Keep only the recovered pieces whose colour matches the anchored
    colony's. The model vouched for the anchor's appearance; a real
    neighbour fused to it is the same kind of thing, whereas the usual
    false recoveries - the shadow halo between pale colonies, a grid
    junction, a rim fragment - differ from it in colour by a good
    fraction of the anchor's own contrast against the agar.
    """
    if not pieces:
        return []
    lab = cv2.cvtColor(cv2.GaussianBlur(crop, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    b = max(2, min(h, w) // 8)
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    agar = np.median(border, axis=0)
    ax, ay, ar = anchor
    a_col = _disc_mean_lab(lab, ax, ay, 0.6 * ar)
    if a_col is None:
        return []
    a_contrast = float(np.linalg.norm(a_col - agar))
    if a_contrast < 4:
        return []
    kept = []
    for cx, cy, r in pieces:
        p_col = _disc_mean_lab(lab, cx, cy, 0.6 * r)
        if p_col is None:
            continue
        if np.linalg.norm(p_col - a_col) <= max_rel_diff * a_contrast:
            kept.append((cx, cy, r))
    return kept


def split_merged_blob(crop, min_r, min_circularity=0.5, anchor=None, anchor_r=None):
    """
    Given a crop around one detection the model thinks is a single (but
    suspiciously large or elongated) colony, try to resolve it into the
    individual colonies it actually contains. This is the MCount idea -
    cut the outline at its concave corners, then fit circles to the
    pieces - applied only inside a box the model has already vouched
    for, so printed grid lines and the dish rim never reach it.

    Returns a list of (cx, cy, r) in crop coordinates, or [] when no
    confident multi-colony reading exists, in which case the caller
    keeps the model's original detection. Confidence means at least two
    pieces, each roundish and no smaller than min_r (the plate's own
    smallest colonies, so a lobe on one irregular colony can't pass as
    a second one). With `anchor` (x, y) given, only the connected blob
    containing that point is considered, so a box's neighbours that are
    separate blobs (and so separately visible to the model) are ignored
    and only colonies *fused* to the anchored one are recovered. With
    `anchor_r` also given, a blob no bigger than that one colony is
    left alone outright - there is nothing merged in it to recover.
    """
    if crop is None or crop.size == 0:
        return []
    h, w = crop.shape[:2]
    crop = _inpaint_grid_lines(crop)
    d8, mask = _agar_distance_mask(crop)
    min_area = max(np.pi * min_r * min_r * 0.5, 5.0)

    if anchor is not None:
        ax = min(max(int(round(anchor[0])), 0), w - 1)
        ay = min(max(int(round(anchor[1])), 0), h - 1)
        # Otsu can put faint colonies on the background side when
        # something high-contrast (a bubble, a leftover line) is in the
        # crop. The anchor is a colony the model vouched for, so if it
        # isn't foreground, re-cut at half its own contrast instead.
        if mask[ay, ax] == 0:
            anchor_contrast = np.median(d8[max(0, ay - 2):ay + 3, max(0, ax - 2):ax + 3])
            if anchor_contrast < 8:
                return []
            _, mask = cv2.threshold(d8, max(anchor_contrast * 0.5, 4), 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    if anchor is not None:
        n_lab, labels = cv2.connectedComponents(mask)
        lab_id = labels[ay, ax]
        if lab_id == 0:
            # Anchor fell on background (threshold missed the colony centre,
            # e.g. a translucent colony): take the nearest blob within a
            # small radius, else give up.
            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                return []
            dd = np.hypot(xs - ax, ys - ay)
            k = int(np.argmin(dd))
            if dd[k] > max(min_r * 2, 4):
                return []
            lab_id = labels[ys[k], xs[k]]
        mask = np.uint8(labels == lab_id) * 255
        if anchor_r is not None and (mask > 0).sum() < 1.3 * np.pi * anchor_r * anchor_r:
            return []

    found = []
    for m in _watershed_pieces(mask, crop):
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) < min_area:
                continue
            for piece in _split_touching(c, min_area):
                parea = cv2.contourArea(piece)
                perim = cv2.arcLength(piece, True)
                if parea < min_area or perim == 0:
                    continue
                if 4 * np.pi * parea / (perim * perim) < min_circularity:
                    continue
                cx, cy, r = _fit_circle(piece)
                if r < min_r or r > 0.6 * min(h, w):
                    continue
                if not (0 <= cx < w and 0 <= cy < h):
                    continue
                found.append((cx, cy, r))

    if len(found) < 2:
        return []
    # Drop near-duplicate circles (watershed and the convexity cut can
    # both report the same colony).
    kept = []
    for cx, cy, r in sorted(found, key=lambda t: -t[2]):
        if all(np.hypot(cx - kx, cy - ky) >= 0.6 * max(r, kr) for kx, ky, kr in kept):
            kept.append((cx, cy, r))
    return kept if len(kept) >= 2 else []


def _probe_local_contrast(crop, px, py, min_area, min_circularity):
    """
    Segment the crop by *colour distance from the surrounding agar* rather
    than by the whole-plate brightness/saturation rules. The crop's border
    is almost always agar (the colony is in the middle, under the click),
    so its median Lab colour is a robust background estimate, and anything
    far from that colour is colony - whether it's paler, darker, or a
    different hue than the agar. Otsu picks the cut; if the clicked pixel
    itself falls below it (a faint colony next to a strong one), the cut is
    lowered to half the click's own contrast so the faint one still counts.
    """
    h, w = crop.shape[:2]
    crop = _inpaint_grid_lines(crop)
    d8, mask = _agar_distance_mask(crop)

    cx = min(max(int(round(px)), 0), w - 1)
    cy = min(max(int(round(py)), 0), h - 1)
    if mask[cy, cx] == 0:
        click_contrast = np.median(d8[max(0, cy - 1):cy + 2, max(0, cx - 1):cx + 2])
        if click_contrast < 8:
            return {"ok": False}  # click is on plain agar
        _, mask = cv2.threshold(d8, max(click_contrast * 0.5, 4), 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    max_r = min(h, w) / 3
    for m in _watershed_pieces(mask, crop):
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < min_area or area > 0.5 * h * w:
                continue
            if cv2.pointPolygonTest(c, (float(px), float(py)), True) < -2.0:
                continue
            for piece in _split_touching(c, min_area):
                r = _measure_piece_at(piece, px, py, min_area, min_circularity, max_r)
                if r is not None:
                    return {"ok": True, "r": r}
    return {"ok": False}


def _probe_plate_threshold(crop, px, py, min_area, min_circularity):
    """
    Fallback: the same brightness/saturation threshold detect_colonies()
    uses on a whole plate, applied to the crop without a dish mask or an
    upper area cap (the click already says "there's a real colony here").
    """
    h, w = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    full_mask = np.full((h, w), 255, np.uint8)
    mode = _auto_mode(gray, full_mask, hsv)
    thresh = _threshold(crop, gray, hsv, full_mask, mode)
    max_r = min(h, w) / 3
    for m in _watershed_pieces(thresh, crop):
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < min_area or area > 0.7 * h * w:
                continue
            for piece in _split_touching(c, min_area):
                r = _measure_piece_at(piece, px, py, min_area, min_circularity, max_r)
                if r is not None:
                    return {"ok": True, "r": r}
    return {"ok": False}


def probe_colony_size(image_crop_bgr, px, py, min_area=5, min_circularity=0.45):
    """
    Measure the real size of whatever colony sits under (px, py) in a small
    crop centred on a manually-placed mark, so the mark can be auto-sized
    from the actual blob instead of a generic default box.

    Two segmentations are tried in order. The first measures each pixel's
    colour distance from the agar around the crop's edge - it doesn't care
    whether colonies are paler or darker than the agar, which the whole-
    plate rules in _threshold() have to guess from plate-wide statistics
    that a small crop doesn't have. On the labelled dataset that lifted the
    hit rate from ~32% to ~66% of colonies with ~8% false fits on empty
    agar. The second is that whole-plate threshold as a fallback, which
    picks up a few percent more.

    Either way a click can land on one colony of a touching pair/cluster,
    so blobs are watershed- and convexity-split before measuring, and the
    piece reported is the one actually containing the click. min_area is
    much lower than detect_colonies' default (~1.3px radius vs ~5.3px):
    scanning a whole plate at that sensitivity would pick up dust as
    colonies, but here the user already pointed at a spot they trust is a
    real, if pinhead-sized, colony. Returns {"ok": False} whenever neither
    pass yields a confident, click-containing, roundish contour, so the
    caller can fall back to the default box size.
    """
    if image_crop_bgr is None or image_crop_bgr.size == 0:
        return {"ok": False}
    res = _probe_local_contrast(image_crop_bgr, px, py, min_area, min_circularity)
    if res["ok"]:
        return res
    return _probe_plate_threshold(image_crop_bgr, px, py, min_area, min_circularity)


def detect_colonies(image_bgr, min_area=90, max_area=8000,
                    min_circularity=0.55, mode="auto"):
    work, _scale = _resize_for_work(image_bgr)
    h, w = work.shape[:2]
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)

    dish_mask, dish_found = _find_dish(gray)

    if mode == "auto" and _has_grid(gray, dish_mask):
        return detect_colonies_gridded(image_bgr, min_area=max(min_area, 40))

    if mode == "auto":
        mode = _auto_mode(gray, dish_mask, hsv)

    thresh = _threshold(work, gray, hsv, dish_mask, mode)

    sure_bg = cv2.dilate(thresh, np.ones((3, 3), np.uint8), iterations=3)
    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    if dist.max() > 0:
        _, sure_fg = cv2.threshold(dist, 0.40 * dist.max(), 255, 0)
    else:
        sure_fg = np.zeros_like(thresh)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(work.copy(), markers)

    colonies = []
    for mid in np.unique(markers):
        if mid <= 1:
            continue
        m = np.uint8(markers == mid) * 255
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            # Densely-packed plates are exactly where colonies most often
            # touch closely enough that watershed alone doesn't separate
            # them, leaving one blob that's non-circular or bigger than a
            # single colony should be - which used to just get dropped
            # here rather than split. Try to split any such blob before
            # applying the per-colony area/shape checks below, so this
            # can only recover colonies, not lose ones that were already
            # fine (the max_area cap only ever applied per-colony below).
            for piece in _split_touching(c, min_area):
                area = cv2.contourArea(piece)
                if area < min_area or area > max_area:
                    continue
                perim = cv2.arcLength(piece, True)
                if perim == 0:
                    continue
                circ = 4 * np.pi * area / (perim * perim)
                if circ < min_circularity:
                    continue
                (x, y), r = cv2.minEnclosingCircle(piece)
                colonies.append({"x": int(x), "y": int(y), "r": max(int(r), 5)})

    return {
        "colonies": colonies,
        "count": len(colonies),
        "width": w,
        "height": h,
        "mode": mode,
        "dish_found": dish_found,
    }
