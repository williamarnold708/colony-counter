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
            area = cv2.contourArea(c)
            if area < min_area or area > max_area:
                continue
            perim = cv2.arcLength(c, True)
            if perim == 0:
                continue
            circ = 4 * np.pi * area / (perim * perim)
            if circ < min_circularity:
                continue
            (x, y), r = cv2.minEnclosingCircle(c)
            colonies.append({"x": int(x), "y": int(y), "r": max(int(r), 5)})

    return {
        "colonies": colonies,
        "count": len(colonies),
        "width": w,
        "height": h,
        "mode": mode,
        "dish_found": dish_found,
    }
