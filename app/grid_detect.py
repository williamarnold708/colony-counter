"""
Grid-aware colony detection via line inpainting (classical fallback).

For gridded counting plates: detect the printed grid, inpaint it away so
colonies sitting on lines survive, then detect colonies on the clean image.
Warm-coloured colonies are found via the LAB b-channel.
"""
import cv2
import numpy as np

WORK_EDGE = 1400


def _resize_for_work(img):
    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge <= WORK_EDGE:
        return img
    scale = WORK_EDGE / long_edge
    return cv2.resize(img, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_AREA)


def _find_dish(gray, shrink=0.80):
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    _, bright = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
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
    cv2.circle(mask, (int(cx), int(cy)), int(rad * shrink), 255, -1)
    return mask, True


def detect_colonies_gridded(image_bgr, min_area=40, max_area=8000,
                            min_circularity=0.30, b_offset=12):
    work = _resize_for_work(image_bgr)
    h, w = work.shape[:2]
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)

    dish_wide, dish_found = _find_dish(gray, shrink=0.93)
    dish_tight, _ = _find_dish(gray, shrink=0.80)

    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    dark = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 25, 8)
    dark = cv2.bitwise_and(dark, dish_wide)
    line_len = int(w * 0.08)
    horiz = cv2.morphologyEx(dark, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (line_len, 1)))
    vert = cv2.morphologyEx(dark, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_len)))
    grid = cv2.bitwise_or(horiz, vert)
    grid = cv2.dilate(grid, np.ones((3, 3), np.uint8), iterations=2)

    clean = cv2.inpaint(work, grid, inpaintRadius=4, flags=cv2.INPAINT_TELEA)

    lab = cv2.cvtColor(clean, cv2.COLOR_BGR2LAB)
    B = lab[:, :, 2]
    med = np.median(B[dish_tight > 0])
    mask = ((B > med + b_offset) & (dish_tight > 0)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    n, labels, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    colonies = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        comp = (labels == i).astype(np.uint8) * 255
        cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            continue
        c = cs[0]
        perim = cv2.arcLength(c, True)
        if perim == 0:
            continue
        circ = 4 * np.pi * area / (perim * perim)
        if circ < min_circularity:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        colonies.append({"x": int(x), "y": int(y), "r": max(int(r), 6)})

    return {
        "colonies": colonies,
        "count": len(colonies),
        "width": w,
        "height": h,
        "mode": "gridded",
        "dish_found": dish_found,
    }
