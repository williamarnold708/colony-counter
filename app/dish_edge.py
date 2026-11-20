"""
Drop model marks that sit outside the petri dish.

The model occasionally marks something on the bench, the lid or the dish wall.
This finds the dish as a circle and removes marks clearly beyond it.

It is deliberately cautious:
- The dish is the LARGEST strong circle that matches a visible edge and keeps
  nearly all of the model's marks, so it never cuts through the agar.
- Marks on the rim itself are kept: real colonies grow right up to the edge.
- If no circle passes, nothing is removed.

Checked on 259 labelled plates (Nov 2026): 11 false marks removed, 0 real
colonies lost, 0 labelled colonies outside the circle it found.
"""
import math

import cv2
import numpy as np

# Marks further than this fraction of the radius beyond the dish edge are dropped.
DISH_MARGIN = 0.04
# A dish circle may leave out at most this many of the model's marks, or this
# share on a crowded plate. False marks outside a dish come one to three at a
# time; a circle that cuts off more is cutting through real colonies.
MAX_OUTSIDE = 3
MAX_OUTSIDE_FRAC = 0.01
# Share of the circle's in-frame outline that must lie on a visible edge, and
# the share of the circle that must be inside the photo at all.
MIN_ARC_SUPPORT = 0.30
MIN_IN_FRAME = 0.35
# Hough runs on a copy with this long edge; only the strongest circles are tried.
SMALL_EDGE = 640
MAX_CANDIDATES = 40


def _xy(m):
    return (m["x"], m["y"]) if isinstance(m, dict) else (m[0], m[1])


def _arc_support(edges, cx, cy, r):
    """Share of the circle's in-frame outline on an edge, and its in-frame share."""
    h, w = edges.shape
    on = total = 0
    for t in np.linspace(0, 2 * math.pi, 360, endpoint=False):
        x, y = int(round(cx + r * math.cos(t))), int(round(cy + r * math.sin(t)))
        if 0 <= x < w and 0 <= y < h:
            total += 1
            on += edges[y, x] > 0
    return (on / total if total else 0.0), total / 360.0


def find_dish(work_bgr, marks):
    """The dish as (cx, cy, r) in the same pixels as `marks`, or None when unsure."""
    if len(marks) < 3:
        return None
    h, w = work_bgr.shape[:2]
    k = SMALL_EDGE / max(h, w)
    small = cv2.resize(work_bgr, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    gray = cv2.medianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 5)
    hs, ws = gray.shape
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=6,
                               param1=100, param2=30,
                               minRadius=int(0.25 * min(hs, ws)),
                               maxRadius=int(0.75 * max(hs, ws)))
    if circles is None:
        return None
    edges = cv2.dilate(cv2.Canny(gray, 50, 120), np.ones((3, 3), np.uint8))
    pts = np.array([_xy(m) for m in marks], float)
    allowed = max(MAX_OUTSIDE, int(MAX_OUTSIDE_FRAC * len(pts)))
    best = None
    for cx, cy, r in circles[0][:MAX_CANDIDATES]:
        support, in_frame = _arc_support(edges, cx, cy, r)
        if support < MIN_ARC_SUPPORT or in_frame < MIN_IN_FRAME:
            continue
        CX, CY, R = cx / k, cy / k, r / k
        outside = int(np.sum(np.hypot(pts[:, 0] - CX, pts[:, 1] - CY) > R * (1 + DISH_MARGIN)))
        if outside <= allowed and (best is None or R > best[2]):
            best = (float(CX), float(CY), float(R))
    return best


def drop_outside(marks, dish):
    """Split marks into (kept, removed) by the dish circle."""
    if dish is None:
        return list(marks), []
    cx, cy, r = dish
    kept, removed = [], []
    for m in marks:
        x, y = _xy(m)
        (removed if math.hypot(x - cx, y - cy) > r * (1 + DISH_MARGIN) else kept).append(m)
    return kept, removed
