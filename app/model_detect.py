"""
YOLO model-backed colony detection.

Uses a trained detector (best.pt from the training notebook) to find colonies.
Returns the same coordinate contract as the classical detector, so the app and
its click-correction UI work unchanged.

The model file lives at app/model/best.pt. If it's missing locally, it's
downloaded from the GitHub Release asset at MODEL_URL on first use. If that
also fails, is_available() returns False and the app falls back to classical CV.
"""
import os
import math
import urllib.request

import numpy as np

from detector import split_merged_blob, pieces_like_anchor

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "model", "best.pt")
# Default is the v2.0 release (retrained with copy-paste augmentation). To
# roll back without a code change, set MODEL_URL to the model-v1.0 release.
MODEL_URL = os.environ.get(
    "MODEL_URL",
    "https://github.com/williamarnold708/colony-counter/releases/download/model-v2.0/best.pt",
)

_model = None
_load_failed = False


def _ensure_downloaded():
    if os.path.exists(MODEL_PATH):
        return True
    if not MODEL_URL:
        return False
    print(f"[model_detect] downloading model from {MODEL_URL}")
    tmp_path = f"{MODEL_PATH}.{os.getpid()}.part"
    try:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, tmp_path)
        os.replace(tmp_path, MODEL_PATH)
        print("[model_detect] model download complete")
        return True
    except Exception as e:
        print(f"[model_detect] model download failed: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False


def is_available():
    if _load_failed:
        return False
    return os.path.exists(MODEL_PATH) or bool(MODEL_URL)


def _get_model():
    global _model, _load_failed
    if _model is not None:
        return _model
    if _load_failed:
        return None
    if not _ensure_downloaded():
        _load_failed = True
        return None
    try:
        from ultralytics import YOLO
        _model = YOLO(MODEL_PATH)
        return _model
    except Exception as e:
        print(f"[model_detect] could not load model: {e}")
        _load_failed = True
        return None


def ensure_model_ready():
    """Load (downloading if needed) the model at startup, so the first
    request isn't the one paying for the download/load. Safe to call even
    when the model can't be obtained; the app falls back to classical CV."""
    return _get_model() is not None


WORK_EDGE = 1400


def _dedupe(colonies, iou_like=0.6):
    """
    Collapse detections that sit on top of each other (one colony marked
    twice). Two marks are treated as the same colony when the distance
    between their centres is small relative to their radii. Keeps the first
    (higher-confidence, since YOLO returns them confidence-ordered).
    """
    kept = []
    for c in colonies:
        dup = False
        for k in kept:
            dist = math.hypot(c["x"] - k["x"], c["y"] - k["y"])
            # If centres are closer than ~60% of the larger radius, same colony.
            thresh = max(c["r"], k["r"]) * iou_like
            if dist < thresh:
                dup = True
                break
        if not dup:
            kept.append(c)
    return kept


# Hybrid split stage: on by default, HYBRID_SPLIT=0 turns it off so the
# two behaviours can be compared side by side (e.g. local vs. deployed).
HYBRID_SPLIT_DEFAULT = os.environ.get("HYBRID_SPLIT", "0") not in ("0", "false", "off")
# A neighbour recovered by the split stage must be a plausible colony for
# this plate: radius within these multiples of the plate's median radius.
SPLIT_R_MIN = 0.4
SPLIT_R_MAX = 1.6
# Most extra colonies one box may contribute (a lawn fragment could
# otherwise explode into dozens of "colonies").
SPLIT_MAX_PER_BOX = 4
SPLIT_MIN_DETECTIONS = 3


# Inference resolution. The model was trained at 1024. It was run at 640
# for speed on Render's CPU, but at 640 a colony on a dense plate is only
# a few pixels wide and most are missed; on the labelled set 1024 cut the
# overall count error by 15% (dense-plate recall 0.86 -> 0.93) for about
# twice the inference time. So 1024 is the default for every plate.
#
# For a slow host there is a two-tier option: set INFERENCE_IMGSZ=640 and
# DENSE_IMGSZ=1024, and a plate whose fast first pass finds at least
# DENSE_TRIGGER colonies gets a second pass at the higher size. (On the
# labelled set this recovers most, not all, of the always-1024 gain: the
# first pass undercounts medium plates so some never trigger.)
BASE_IMGSZ = int(os.environ.get("INFERENCE_IMGSZ", "1024"))
DENSE_IMGSZ = int(os.environ.get("DENSE_IMGSZ", "1024"))
DENSE_TRIGGER = int(os.environ.get("DENSE_TRIGGER", "12"))
# Ultralytics caps detections per image at 300 by default, which silently
# truncated dense plates. Raise it well past any countable plate.
MAX_DET = 3000
# Above this many colonies the plate is outside the range where counts are
# reliable by any method (standard practice is 25-250 per plate); the UI
# shows a too-numerous-to-count advisory.
TNTC_COUNT = 250


def raw_detections(work_bgr, conf=0.25, imgsz=None):
    """Run the model on an already work-sized image; boxes as dicts."""
    model = _get_model()
    if model is None:
        raise RuntimeError("Model not available")
    if imgsz is None:
        imgsz = BASE_IMGSZ
    # iou=0.45 tightens YOLO's own non-max suppression so heavily overlapping
    # boxes are merged before they reach us.
    results = model.predict(work_bgr, conf=conf, imgsz=imgsz, iou=0.45,
                            max_det=MAX_DET, verbose=False)
    dets = []
    if results:
        for b in results[0].boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            dets.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                         "conf": float(b.conf[0])})
    return dets


def _to_colony(d):
    cx = (d["x1"] + d["x2"]) / 2.0
    cy = (d["y1"] + d["y2"]) / 2.0
    r = max((d["x2"] - d["x1"]), (d["y2"] - d["y1"])) / 2.0
    return {"x": int(cx), "y": int(cy), "r": max(int(r), 5)}


# A fused neighbour's centre sits about (r_anchor + r_piece) from the
# anchor's centre. Anything much closer than that is a fragment *of* the
# anchored colony (its shadow halo, a highlight) rather than a colony
# beside it, and is rejected.
SPLIT_MIN_SEP = 0.85
# A neighbour must also be a reasonable fraction of the anchor's own size;
# a small fragment next to a big colony is nearly always halo or debris.
SPLIT_REL_ANCHOR_MIN = 0.5


def hybrid_split(work_bgr, dets, r_min=SPLIT_R_MIN, r_max=SPLIT_R_MAX,
                 max_per_box=SPLIT_MAX_PER_BOX, min_detections=SPLIT_MIN_DETECTIONS,
                 min_sep=SPLIT_MIN_SEP, rel_anchor_min=SPLIT_REL_ANCHOR_MIN):
    """
    Second stage over the model's boxes. On dense plates YOLO tends to box
    one colony of a touching pair and lose the other entirely: the box it
    keeps is a normal size, so box geometry alone can't tell. Instead,
    each box's surroundings are segmented geometrically (the MCount idea:
    threshold, cut the blob at its concave corners, fit circles to the
    pieces), and any round piece that no existing detection already
    covers is added as a recovered colony. Because this only ever runs
    inside the neighbourhood of a box the model vouched for, grid lines
    and the dish rim never get a chance to be counted.

    Returns (colonies, n_added).
    """
    colonies = [_to_colony(d) for d in dets]
    if len(dets) < min_detections:
        return colonies, 0
    h, w = work_bgr.shape[:2]
    med_r = float(np.median([c["r"] for c in colonies]))
    lo, hi = max(r_min * med_r, 2.0), r_max * med_r

    added = []
    for d in dets:
        bw, bh = d["x2"] - d["x1"], d["y2"] - d["y1"]
        side = max(bw, bh)
        pad = 0.75 * side
        x0 = int(max(0, math.floor(d["x1"] - pad)))
        y0 = int(max(0, math.floor(d["y1"] - pad)))
        x1 = int(min(w, math.ceil(d["x2"] + pad)))
        y1 = int(min(h, math.ceil(d["y2"] + pad)))
        cx_box = (d["x1"] + d["x2"]) / 2.0 - x0
        cy_box = (d["y1"] + d["y2"]) / 2.0 - y0
        r_box = side / 2.0
        crop = work_bgr[y0:y1, x0:x1]
        pieces = split_merged_blob(crop, lo, anchor=(cx_box, cy_box), anchor_r=r_box)
        pieces = pieces_like_anchor(crop, (cx_box, cy_box, r_box), pieces)
        n_box = 0
        for cx, cy, r in pieces:
            if not (lo <= r <= hi) or r < rel_anchor_min * r_box:
                continue
            if math.hypot(cx - cx_box, cy - cy_box) < min_sep * (r_box + r):
                continue
            gx, gy = cx + x0, cy + y0
            # Skip pieces the model already found (or this pass already added).
            if any(math.hypot(gx - c["x"], gy - c["y"]) < 0.7 * max(r, c["r"])
                   for c in colonies + added):
                continue
            added.append({"x": int(gx), "y": int(gy), "r": max(int(r), 5),
                          "stage": "split"})
            n_box += 1
            if n_box >= max_per_box:
                break
    return colonies + added, len(added)


def detect_colonies_model(image_bgr, conf=0.25, hybrid=None):
    import cv2

    if hybrid is None:
        hybrid = HYBRID_SPLIT_DEFAULT

    h0, w0 = image_bgr.shape[:2]
    long_edge = max(h0, w0)
    scale = WORK_EDGE / long_edge if long_edge > WORK_EDGE else 1.0
    work = cv2.resize(image_bgr, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_AREA) if scale != 1.0 else image_bgr
    h, w = work.shape[:2]

    dets = raw_detections(work, conf=conf, imgsz=BASE_IMGSZ)
    imgsz_used = BASE_IMGSZ
    if len(dets) >= DENSE_TRIGGER and DENSE_IMGSZ > BASE_IMGSZ:
        dets = raw_detections(work, conf=conf, imgsz=DENSE_IMGSZ)
        imgsz_used = DENSE_IMGSZ
    if hybrid:
        colonies, added = hybrid_split(work, dets)
    else:
        colonies, added = [_to_colony(d) for d in dets], 0

    # Second pass: our own centre-distance dedupe, to catch any duplicates that
    # slip through at very low confidence thresholds (and, with the hybrid
    # stage on, a split piece that lands on a neighbouring box).
    colonies = _dedupe(colonies)

    return {
        "colonies": colonies,
        "count": len(colonies),
        "width": w,
        "height": h,
        "mode": "model",
        "dish_found": True,
        "split_added": added,
        "imgsz": imgsz_used,
        "tntc": len(colonies) > TNTC_COUNT,
    }
