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

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "model", "best.pt")
MODEL_URL = os.environ.get(
    "MODEL_URL",
    "https://github.com/williamarnold708/colony-counter/releases/download/model-v1.0/best.pt",
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


def detect_colonies_model(image_bgr, conf=0.25):
    import cv2

    model = _get_model()
    if model is None:
        raise RuntimeError("Model not available")

    h0, w0 = image_bgr.shape[:2]
    long_edge = max(h0, w0)
    scale = WORK_EDGE / long_edge if long_edge > WORK_EDGE else 1.0
    work = cv2.resize(image_bgr, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_AREA) if scale != 1.0 else image_bgr
    h, w = work.shape[:2]

    # iou=0.45 tightens YOLO's own non-max suppression so heavily overlapping
    # boxes are merged before they reach us.
    results = model.predict(work, conf=conf, imgsz=1024, iou=0.45, verbose=False)
    colonies = []
    if results:
        for b in results[0].boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            r = max((x2 - x1), (y2 - y1)) / 2.0
            colonies.append({"x": int(cx), "y": int(cy), "r": max(int(r), 5)})

    # Second pass: our own centre-distance dedupe, to catch any duplicates that
    # slip through at very low confidence thresholds.
    colonies = _dedupe(colonies)

    return {
        "colonies": colonies,
        "count": len(colonies),
        "width": w,
        "height": h,
        "mode": "model",
        "dish_found": True,
    }
