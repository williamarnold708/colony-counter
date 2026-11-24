"""
Run the colony detector with ONNX Runtime instead of PyTorch.

PyTorch alone takes most of a 512 MB host's memory; ONNX Runtime needs a small
fraction of it, which lets the model run at full resolution (1024) there.

This reproduces Ultralytics' own predict() for a single image, step for step:
letterbox (scale the long edge to imgsz, pad only up to the next multiple of 32,
centred, grey 114) -> RGB, 0..1, NCHW -> model -> confidence filter -> greedy NMS
-> boxes back to the input image (ops.scale_boxes). The Android app uses the same
recipe and matched the Python server mark for mark. Export the .onnx with
tools/export_onnx.py (dynamic shape).
"""
import os

import cv2
import numpy as np

STRIDE = 32
# Threads for one inference. 1 suits a small shared host (0.5 CPU); raise it
# with ORT_THREADS on a machine with cores to spare.
ORT_THREADS = int(os.environ.get("ORT_THREADS", "1"))


class OnnxDetector:
    is_onnx = True

    def __init__(self, path):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        # One request at a time on a small host: keep the thread pool and the
        # memory arena modest.
        opts.intra_op_num_threads = ORT_THREADS
        opts.inter_op_num_threads = 1
        opts.enable_cpu_mem_arena = False
        self.session = ort.InferenceSession(path, sess_options=opts,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, work_bgr, conf=0.25, imgsz=1024, iou=0.45, max_det=3000):
        """Boxes as dicts {x1, y1, x2, y2, conf} in work_bgr pixels, best first."""
        h0, w0 = work_bgr.shape[:2]
        r = min(imgsz / h0, imgsz / w0)
        nw, nh = int(round(w0 * r)), int(round(h0 * r))
        dw = ((imgsz - nw) % STRIDE) / 2.0
        dh = ((imgsz - nh) % STRIDE) / 2.0
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = work_bgr
        if (nw, nh) != (w0, h0):
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                 value=(114, 114, 114))
        h1, w1 = img.shape[:2]
        blob = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0

        out = self.session.run(None, {self.input_name: blob})[0][0]  # (5, anchors)
        scores = out[4]
        keep = scores > conf
        if not np.any(keep):
            return []
        cx, cy, bw, bh = out[0][keep], out[1][keep], out[2][keep], out[3][keep]
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        scores = scores[keep]
        order = np.argsort(-scores, kind="stable")
        boxes, scores = boxes[order], scores[order]
        kept = _nms(boxes, scores, iou)[:max_det]
        boxes, scores = boxes[kept], scores[kept]

        # Back to work-image pixels (Ultralytics ops.scale_boxes).
        gain = min(h1 / h0, w1 / w0)
        pad_x = round((w1 - w0 * gain) / 2 - 0.1)
        pad_y = round((h1 - h0 * gain) / 2 - 0.1)
        boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - pad_x) / gain, 0, w0)
        boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - pad_y) / gain, 0, h0)
        return [{"x1": float(b[0]), "y1": float(b[1]), "x2": float(b[2]), "y2": float(b[3]),
                 "conf": float(s)} for b, s in zip(boxes, scores)]


def _nms(boxes, scores, iou_threshold):
    """Greedy non-maximum suppression; boxes already sorted by score, best first."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    suppressed = np.zeros(len(boxes), bool)
    kept = []
    for i in range(len(boxes)):
        if suppressed[i]:
            continue
        kept.append(i)
        rest = np.arange(i + 1, len(boxes))
        rest = rest[~suppressed[rest]]
        if not len(rest):
            continue
        iw = np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest])
        ih = np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest])
        inter = np.where((iw > 0) & (ih > 0), iw * ih, 0.0)
        union = areas[i] + areas[rest] - inter
        suppressed[rest[inter / union > iou_threshold]] = True
    return kept
