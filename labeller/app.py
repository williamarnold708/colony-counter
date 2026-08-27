"""
Colony labelling tool — builds a YOLO training dataset from plate photos.

Drop plate images into inbox/, run this, and mark every colony on each plate.
Auto-detection pre-fills what it can (so you confirm rather than click from
scratch). Labels are written in YOLO format, ready for training.

Produces:
    data/images/<name>.jpg   data/labels/<name>.txt   data/classes.txt
    data/progress.json
"""
import os
import json
import base64
import sys

import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "app"))
try:
    from detector import detect_colonies
    HAVE_DETECTOR = True
except Exception:
    HAVE_DETECTOR = False

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
IMAGES = os.path.join(DATA, "images")
LABELS = os.path.join(DATA, "labels")
INBOX = os.path.join(BASE, "inbox")
PROGRESS = os.path.join(DATA, "progress.json")
CLASSES_FILE = os.path.join(DATA, "classes.txt")

DEFAULT_CLASSES = ["colony"]
DEFAULT_BOX_FRAC = 0.030

for d in (IMAGES, LABELS, INBOX):
    os.makedirs(d, exist_ok=True)

app = Flask(__name__)


def load_classes():
    if os.path.exists(CLASSES_FILE):
        with open(CLASSES_FILE) as f:
            names = [l.strip() for l in f if l.strip()]
        if names:
            return names
    save_classes(DEFAULT_CLASSES)
    return list(DEFAULT_CLASSES)


def save_classes(names):
    with open(CLASSES_FILE, "w") as f:
        f.write("\n".join(names) + "\n")


def load_progress():
    if os.path.exists(PROGRESS):
        try:
            with open(PROGRESS) as f:
                return json.load(f)
        except Exception:
            pass
    return {"done": []}


def save_progress(p):
    with open(PROGRESS, "w") as f:
        json.dump(p, f, indent=2)


def inbox_images():
    exts = (".jpg", ".jpeg", ".png")
    return sorted(f for f in os.listdir(INBOX) if f.lower().endswith(exts))


@app.route("/")
def index():
    return render_template("label.html")


@app.route("/api/state")
def state():
    prog = load_progress()
    files = inbox_images()
    done = set(prog.get("done", []))
    todo = [f for f in files if f not in done]
    return jsonify({
        "total": len(files), "done": len(done), "remaining": len(todo),
        "next": todo[0] if todo else None,
        "classes": load_classes(), "have_detector": HAVE_DETECTOR,
    })


@app.route("/api/image/<name>")
def get_image(name):
    path = os.path.join(INBOX, name)
    if not os.path.exists(path):
        return jsonify({"error": "Not found"}), 404
    img = cv2.imread(path)
    if img is None:
        return jsonify({"error": "Could not read that image"}), 400

    long_edge = max(img.shape[:2])
    scale = 1400 / long_edge if long_edge > 1400 else 1.0
    work = cv2.resize(img, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_AREA) if scale != 1.0 else img
    h, w = work.shape[:2]

    points = []
    if HAVE_DETECTOR and request.args.get("prefill", "1") == "1":
        try:
            res = detect_colonies(img)
            sx = w / res["width"]
            sy = h / res["height"]
            for c in res["colonies"]:
                points.append({"x": c["x"] * sx, "y": c["y"] * sy, "cls": 0,
                               "r": c["r"] * sx})
        except Exception:
            points = []

    stem = os.path.splitext(name)[0]
    label_path = os.path.join(LABELS, stem + ".txt")
    if os.path.exists(label_path):
        points = []
        with open(label_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls, xc, yc, bw, bh = parts[:5]
                points.append({"x": float(xc) * w, "y": float(yc) * h,
                               "cls": int(cls),
                               "r": max(float(bw) * w, float(bh) * h) / 2})

    ok, buf = cv2.imencode(".jpg", work, [cv2.IMWRITE_JPEG_QUALITY, 88])
    b64 = base64.b64encode(buf).decode("ascii")
    return jsonify({"name": name, "image": "data:image/jpeg;base64," + b64,
                    "width": w, "height": h, "points": points})


@app.route("/api/save", methods=["POST"])
def save():
    data = request.get_json(force=True)
    name = data.get("name")
    points = data.get("points", [])
    width = float(data.get("width", 0))
    height = float(data.get("height", 0))
    if not name or width <= 0 or height <= 0:
        return jsonify({"error": "Bad save request"}), 400
    src = os.path.join(INBOX, name)
    if not os.path.exists(src):
        return jsonify({"error": "Source image missing"}), 404

    stem = os.path.splitext(name)[0]
    img = cv2.imread(src)
    long_edge = max(img.shape[:2])
    scale = 1400 / long_edge if long_edge > 1400 else 1.0
    work = cv2.resize(img, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_AREA) if scale != 1.0 else img
    cv2.imwrite(os.path.join(IMAGES, stem + ".jpg"), work,
                [cv2.IMWRITE_JPEG_QUALITY, 92])

    default_box = DEFAULT_BOX_FRAC * max(width, height)
    lines = []
    for p in points:
        xc = min(max(float(p["x"]) / width, 0.0), 1.0)
        yc = min(max(float(p["y"]) / height, 0.0), 1.0)
        cls = int(p.get("cls", 0))
        r = p.get("r")
        box = max(float(r) * 2, 6) if r else default_box
        bw = box / width
        bh = box / height
        lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    with open(os.path.join(LABELS, stem + ".txt"), "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))

    prog = load_progress()
    if name not in prog["done"]:
        prog["done"].append(name)
    save_progress(prog)
    return jsonify({"ok": True, "saved": len(lines)})


@app.route("/api/skip", methods=["POST"])
def skip():
    data = request.get_json(force=True)
    name = data.get("name")
    prog = load_progress()
    if name and name not in prog["done"]:
        prog["done"].append(name)
    save_progress(prog)
    return jsonify({"ok": True})


@app.route("/api/classes", methods=["POST"])
def set_classes():
    data = request.get_json(force=True)
    names = [n.strip() for n in data.get("classes", []) if n.strip()]
    if not names:
        return jsonify({"error": "Need at least one class"}), 400
    save_classes(names)
    return jsonify({"ok": True, "classes": names})


if __name__ == "__main__":
    print(f"Inbox:   {INBOX}")
    print(f"Dataset: {DATA}")
    print(f"Auto-detect prefill: {'on' if HAVE_DETECTOR else 'OFF'}")
    app.run(debug=True, host="0.0.0.0", port=5001)
