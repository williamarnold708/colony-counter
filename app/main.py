import os
import json
import base64
import time
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify

from detector import detect_colonies, _resize_for_work
import model_detect

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
# Corrected plates are saved into the labeller's dataset so they can be used
# in the next retrain. This connects the "using" loop to the "improving" loop.
DATASET_IMAGES = os.path.join(BASE_DIR, "labeller", "data", "images")
DATASET_LABELS = os.path.join(BASE_DIR, "labeller", "data", "labels")
ALLOWED_EXT = {"png", "jpg", "jpeg"}
DEFAULT_BOX_FRAC = 0.030

app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, "templates"),
            static_folder=os.path.join(BASE_DIR, "static"))

# Downloads best.pt from the GitHub Release if it's not already on disk, so
# the model is ready before the first request (works under gunicorn too,
# not just the `python main.py` dev path below).
engine = "trained model" if model_detect.ensure_model_ready() else "classical CV (model unavailable)"
print(f"Colony counter starting. Detection engine: {engine}")


def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("plate_image")
    if not file or file.filename == "":
        return jsonify({"error": "No image received. Choose a plate photo and try again."}), 400
    if not allowed(file.filename):
        return jsonify({"error": "That file type isn't supported. Use a JPG or PNG."}), 400

    data = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Couldn't read that image. Try a different file."}), 400

    try:
        sensitivity = float(request.form.get("sensitivity", 50))
    except ValueError:
        sensitivity = 50
    sensitivity = max(0.0, min(100.0, sensitivity))

    requested = request.form.get("engine", "auto")
    use_model = (requested in ("auto", "model")) and model_detect.is_available()

    def area_for(sens):
        return int(20 + (sens / 100.0) * 2480)

    if use_model:
        conf = 0.10 + (sensitivity / 100.0) * 0.45
        try:
            result = model_detect.detect_colonies_model(img, conf=conf)
        except Exception as e:
            print(f"[analyze] model failed, falling back: {e}")
            result = detect_colonies(img, min_area=area_for(sensitivity))
    else:
        result = detect_colonies(img, min_area=area_for(sensitivity))

    work, _ = _resize_for_work(img)
    ok, buf = cv2.imencode(".jpg", work, [cv2.IMWRITE_JPEG_QUALITY, 88])
    b64 = base64.b64encode(buf).decode("ascii")
    result["image"] = "data:image/jpeg;base64," + b64
    result["engine"] = "model" if use_model else "classical"
    return jsonify(result)


@app.route("/save_for_training", methods=["POST"])
def save_for_training():
    """
    Save a corrected plate (image + final colony marks) into the labeller's
    dataset in YOLO format, so it's included in the next retrain. This is how
    corrections in the counter feed back into improving the model.
    """
    payload = request.get_json(force=True)
    img_data = payload.get("image", "")
    points = payload.get("points", [])
    width = float(payload.get("width", 0))
    height = float(payload.get("height", 0))
    if not img_data.startswith("data:image") or width <= 0 or height <= 0:
        return jsonify({"error": "Bad save request"}), 400

    os.makedirs(DATASET_IMAGES, exist_ok=True)
    os.makedirs(DATASET_LABELS, exist_ok=True)

    # Decode the (already work-sized) image the frontend showed.
    header, b64 = img_data.split(",", 1)
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Could not decode image"}), 400

    stem = f"corrected_{int(time.time())}"
    cv2.imwrite(os.path.join(DATASET_IMAGES, stem + ".jpg"), img,
                [cv2.IMWRITE_JPEG_QUALITY, 92])

    box = DEFAULT_BOX_FRAC * max(width, height)
    bw = box / width
    bh = box / height
    lines = []
    for p in points:
        xc = min(max(float(p["x"]) / width, 0.0), 1.0)
        yc = min(max(float(p["y"]) / height, 0.0), 1.0)
        cls = int(p.get("cls", 0))
        lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    with open(os.path.join(DATASET_LABELS, stem + ".txt"), "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))

    return jsonify({"ok": True, "saved": len(lines), "name": stem})


@app.route("/status")
def status():
    return jsonify({"model_available": model_detect.is_available()})


if __name__ == "__main__":
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=5000)
