"""
Draws each label file's marked colonies onto its image, so you can visually
spot-check a dataset before training — especially useful when someone else
has been submitting corrections and you want to confirm they only tagged
actual colonies.

Usage:
    python labeller/preview_labels.py                 # preview everything in data/
    python labeller/preview_labels.py corrected_123.jpg  # preview just one
"""
import os
import sys

import cv2

BASE = os.path.dirname(os.path.abspath(__file__))
IMAGES = os.path.join(BASE, "data", "images")
LABELS = os.path.join(BASE, "data", "labels")
REVIEW = os.path.join(BASE, "review")

MARKER_COLOR = (0, 255, 0)   # green, BGR
MARKER_THICKNESS = 2


def draw_labels(image_path, label_path, out_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"  could not read image: {image_path}")
        return False
    h, w = img.shape[:2]

    if os.path.exists(label_path):
        with open(label_path) as f:
            lines = [l.strip() for l in f if l.strip()]
    else:
        lines = []

    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        _, xc, yc, bw, bh = parts[:5]
        cx, cy = float(xc) * w, float(yc) * h
        r = max(float(bw) * w, float(bh) * h) / 2
        cv2.circle(img, (int(cx), int(cy)), max(int(r), 6),
                   MARKER_COLOR, MARKER_THICKNESS)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)
    print(f"  {len(lines)} marks -> {out_path}")
    return True


def main():
    os.makedirs(REVIEW, exist_ok=True)
    target = sys.argv[1] if len(sys.argv) > 1 else None

    if target:
        names = [target]
    else:
        names = sorted(f for f in os.listdir(IMAGES) if f != ".gitkeep")

    if not names:
        print("No images found in labeller/data/images.")
        return

    for name in names:
        stem = os.path.splitext(name)[0]
        image_path = os.path.join(IMAGES, name)
        label_path = os.path.join(LABELS, stem + ".txt")
        out_path = os.path.join(REVIEW, name)
        if not os.path.exists(image_path):
            print(f"No such image: {image_path}")
            continue
        print(f"{name}:")
        draw_labels(image_path, label_path, out_path)

    print(f"\nDone. Open labeller/review/ to look through them.")


if __name__ == "__main__":
    main()
