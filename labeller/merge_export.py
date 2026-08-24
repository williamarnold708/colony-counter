"""
Merges a training-data export downloaded from /export_training_data on Render
into the local labeller dataset (data/images, data/labels), then writes a
ready-to-upload data.zip in the same layout the Colab training notebook
expects.

Usage:
    python labeller/merge_export.py path/to/training_data_<timestamp>.zip
"""
import os
import sys
import shutil
import zipfile
import tempfile

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
IMAGES = os.path.join(DATA, "images")
LABELS = os.path.join(DATA, "labels")
CLASSES_FILE = os.path.join(DATA, "classes.txt")
OUTPUT_ZIP = os.path.join(BASE, "data.zip")


def merge(export_zip_path):
    if not os.path.isfile(export_zip_path):
        sys.exit(f"No such file: {export_zip_path}")

    os.makedirs(IMAGES, exist_ok=True)
    os.makedirs(LABELS, exist_ok=True)

    added = 0
    skipped = []
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(export_zip_path) as z:
            z.extractall(tmp)

        src_images = os.path.join(tmp, "images")
        src_labels = os.path.join(tmp, "labels")
        for name in sorted(os.listdir(src_images)):
            dest = os.path.join(IMAGES, name)
            if os.path.exists(dest):
                skipped.append(name)
                continue
            shutil.copy(os.path.join(src_images, name), dest)
            stem = os.path.splitext(name)[0]
            lbl_src = os.path.join(src_labels, stem + ".txt")
            if os.path.exists(lbl_src):
                shutil.copy(lbl_src, os.path.join(LABELS, stem + ".txt"))
            added += 1

    total = len([f for f in os.listdir(IMAGES) if f != ".gitkeep"])
    print(f"Added {added} new image+label pairs.")
    if skipped:
        print(f"Skipped {len(skipped)} already present (same filename): {skipped}")
    print(f"labeller/data now has {total} images total.")

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for name in os.listdir(IMAGES):
            if name != ".gitkeep":
                z.write(os.path.join(IMAGES, name), f"images/{name}")
        for name in os.listdir(LABELS):
            if name != ".gitkeep":
                z.write(os.path.join(LABELS, name), f"labels/{name}")
        if os.path.exists(CLASSES_FILE):
            z.write(CLASSES_FILE, "classes.txt")

    print(f"Wrote {OUTPUT_ZIP} — upload this to the Colab notebook.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python labeller/merge_export.py path/to/training_data_<timestamp>.zip")
    merge(sys.argv[1])
