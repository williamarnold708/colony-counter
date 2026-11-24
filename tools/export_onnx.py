"""Convert the trained detector (best.pt) to the ONNX file the server runs.

    venv\\Scripts\\python.exe tools\\export_onnx.py [path\\to\\best.pt] [path\\to\\best.onnx]

Defaults: reads app/model/best.pt, writes app/model/best.onnx. Needs `ultralytics`,
`onnx`, `onnxslim` and `onnxruntime` in the environment that runs this script; the
server itself only needs `onnxruntime`.

Why ONNX: the server runs the model with ONNX Runtime instead of PyTorch, which keeps
its memory well inside a 512 MB host (PyTorch alone takes most of that).

The export is dynamic-shape on purpose: at run time a photo is padded only up to the
next multiple of 32, exactly as Ultralytics does with the .pt model, which reproduces
its marks. A fixed 1024x1024 export pads to a square and finds about 3% more colonies.
"""
import os
import shutil
import sys
import tempfile

from ultralytics import YOLO

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "app", "model", "best.pt")
dest = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(src)[0] + ".onnx"

# Export from a temp copy so no stray files land next to the source model.
with tempfile.TemporaryDirectory() as tmp:
    pt = os.path.join(tmp, "best.pt")
    shutil.copy(src, pt)
    out = YOLO(pt).export(format="onnx", imgsz=1024, opset=17, simplify=True,
                          dynamic=True, half=False, nms=False)
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    shutil.copy(out, dest)

print(f"{src}\n  -> {dest}  ({os.path.getsize(dest) / 1e6:.1f} MB)")
