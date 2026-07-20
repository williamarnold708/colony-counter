# Colony Counter

An open, explainable tool for counting microbial colonies on agar plates from
a photo — built for real brewery / food-microbiology QC conditions, including
the **gridded counting plates** that defeat most simple image-processing
approaches.

It combines a trained object-detection model with a **human-in-the-loop**
correction step: the model proposes colonies, you confirm or fix them with a
click. The count you leave with is always one you agreed with.

![status](https://img.shields.io/badge/status-working%20prototype-brightgreen)
![license](https://img.shields.io/badge/license-MIT-blue)

---

## Why this exists

Automated colony counting is a long-standing lab need, but off-the-shelf
methods break on real-world plates: printed counting grids get mistaken for
colonies, faint or coloured colonies are missed, and lighting varies bench to
bench. This project takes an honest two-track approach:

1. **Classical computer vision** as a transparent, no-training-needed baseline
   — including a grid-removal path (inpainting) for gridded plates.
2. **A trained detector** (YOLO) that learns what a colony *looks like* from
   labelled examples, and handles gridded, faint, and mixed plates far better.

Crucially, it never claims to replace the analyst. Automated detection is a
*starting point*; a fast manual-correction UI makes the final count reliable.

## What's in the box

| Component | What it does |
|-----------|--------------|
| **`app/`** | The counter web app. Upload a plate → auto-detect → click to correct → get a count. Uses the trained model if present, else classical CV. |
| **`labeller/`** | A labelling tool to build a training dataset from your own plates. Auto-detection pre-fills each plate so you confirm rather than click from scratch. Exports YOLO format. |
| **`training/`** | A Google Colab notebook that trains a colony detector on your labelled data, on a free GPU. |

The three form a complete loop: **label your plates → train a model → use it
in the counter → label more to improve it.**

## Quick start (counter app)

```bash
git clone https://github.com/williamarnold708/colony-counter.git
cd colony-counter
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
cd app
python main.py
```

Open <http://127.0.0.1:5000>, drop in a plate photo, and correct the
detection by clicking. Without a trained model it runs on classical CV; add
one (below) and it uses that automatically.

### Adding a trained model

Put a trained `best.pt` (from the training notebook) at
`app/model/best.pt`. Restart the app — it prints
`Detection engine: trained model` on startup and uses it. If the file is
absent it falls back to classical CV, so it always works.

## Building your own model

1. **Label** — drop plate photos into `labeller/inbox/`, run
   `python labeller/app.py`, open <http://127.0.0.1:5001>, mark every colony
   (keyboard-driven; auto-detection pre-fills). Produces a YOLO dataset in
   `labeller/data/`.
2. **Train** — zip `labeller/data/`, open `training/colony_training.ipynb` in
   Google Colab, enable a GPU, run it, download `best.pt`.
3. **Use** — drop `best.pt` into `app/model/` and restart the counter.

See the READMEs in each folder for detail.

## How well does it work?

On a first dataset of ~59 mixed brewery plates (gridded and plain, various
colony types), a YOLOv8-nano detector reached **mAP@50 ≈ 0.78**. In practice
it auto-detects dense gridded plates that classical CV cannot handle at all,
and the manual-correction step covers its remaining errors (mostly faint
colonies on sparse plates). Accuracy improves as more plates are labelled and
the model is retrained.

**This is a working prototype, not a validated instrument.** Always confirm
the count against your own judgement before relying on it, and validate
against manual counts for any critical use.

## Limitations (honest)

- **Confluent growth / lawns** are not individually countable by any method —
  report as TNTC and use dilution plating.
- **Sparse plates** are where the model is least confident; the manual step
  matters most here.
- The bundled classical grid path is tuned for warm-coloured colonies; the
  trained model generalises better.
- No result persistence / audit trail yet (see Roadmap).

## Roadmap

- Colony-type classification (e.g. distinguishing indicator-media colours)
- Result export and per-batch count history
- Production hosting for shared lab / multi-user use

## License

MIT — see [LICENSE](LICENSE). Free to use, modify, and deploy, including in
commercial lab settings.

## Citing

If you use this in work you publish, please cite it — see
[CITATION.cff](CITATION.cff).

## Acknowledgements

Developed to solve a real quality-control problem in a brewery microbiology
lab. Built iteratively, with each design decision driven by what actually
failed on real plates rather than what looked good in theory.
iteratively, with each design decision driven by what actually failed on real
plates rather than what looked good in theory.
