# Development notes / methods

This document records *why* the tool is built the way it is — the design
decisions and, importantly, the approaches that failed. It doubles as source
material for a methods write-up.

## Problem

Counting microbial colonies on QC plates by eye is slow and subjective. The
lab uses a mix of plate types, including **printed gridded counting plates**,
under variable phone-camera lighting. The goal: a fast, honest assist that
produces a count the analyst agrees with.

## Approach 1 — classical computer vision (baseline)

Thresholding + watershed to segment and split colonies.

- **Worked** on clean plates: pale colonies on plain or coloured agar
  (e.g. blue nutrient agar) separate well by brightness/saturation.
- **Failed** on gridded plates: the printed grid thresholds identically to
  colonies. Grid-line intersections were counted as colonies, massively
  inflating counts (e.g. 189 "colonies" where ~8 existed).

### Grid handling attempts

1. **Grid subtraction** — detect grid lines by morphology and subtract.
   Failed: colonies *sitting on* lines were deleted with the grid.
2. **Colour separation (LAB b-channel)** — worked for warm-coloured (tan)
   colonies, which stand out from a neutral grid. Failed for dark-green /
   low-saturation colonies whose signal was too weak.
3. **Grid inpainting** (the fix that stuck) — detect grid lines, then
   *inpaint* them (reconstruct pixels from surrounding agar) so colonies
   crossing a line survive. Combined with b-channel detection, this gave
   correct counts on sparse gridded plates with tan colonies.

**Conclusion:** classical CV can be pushed to handle *some* gridded plates,
but every fix was plate-type-specific and brittle. This motivated a learned
approach.

## Approach 2 — trained detector (the answer)

A YOLO object detector learns "colony vs grid vs bubble vs rim" directly from
labelled examples, rather than relying on hand-tuned colour/threshold rules.

- Built a **labelling tool** (reusing the classical detector to pre-fill
  marks, so labelling is confirm-not-click-from-scratch).
- Labelled ~59 mixed plates, single class (`colony`).
- Fine-tuned YOLO-nano on a free Colab GPU: **mAP@50 ≈ 0.78**, recall ≈ 0.82.
- Auto-detects dense gridded plates that classical CV cannot handle at all.

### Known behaviour

- High confidence and accuracy on dense/clear plates.
- Lower confidence on sparse plates — real and false detections overlap in
  score, so no single confidence threshold cleanly separates them. This is the
  signature of a small dataset; more labelled sparse/empty plates is the fix.
- The human-in-the-loop correction step covers the remaining gap in practice.

## Design principle: human-in-the-loop

At no point does the tool claim an autonomous count. The model proposes;
the analyst disposes. This is both more honest and more useful than a
black-box number — the analyst stays accountable for the result, and their
corrections are the natural source of more training data.

## Reproducibility

The full loop is in this repo: label → train → deploy → relabel. Anyone can
build a detector tuned to their own plates and lighting from their own photos.

## The data flywheel (correction feedback)

The counter can save a corrected plate — image plus the analyst's final marks
(auto-detected, minus removed, plus hand-added) — directly into the labeller's
dataset in YOLO format ("Save this plate for training"). This does **not**
change the running model; it adds a training example.

The effect over time: every plate the analyst corrects becomes a labelled
example for the next retrain. The two loops connect —

- **Using** the tool (correcting counts) generates training data as a
  by-product.
- **Retraining** periodically folds those corrections back into the model.

So the model improves precisely on the cases the analyst had to correct — a
targeted, low-effort path to better accuracy that requires no separate
labelling sessions.
