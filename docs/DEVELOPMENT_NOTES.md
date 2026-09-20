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
- Fine-tuned YOLO11-nano (`yolo11n.pt`, imgsz 1024) on a free Colab GPU: **mAP@50 ≈ 0.78**, recall ≈ 0.82.
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

## Touching colonies (classical path)

Watershed alone leaves closely-touching colonies as one blob, which then
fails the per-colony size/roundness checks and is dropped entirely. The
fix that stuck: before those checks, look for **convexity defects** — the
sharp inward notches at the "waist" where two round outlines meet — and
cut the blob along the line between each pair of nearby notches, recursing
on the halves. A single colony's outline has no deep notches, so it is
never split; the pass can only recover colonies, not lose them.

On the 40 labelled plates checked, this reduced the classical detector's
summed count error from 895 to 854, with one dense plate going from 63 to
100 detected (138 true). Still far behind the trained model, which remains
the primary engine.

## Auto-sizing hand-placed marks

A mark the analyst adds by hand needs a box size for training. The first
version reused the whole-plate threshold rules on a small crop around the
tap. That measured only **32%** of labelled colonies: those rules decide
"pale-on-dark vs dark-on-pale" from plate-wide statistics a small crop
doesn't have.

What worked: segment the crop by **colour distance from the agar at the
crop's edge** (median Lab colour of the border), which is polarity-free,
then watershed and convexity-split as above and keep the piece containing
the tap. Hit rate rose to **66%**, with ~8% false fits on empty agar.
Looser roundness thresholds bought a few more hits but roughly doubled the
false fits; a confidently wrong size is worse than a flagged default, so
0.45 circularity stayed. The old whole-plate rules remain as a fallback.

The remaining misses on gridded plates were mostly colonies sitting on a
printed line, which merges colony and line into one long blob. Inpainting
any long thin dark lines found in the crop first (the same trick as the
whole-plate gridded path, skipped when no line is present) took the hit
rate to **68%** at the same false-fit rate. Colonies at a line
intersection still often fail and stay flagged for a manual size.

Marks that couldn't be measured are drawn dashed so the analyst can size
them by hand, and the training export records per-box whether the size was
measured or defaulted (`<name>.sizesrc.json`) so they can be audited later.

## Inference resolution

The model was trained at `imgsz=1024`, but the app ran inference at 640 -
a deliberate trade for speed on Render's free-tier CPU. At 640 a colony on
a dense plate is only a few pixels wide and most go undetected. On the
labelled set (108 plates, 3071 colonies, standard sensitivity):

| imgsz | avg CPU time/plate | dense recall | ALL sum\|err\| |
|---|---|---|---|
| 640  | 0.12s | 0.858 | 479 |
| 1024 | 0.22s | 0.931 | 399 |
| 1280 | 0.34s | 0.919 | 425 |
| 1600 | 0.57s | 0.925 | 469 |

1024 (matching the training resolution) is both the most accurate and,
past that point, more resolution stops helping - precision on sparse
plates actually drops as noise gets promoted to detections. **1024 is now
the default for every plate**, at roughly double the CPU cost of 640. A
two-tier mode exists for a slower host: set `INFERENCE_IMGSZ=640` and
`DENSE_IMGSZ=1024`, and a plate whose fast first pass finds at least
`DENSE_TRIGGER` (12) colonies gets a second pass at 1024. This recovers
most, not all, of the always-1024 gain, since some dense plates undercount
enough at 640 to never trigger the second pass.

Separately, Ultralytics caps detections at 300 per image by default. That
silently truncated dense plates (one benchmark plate labelled 430 came
back as exactly 300). The cap is now raised to 3000.

## Too numerous to count

Above 250 colonies a plate is outside the range standard practice
considers reliably countable by any method (the usual guidance is 25-250
per plate; more than that and the recommendation is to plate the next
dilution). The app now shows a too-numerous-to-count advisory above that
count rather than presenting a confident number. It's advisory only - the
count and full correction UI still work above the line, since a lab may
still want the estimate.

## Hybrid: model detections + geometric splitting

MCount (Kim et al., PLOS ONE 2024) resolves merged colonies by cutting a
blob's outline at its concave corners and fitting circles to the pieces,
reporting ~4% error on fluorescent E. coli arrays. It cannot tell a grid
line from a colony, so on its own it would fail here the way classical CV
did. The combination tried: YOLO decides where colonies are, and the
geometric splitter runs only in the neighbourhood of each detection.

**What the data showed first.** The obvious trigger, "split any box that
is much larger than the plate's typical box", never fires: across 108
plates only 6 boxes were oversized. YOLO's non-max suppression makes a
touching pair come out as one *normal-sized* box with the neighbour simply
gone. Of 294 colonies the model missed, 163 sat inside or right beside a
normal box. So the stage is blob-driven instead: segment around each box,
keep only the blob fused to the anchored colony, cut it, and add pieces no
existing detection covers.

**What it took to keep precision.** Naively this added 744 colonies, of
which most were wrong (dense-plate precision 0.97 → 0.78). Three guards
brought it back: the piece must match the anchored colony's colour
(rejects shadow halos, grid junctions, rim fragments), the anchored blob
must be bigger than one colony, and the piece's centre must sit at least
0.85 × (r_anchor + r_piece) away (a fragment *of* the colony is closer
than a colony *beside* it). Grid inpainting also had to look for lines at
several angles, since plates are rarely photographed square to the grid.

**Result on the labelled set at imgsz 640 (108 plates, 3071 colonies,
slider at standard):**

| | Dense plates (≥50) precision / recall | All plates count error |
|---|---|---|
| Model only | 0.969 / 0.858 | 488 |
| Hybrid (default settings) | 0.938 / 0.866 | 471 |

At 640 it was a modest net gain: it recovered roughly one in five of its
additions as a labelled colony and the rest were false, and the count
error still fell because the model's undercount on dense plates was
larger than the noise added.

**This result does not hold once inference moved to 1024.** Re-run at the
new default resolution:

| | Dense plates (≥50) precision / recall | All plates count error |
|---|---|---|
| Model only @1024 | 0.901 / 0.931 | 399 |
| Hybrid @1024 (default settings) | 0.888 / 0.932 | 433 |
| Hybrid @1024 (stricter thresholds) | 0.895 / 0.931 | 416 |

Recall is identical to three decimals in every hybrid variant tried: none
of its additions matched a label that the higher-resolution pass hadn't
already found. At 640 the geometric splitter was recovering real touching
colonies the model missed outright; at 1024 the model already finds
nearly all of them itself, so the splitter's remaining triggers are
almost entirely noise (halos, grid-line remnants, rim fragments), and it
makes the count worse. **The hybrid stage is now off by default** and
left as an opt-in checkbox on the review screen (or `HYBRID_SPLIT=1`
globally) rather than removed outright, since a specific plate or a lower
inference resolution can still make it worthwhile - toggle it on and off
on the same plate to check.

**Caveat on the measurement.** The dataset contains the same plate saved
twice from the counter with labels that differ by 61 colonies (316 vs
255), so label noise on dense plates is of the same order as the effects
being measured here. A cleaner dense-plate validation set is needed
before tuning either the resolution or the hybrid stage further; more
labelled dense plates in training is the higher-leverage fix for the
remaining undercount.
