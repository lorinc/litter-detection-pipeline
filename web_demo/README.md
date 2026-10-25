# Web demo

A static page that runs a trained detector (YOLO11s) in the browser with ONNX Runtime Web: on the GPU through
WebGPU where the browser offers it, otherwise on the CPU through WebAssembly. Photos are processed locally and never
uploaded. Live at https://lorinc.github.io/litter-detection-pipeline/web_demo/ (GitHub Pages, served from this
repository's main branch).

## Models

A selector on the page switches between two models; switching runs detection again on the current photo. Both are
YOLO11s trained on the same 87 train photos, so the four example photos (val photos) are unseen by both.

- **04 Hard negatives (fold 0)**, the default: `model/yolo11s-04-cv-hard-negatives-f0.onnx`, the fold 0 model of run
  `04-cv-hard-negatives` in `docs/cross-validation.md`, final-epoch checkpoint (`last.pt`). The recipe is run 02's plus
  training tiles centred on things that look like butts but are not.
- **02 Maybe masked**: `model/yolo11s-02-maybe-masked.onnx`, run `02-maybe-masked` of `docs/training-runs.md`,
  checkpoint `best.pt`.

The number in a model's name is its run's place in the sequence of training runs.

Measured accuracy (threshold 0.25 unless stated):

| | Recall | False positives per photo | Source |
|---|---|---|---|
| 04 Hard negatives (fold 0), 19 val photos | 82.6% (38 of 46) | 0.84 | `docs/cross-validation.md` |
| 02 Maybe masked, 19 val photos | 89.1% (41 of 46) | 2.37 | `docs/cross-validation.md` |
| Recipe 04 (hard negatives), pooled over 106 photos (5 folds) | 83.8% | 1.31 | `docs/cross-validation.md` |
| Recipe 02, as run `03-cv-baseline`, pooled over 106 photos (5 folds) | 86.1% | 1.91 | `docs/cross-validation.md` |

- The pooled false-positive cut (0.60 per photo) is real by the paired bootstrap; the pooled recall difference
  (−2.3 points) is inside noise.
- At equal recall the two recipes are about equal between 75% and 86% recall, so part of the gain at threshold 0.25
  comes from the hard-negative model scoring lower overall. At 88% recall the false positives per photo are 2.3
  (hard negatives) against 3.2; at 90%, 2.6 against 4.7. These are read from the pooled curves and have no interval.
- Run `05-cv-more-empty-tiles` (recipe 02 with as many random butt-free tiles as recipe 04 has) cuts false positives
  per photo to 1.51 (real against 1.91); recipe 04's further 0.20 is inside noise. Much of the hard-negative gain is
  matched by more butt-free tiles of any kind.
- 02's `best.pt` was selected by its score on the same val photos, which flatters it there.

**Threshold:** each model's default is 0.25, the threshold its accuracy is stated at. The slider keeps the visitor's
setting when the model is switched. The two models' scores are not on the same scale, and the info panel says so.

## Files

- `index.html`, `app.js`: the page. `app.js` ports the pipeline's scoring exactly (`pipeline/detectors.py` YOLO
  adapter and `pipeline/tiling.py` with `fit=True`): 640 px tiles at 384 px spacing with edge tiles shifted inward,
  per-tile NMS at IoU 0.7, boxes clipped to the tile (as Ultralytics does), mapped back to the photo and merged with
  NMS at IoU 0.3.
- `model/`: both models exported with Ultralytics 8.4.184
  (`YOLO(weights).export(format='onnx', imgsz=640, simplify=True, dynamic=False, opset=17)`), 38 MB each, input
  `images` [1, 3, 640, 640] (RGB / 255), output `output0` [1, 5, 8400] (cx, cy, w, h, score).
- `examples/`: four validation photos (not used in training), JPEG, with thumbnails. CC BY 4.0, like the dataset.

## Check against the harness

A numpy reference of the page's algorithm, run with ONNX Runtime on the exported models:

- **Reference against the harness:** on the original PNGs of all four example photos, the reference reproduces the
  harness's cached detections (every box down to raw score 0.01) for both models, to the harness's rounding (scores to
  four decimals, coordinates to 0.1 px).
- **Page against the reference:** in headless Chrome on the CPU path, on the JPEG of `20261001_135420`, the page's
  detections equal the reference's box for box, for both models (21 and 26 boxes; scores within 0.0001, coordinates
  within 0.05 px, the reference's rounding).
- **JPEG against PNG:** the JPEG copies move scores slightly. For the hard-negative model at threshold 0.25, 18 of the
  harness's 19 boxes on the four examples are the same on the JPEGs; on `20261001_134246` one box drops below 0.25 and
  another rises above it.

## Speed

About 0.8–1.2 s per tile (40–59 s for a 50-tile photo) on the single-threaded CPU path on an Intel i5-1145G7, the
same for both models; the WebGPU path has not been timed on real GPU hardware.

## Run locally

From this folder, `python3 -m http.server 8765`, then open http://127.0.0.1:8765/. `?cpu=1` skips WebGPU;
`?example=STEM` runs one example at load; `?model=02-maybe-masked` starts with run 02.
