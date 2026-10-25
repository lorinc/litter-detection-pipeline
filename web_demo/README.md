# Web demo

A static page that runs a trained detector (YOLO11s or YOLO11n) in the browser with ONNX Runtime Web: on the GPU through
WebGPU where the browser offers it, otherwise on the CPU through WebAssembly. Photos are processed locally and never
uploaded. Live at https://lorinc.github.io/litter-detection-pipeline/web_demo/ (GitHub Pages, served from this
repository's main branch).

## Models

A selector on the page switches between three models; switching runs detection again on the current photo. All three
train on the same 87 train photos, so the four example photos (val photos) are unseen by all of them. Visitors see
plain names (YOLO11s first, YOLO11s adjusted, YOLO11n fast) in that order; run names appear only in the code (model
keys, `?model=`) and here. Under the selector the page states the selected model's size, download and training in a
sentence. OWLv2 is named as an off-the-shelf, generic vision model, because the adjusted training's hard-negative
tiles come from its rejected proposals.

- **YOLO11s adjusted** (key `04-cv-hard-negatives-f0`): `model/yolo11s-04-cv-hard-negatives-f0.onnx`, the fold 0 model of run
  `04-cv-hard-negatives` in `docs/cross-validation.md`, final-epoch checkpoint (`last.pt`). The recipe is run 02's plus
  training tiles centred on things that look like butts but are not.
- **YOLO11n fast**, the default (key `06-cv-yolo11n-f0`): `model/yolo11n-06-cv-yolo11n-f0.onnx`, the fold 0 model of run `06-cv-yolo11n` in
  `docs/cross-validation.md`, `last.pt`: recipe 04 on the nano model, 2.6 M parameters against YOLO11s's 9.4 M.
- **YOLO11s first** (key `02-maybe-masked`): `model/yolo11s-02-maybe-masked.onnx`, run `02-maybe-masked` of `docs/training-runs.md`,
  checkpoint `best.pt`.

Measured accuracy (threshold 0.25 unless stated):

| | Recall | False positives per photo | Source |
|---|---|---|---|
| YOLO11s adjusted (04 fold 0), 19 val photos | 82.6% (38 of 46) | 0.84 | `docs/cross-validation.md` |
| YOLO11n fast (06 fold 0), 19 val photos | 78.3% (36 of 46) | 0.74 | `docs/cross-validation.md` |
| YOLO11s first (02), 19 val photos | 89.1% (41 of 46) | 2.37 | `docs/cross-validation.md` |
| Recipe 04 (hard negatives), pooled over 106 photos (5 folds) | 84.0% | 1.26 | `docs/cross-validation.md` |
| Recipe 06 (04 on YOLO11n), pooled over 106 photos (5 folds) | 81.4% | 1.01 | `docs/cross-validation.md` |
| Recipe 02, as run `03-cv-baseline`, pooled over 106 photos (5 folds) | 86.3% | 1.85 | `docs/cross-validation.md` |

- Pooled scores use the reviewed labels. Recipe 04's false-positive cut against 02's recipe (0.59 per photo) is real
  by the paired bootstrap; the recall difference (−2.3 points) is inside noise. Recipe 06 against 04: 0.25 fewer false
  positives per photo (real), −2.6 points recall (inside noise).
- At matched recall (80%, 85%, 88%) every paired difference is inside noise, so most of the gains at threshold 0.25
  come from lower scores overall. YOLO11n reaches at most 89.0% recall at the lowest threshold, against 92.4% for 04.
- Run `05-cv-more-empty-tiles` (recipe 02 with as many random butt-free tiles as recipe 04 has) cuts false positives
  per photo to 1.46 (real against 1.85); recipe 04's further 0.20 is inside noise. Much of the hard-negative gain is
  matched by more butt-free tiles of any kind.
- 02's `best.pt` was selected by its score on the same val photos, which flatters it there.

**Score bands, no threshold:** the page shows every detection down to the raw floor (0.01), labelled with its score
in percent and coloured by band: green at 78% and up, orange at 65–78%, red below 65%. This README gives each
model's recall and false positives per photo on the 19 val photos counting from each band up (from the harness's
curve in each run's `_val.json`): green only 74% / 0.05 (adjusted), 72% / 0 (fast), 67% / 0.26 (first); green and
orange 78% / 0.16, 74% / 0.11, 80% / 0.68; all boxes 89% / 9.6, 87% / 7.4, 100% / 21. The models' scores are not on
the same scale.

## Files

- `index.html`, `app.js`: the page. `app.js` ports the pipeline's scoring exactly (`pipeline/detectors.py` YOLO
  adapter and `pipeline/tiling.py` with `fit=True`): 640 px tiles at 384 px spacing with edge tiles shifted inward,
  per-tile NMS at IoU 0.7, boxes clipped to the tile (as Ultralytics does), mapped back to the photo and merged with
  NMS at IoU 0.3.
- `model/`: all three models exported with Ultralytics 8.4.184
  (`YOLO(weights).export(format='onnx', imgsz=640, simplify=True, dynamic=False, opset=17)`), input
  `images` [1, 3, 640, 640] (RGB / 255), output `output0` [1, 5, 8400] (cx, cy, w, h, score). The YOLO11s models are
  38 MB each, the YOLO11n model 11 MB.
- `examples/`: twelve validation photos (not used in training), JPEG quality 92, with 480×216 thumbnails at quality 85, shown
  in a 6-column grid (3 columns under 700 px). CC BY 4.0, like the dataset.

## Check against the harness

A numpy reference of the page's algorithm, run with ONNX Runtime on the exported models:

- **Reference against the harness:** on the original PNGs of all four example photos, the reference reproduces the
  harness's cached detections (every box down to raw score 0.01) for all three models, to the harness's rounding (scores to
  four decimals, coordinates to 0.1 px).
- **Page against the reference:** in headless Chrome on the CPU path, on the JPEG of `20261001_135420`, the page's
  detections equal the reference's box for box, for all three models (21 and 26 boxes for the YOLO11s models, 15 for
  YOLO11n; scores and coordinates within 0.0001).
- **JPEG against PNG:** the JPEG copies move scores slightly. For the hard-negative model at threshold 0.25, 18 of the
  harness's 19 boxes on the four examples are the same on the JPEGs; on `20261001_134246` one box drops below 0.25 and
  another rises above it.

## Speed

On the single-threaded CPU path on an Intel i5-1145G7, a 50-tile photo takes 39–59 s with either YOLO11s model and
14 s with YOLO11n (`20261001_135420`: 39.3 s for 04, 13.7 s for 06, same browser session). The WebGPU path has not
been timed on real GPU hardware.

## Robot dog estimate

Each model's line on the page gives an estimated frame rate on a robot dog (Unitree Go2 class, Jetson Orin NX, 1920×1080
camera): about 15 fps for YOLO11n, about 5 fps for YOLO11s. Not measured; an inference from these assumptions:

- A 1920×1080 frame takes 15 tiles (5 × 3) with this page's tiling (640 px, 384 px spacing).
- Compute per tile from the harness (`<run>_val.json`, `compute`): 6.4 GFLOPs (YOLO11n), 21.4 GFLOPs (YOLO11s).
- Effective throughput on the Orin NX with TensorRT FP16: 1–3 TFLOPS, assumed without a source (the board's 100 TOPS
  is an INT8 peak). This gives 10–30 fps (YOLO11n) and 3–9 fps (YOLO11s); the page states the rounded middle.
- Model time only: tiling, merging and camera handling are not counted.
- Accuracy at 1920 px width is untested: the models trained on 4000 px-wide photos, so at a similar field of view a
  butt covers about half the pixels, and many fall below the 30 px the training data starts at.

## Run locally

From this folder, `python3 -m http.server 8765`, then open http://127.0.0.1:8765/. `?cpu=1` skips WebGPU;
`?example=STEM` runs one example at load; `?model=06-cv-yolo11n-f0` or `?model=02-maybe-masked` starts with that model.
