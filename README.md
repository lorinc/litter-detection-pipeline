# litter-detection-pipeline

A pipeline for detecting small objects in high-resolution ground photos: annotate, split, export square tiles,
train an Ultralytics detector on Kaggle, and score any detector on whole frames with one evaluation harness.
Code only; no images or labels are included.

## Layout

- `pipeline/config.json`: every setting. Each script takes `--config PATH`; paths are relative to the repository root.
- `tool/index.html`: the browser page of the annotation tool, served by `pipeline/annotate.py`.

| Script | What it does |
|---|---|
| `preprocess.py` | Upright, metadata-free image copies, privacy blurring, `meta.csv` |
| `make_split.py` | Time-group train / val / test split |
| `propose.py`, `owl.py`, `train_filter.py` | Zero-shot OWLv2 proposals for annotation, with an optional box filter |
| `annotate.py` | Local annotation server (proposal mode for train images, blind mode for val and test) |
| `labels_util.py` | Label-record helpers |
| `downscale_detect.py`, `downscale_report.py` | Recall per object size as frames are shrunk |
| `perspective_check.py` | Ground-plane size check of detector boxes |
| `export.py` | Square tiles and YOLO labels from full frames and JSON labels |
| `train.py` | Ultralytics training on an export |
| `tiling.py`, `metrics.py`, `detectors.py`, `evaluate.py` | Evaluation harness: tiling, matching, detector adapters, full-frame scoring |
| `run.py`, `publish.py` | One run per name: sync code, train on Kaggle, download, evaluate |
| `upload_kaggle.py` | Upload the image copies and labels as a private Kaggle dataset |

## Running

Run scripts from `pipeline/`, e.g. `python export.py --data DATA_DIR --out EXPORT_DIR`. Dependencies: Python 3.12,
torch, torchvision, transformers, ultralytics, pillow, numpy, scikit-learn, matplotlib, kaggle, kagglehub.

## Licence

AGPL-3.0 (see `LICENSE`).
