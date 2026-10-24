# litter-detection-pipeline

A pipeline for detecting small objects in high-resolution ground photos: annotate, split, export square tiles,
train an Ultralytics detector on Kaggle, and score any detector on whole frames with one evaluation harness.
Code, the annotation tool with four example photos, the labelling policy and a capture protocol; the full image set and its labels are not included (see Data).

## Layout

- `pipeline/config.json`: every setting. Each script takes `--config PATH`; paths are relative to the repository root.
- `annotation_tool/index.html`: the browser page of the annotation tool, served by `pipeline/annotate.py`.
- `annotation_tool/LABELING_POLICY.md`: the rule the annotation followed, and what it means for the labels.
- `annotation_tool/example/`: four example photos (portrait, JPEG, no metadata) with their own config; two are training photos with zero-shot proposals, two are test photos shown in blind mode.
- `docs/capture-protocol.md`: how to shoot the next photo set, and which weakness of the current one each rule removes.

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
| `run.py`, `publish.py` | Sync code, train one or more runs on Kaggle (one GPU each), download, evaluate |
| `upload_kaggle.py` | Upload the image copies and labels as a private Kaggle dataset |

## Trying the annotation tool

From `pipeline/`: `python annotate.py --config ../annotation_tool/example/config.json`, then open http://127.0.0.1:8712/. It needs only the Python standard library. Labels are saved to `annotation_tool/example/labels/`.

## Data

The complete dataset (125 street photos from one capture session, privacy-blurred and stripped of metadata, with hand-drawn labels and a frozen train / val / test split) is uploaded to Kaggle as a private dataset, which the training runs mount.

## Running

Run scripts from `pipeline/`, e.g. `python export.py --data DATA_DIR --out EXPORT_DIR`. Dependencies: Python 3.12,
torch, torchvision, transformers, ultralytics, pillow, numpy, scikit-learn, matplotlib, kaggle, kagglehub.

## Licence

AGPL-3.0 (see `LICENSE`).
