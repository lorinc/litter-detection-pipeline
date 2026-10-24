# Web demo

A static page that runs the trained detector (YOLO11s, run 2 of `docs/training-runs.md`) in the browser with
ONNX Runtime Web: on the GPU through WebGPU where the browser offers it, otherwise on the CPU through WebAssembly.
Photos are processed locally and never uploaded.

- `index.html`, `app.js`: the page. `app.js` ports the pipeline's scoring exactly (`pipeline/detectors.py` YOLO
  adapter and `pipeline/tiling.py` with `fit=True`): 640 px tiles at 384 px spacing with edge tiles shifted inward,
  per-tile NMS at IoU 0.7, boxes mapped back to the photo and merged with NMS at IoU 0.3, default threshold 0.25.
- `model/yolo11s-run2.onnx`: the run 2 weights exported with Ultralytics 8.4.184
  (`YOLO('best.pt').export(format='onnx', imgsz=640, simplify=True, dynamic=False, opset=17)`), 38 MB, input
  `images` [1, 3, 640, 640] (RGB / 255), output `output0` [1, 5, 8400] (cx, cy, w, h, score).
- `examples/`: four validation photos (not used in training), JPEG, with thumbnails. CC BY 4.0, like the dataset.

**Check against the harness:** on validation photo `20261001_135420`, the page's detections equal, box for box and
to four decimals, a numpy reference of the same algorithm run on the same JPEG, and that reference reproduces the
harness's detections on the original PNG exactly. The JPEG copy shifts low scores slightly: at threshold 0.25, 6 of
the harness's 7 boxes are the same, one drops just below 0.25 and another rises above it.

**Speed:** about 1.2 s per tile (59 s for a 50-tile photo) on the single-threaded CPU path on an Intel i5-1145G7;
the WebGPU path has not been timed on real GPU hardware.

**Run locally:** from this folder, `python3 -m http.server 8765`, then open http://127.0.0.1:8765/.
`?cpu=1` skips WebGPU; `?example=STEM` runs one example at load.
