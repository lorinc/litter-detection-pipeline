# Open questions and untested hypotheses

Model development stops at the current two detectors: YOLO11s trained on recipe `04-cv-hard-negatives` and YOLO11n
on recipe `06-cv-yolo11n`, both the cross-validation fold 0 models in the web demo. This page lists what the work
leaves open. Each item is either a hypothesis the data neither shows nor rules out, or a test that has not been run.
None of them is resolved, and none is dismissed by being listed here.

Scope of every result in this repository: one capture session, one phone, one area of Valencia, Spain, daylight
without rain; validation and test photos held out by time, not by location; one training seed per setting.

## Levers that could improve the detector, untested

- **Mined hard negatives.** Hypothesis: butt-free tiles showing lookalikes (leaves, paper, stones) teach the model
  what is not a butt better than plain pavement does. Current evidence does not decide it: hard-negative tiles cut
  false positives by 0.59 per photo at threshold 0.25, a real cut, but random butt-free tiles in the same share cut
  0.39, the two differ by less than noise, and at matched recall no recipe differs from another by more than noise
  ([cross-validation](cross-validation.md)). Test: tiles centred on each fold model's own confident false positives
  on its training photos, scored by cross-validation against `04-cv-hard-negatives`.
- **More and more varied data.** Of 42 misses in the error analysis, 16 are crushed or flattened butts, 8 blurred,
  7 small, and 15 lie on one surface that appears in only 3 training photos.
  Hypothesis: a second session at new locations, aimed at these cases and at lookalikes, is the largest lever.
  The [capture protocol](capture-protocol.md) describes that session; it has not been run.
- **Context at tile edges.** A person rules out many lookalikes from their surroundings (one of many identical
  pavement marks, part of a leaf). The model sees 640 px tiles. In the first-pass error analysis 43 of 139 false
  positives are such pattern cases and 89 are local lookalikes; 42% of false-positive boxes are cut by a tile edge
  against 24% of boxes on butts. Hypothesis: context matters mainly where a tile cuts an object, so tile overlap or
  a shifted grid helps more than wider context everywhere. The cause classification is a first pass by eye and is
  unchecked.
- **Export variations,** each to be kept only if cross-validation shows a real gain: a shifted training grid, scale
  jitter from the measured size spread.
- **Larger and different models:** YOLO11m, then a second model family, through the same evaluation harness.
- **Pretraining on public litter data:** a public dataset with cigarette butts, then fine-tuning on this set. No
  suitable dataset has been searched for yet.
- **Masking partly visible butts** (tiles that cut a butt now show it as background). Expected to be neutral: 5–7%
  of training tiles hold a butt 25–80% inside, the trained models fire on 66–77% of such fragments anyway, and a whole
  butt's score does not depend on where it sits in the tile. Not tested by
  training.

## What the current numbers cannot tell

- **How much retraining alone moves the scores.** Every setting has one seed; the bootstrap intervals hold the
  trained models fixed. A second seed of `03-cv-baseline` would measure the training-noise floor.
- **Performance at new locations.** Validation and test photos come from the same walk as the training photos,
  which flatters the trained models against the zero-shot OWLv2 baseline. A session at new locations is the real
  test.
- **The trained models' size floor.** OWLv2 keeps recall at or above 90% down to about 20–30 px butt length. The same test has not been repeated on the trained models,
  and the camera resolution and compute a robot needs depend on that number.
- **The right operating point.** Trained models are scored at 0.25, Ultralytics' default threshold. No target for
  recall, false positives per photo or speed has been set, so no threshold is preferred over another.
- **Label quality.** A blind re-annotation of 10 training photos by the same annotator three weeks later finds 1 of
  46 butts missing from the original labels (2.2%, 95% interval 0.4–11.3%); this measures one person's
  self-consistency, not true completeness. The `maybe` class is unstable: of 23 original `maybe` boxes, 4 come back
  as `maybe`, 4 as `butt`, 15 unmarked. Small, unsaturated butts are
  probably under-labelled; this is not measured.
- **Robot hardware speed.** The robot-dog frame rates on the web demo (about 15 frames per second for YOLO11n and 5
  for YOLO11s on a Jetson Orin NX at 1080p) are estimates from an assumed throughput, not a benchmark or a
  measurement. No GPU board has been tested; OWLv2 and YOLO11n have not been timed on any GPU.
- **Transfer from phone photos to a robot camera:** sensor, motion blur and mounting are unexamined.

## Robot design questions raised, not answered

- **Two-stage perception:** a far detector tuned for recall plus a close-range check, which would split the
  hardware calculation into a size floor for wide views and a low-pixel check for close-ups.
- **A ground-plane size filter:** rejecting boxes that are the wrong size for their image row; whether to use it,
  and at what tolerance, is open.
- **Butt orientation for pickup:** the detector gives boxes, not the pose a gripper needs.
