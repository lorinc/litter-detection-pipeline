# Cross-validation: hard-negative tiles against run 2's recipe

**Scope label:** YOLO11s trained on one phone session (Valencia, Spain, 2026-10-01); 5-fold cross-validation over the 106 train and val photos (259 butts), one seed per fold; folds 1–4 carry proposal-assisted labels.

## Question

Do hard-negative tiles (training tiles centred on things that look like butts but are not) make the trained detector better? The target is the false positives the trained model makes on wall cracks, gum spots and leaves.

## Setup

**Recipes** (full configs in `pipeline/recipes/mask.json` and `pipeline/recipes/mask_hn.json`):

| | `mask` | `mask_hn` |
|---|---|---|
| Model | YOLO11s, COCO-pretrained, Ultralytics 8.4.184 | same |
| `maybe` boxes | masked (painted flat grey) | same |
| Empty grid tiles | 10% of train tiles | same |
| Hard-negative tiles | none | one 640 px tile per rejected OWLv2 proposal of the fold's train photos (proposals already fully inside an earlier such tile are skipped), on top of the empty tiles |
| Training | Ultralytics defaults, 100 epochs, no per-epoch validation | same |

`mask` is run 2's recipe. The hard-negative tiles change the training set a lot: in fold 1, `mask_hn` trains on 710 tiles (247 without a butt) against `mask`'s 446 (45 without a butt); across folds, 33–39% of `mask_hn`'s train tiles hold no butt, against 10% for `mask` (tile counts from each run's `kaggle/train.log`). The switch therefore adds harder butt-free tiles and many more butt-free tiles at once, and a difference does not say which of the two caused it.

**Folds** (`data/folds.json`):

| Fold | Photos | Butts | What it is |
|---|---|---|---|
| 0 | 19 | 46 | the frozen val split, labelled blind (no detector suggestions) |
| 1 | 23 | 76 | train photos, cut by 14 s time groups |
| 2 | 21 | 71 | train photos |
| 3 | 24 | 38 | train photos; holds every train close-up (6) and so its model learns close-ups from fold 0's 3 only |
| 4 | 19 | 28 | train photos; holds most of the sunny photos |

- **Training and scoring:** each recipe is trained once per fold on the other four folds and scored on the fold it did not see. All 10 runs trained on Kaggle's T4 machine, two at a time, from one kernel. The shared harness (`pipeline/evaluate.py`) scores the final-epoch checkpoint (`last.pt`) on whole photos at threshold 0.25: 640 px tiles, detections mapped back to the photo and merged, a butt found when a detection's centre falls inside its box or they overlap by at least 30%, `maybe` boxes counting neither way.
- **Pooled:** every one of the 106 photos is held out exactly once, so the pooled score is all 106 photos (259 butts), each scored by the model that did not train on it (`pipeline/cv_report.py`). The pooled curve applies one threshold to all five models of a recipe.
- **Paired comparison:** a bootstrap of 1,000 resamples of the same photos, scoring both recipes on each resample. A difference counts as real only when its 95% interval excludes zero.
- **Labels:** folds 1–4 carry train labels, made with OWLv2's proposals on screen. Pooled scores therefore compare the two recipes with each other, not with OWLv2; the comparison with OWLv2 stays on fold 0.
- **Records:** `runs/cv_mask.json`, `runs/cv_mask_hn.json`, `runs/cv_mask_vs_mask_hn.json`, `runs/cv_mask_vs_mask_hn.log`, and per run `runs/<recipe>_f<k>/` (config, commits, training output, harness result).

## Results

![Pooled recall against false positives per photo for both recipes, with the operating points at threshold 0.25](fig/cv_mask_vs_mask_hn.png)

**Pooled, at threshold 0.25** (95% intervals: Wilson for recall, a bootstrap over whole photos for false positives):

| | Recall | False positives per photo | Boxes per photo | Precision |
|---|---|---|---|---|
| `mask` | 86.1% (81.4–89.8%), 223 of 259 | 1.91 (1.52–2.32) | 4.46 | 0.56 |
| `mask_hn` | 83.8% (78.8–87.8%), 217 of 259 | 1.31 (1.01–1.63) | 3.74 | 0.63 |
| `mask_hn` − `mask` (paired) | −2.3 points (−5.5 to +1.5): inside noise | −0.60 (−1.01 to −0.23): real | | |

**Per fold, at threshold 0.25** (recall as found of butts, false positives per photo):

| Fold | Photos | Butts | `mask` recall | `mask_hn` recall | `mask` FP/photo | `mask_hn` FP/photo |
|---|---|---|---|---|---|---|
| 0 (val, blind) | 19 | 46 | 82.6% (38) | 82.6% (38) | 1.58 | 0.84 |
| 1 | 23 | 76 | 78.9% (60) | 69.7% (53) | 2.22 | 1.61 |
| 2 | 21 | 71 | 85.9% (61) | 85.9% (61) | 2.95 | 2.00 |
| 3 (close-ups) | 24 | 38 | 97.4% (37) | 100% (38) | 1.71 | 1.29 |
| 4 (sun) | 19 | 28 | 96.4% (27) | 96.4% (27) | 0.95 | 0.68 |

`mask_hn` has fewer false positives in every fold. Recall is equal in three folds, one butt higher in fold 3 and 7 butts lower in fold 1.

**At matched recall** (false positives per photo at the highest threshold whose pooled recall reaches the level; read from the pooled curves, no interval):

| Recall at least | `mask` | `mask_hn` |
|---|---|---|
| 75% | 0.32 | 0.33 |
| 80% | 0.51 | 0.67 |
| 83.8% (`mask_hn` at 0.25) | 0.91 (threshold 0.48) | 1.26 (threshold 0.26) |
| 86.1% (`mask` at 0.25) | 1.73 (threshold 0.28) | 1.74 (threshold 0.17) |
| 88% | 3.18 | 2.30 |
| 90% | 4.72 | 2.64 |
| 92% | 6.99 | 8.42 |

The highest recall either recipe reaches, at the lowest threshold kept (0.01), is 93.8% (`mask`) and 92.3% (`mask_hn`).

**Recall by butt size** (box long side, native px; pooled, threshold 0.25; no butt is under 20 px):

| Size | Butts | `mask` | `mask_hn` |
|---|---|---|---|
| 20–30 px | 2 | 0 | 0 |
| 30–45 px | 30 | 23 (77%) | 20 (67%) |
| 45–60 px | 55 | 45 (82%) | 44 (80%) |
| 60–90 px | 74 | 62 (84%) | 63 (85%) |
| 90–120 px | 53 | 52 (98%) | 49 (92%) |
| 120 px and up | 45 | 41 (91%) | 41 (91%) |

**By shot type** (pooled, threshold 0.25; recall with found butts, false positives per photo):

| Shot type | Photos | Butts | `mask` | `mask_hn` |
|---|---|---|---|---|
| Wide | 92 | 228 | 86% (196), 1.88 | 83% (189), 1.34 |
| Close-up | 9 | 20 | 85% (17), 2.33 | 90% (18), 1.11 |
| High | 5 | 11 | 91% (10), 1.60 | 91% (10), 1.20 |

**Fold 0 against run 2.** Fold 0's models train on the same 87 photos as run 2, and `mask` is run 2's recipe; the val labels are identical (same label hash in all three harness results).

| | Checkpoint scored | Recall | False positives per photo | Source |
|---|---|---|---|---|
| Run 2 | `best.pt` | 89.1% (41 of 46) | 2.37 | `runs/run2/run2_val.json` |
| `mask`, fold 0 | `last.pt` | 82.6% (38 of 46) | 1.58 | `runs/mask_f0/mask_f0_val.json` |
| `mask_hn`, fold 0 | `last.pt` | 82.6% (38 of 46) | 0.84 | `runs/mask_hn_f0/mask_hn_f0_val.json` |

Run 2 and `mask` fold 0 differ in two ways besides training randomness: run 2 kept the checkpoint with the best tile-level score on these same val photos (`best.pt`), which flatters it on val, and it validated after every epoch. The 3-butt gap is inside both runs' recall intervals (69.3–90.9% and 77.0–95.3%), so it does not separate checkpoint choice from training noise.

**Training time:** 100 epochs took 11–14 min per `mask` run (0.191–0.225 h) and 17–21 min per `mask_hn` run (0.283–0.350 h), with two runs sharing the machine. The whole kernel, 10 runs in 5 rounds, ran 99 min (`runs/mask_f0/kaggle/butt-train-mask-mask-hn-cv.log`). Both recipes are the same YOLO11s, so detection cost per photo is the same (9.4 M parameters, 1,071 GFLOPs); the harness timed 4.7–6.8 s per photo on this PC's CPU across the 10 runs, a spread not caused by the recipe.

## Reading

- **At the fixed threshold 0.25, hard-negative tiles cut false positives by 0.6 per photo (about a third), and the recall change is inside noise.** The cut is real by the paired test and shows in every fold, including the blind fold 0.
- **At equal recall, the two recipes are about the same between 75% and 86% recall.** The curves nearly overlap there, and `mask` is slightly ahead at 80–84%. This is an inference read off the curves, without an interval: most of the fixed-threshold gain comes from `mask_hn` giving lower confidence scores overall, so the same threshold sits at a slightly lower recall on almost the same curve. The cut in false positives at 0.25 is real, but it does not show a better detector in this range.
- **`mask_hn` is ahead at 88–90% recall** (2.3–2.6 against 3.2–4.7 false positives per photo) and behind again at 92%. No interval exists for these differences, so they are a pattern to test, not a finding.
- **Where `mask_hn` loses recall:** fold 1 (7 butts) and small butts (30–45 px: 20 against 23 of 30). Close-ups gain one butt (fold 3, 18 against 17 of 20 pooled). Each of these rests on a handful of butts and is inside the intervals.
- **What this does not show:** whether harder or merely more butt-free tiles cause the change (the switch adds both); how either recipe does on new streets (all photos come from one 22-minute walk); how much of any difference is training randomness (one seed per fold).

## Known gaps

- One seed per fold. The paired bootstrap resamples photos with the trained models held fixed, so training-to-training variation is not in any interval.
- Matched-recall differences have no interval; they are read off the pooled curves.
- The pooled curve applies one threshold to five separately trained models whose scores need not be on the same scale.
- Folds 1–4 carry labels made with OWLv2's proposals on screen; only fold 0's 46 butts are labelled blind.
- Fold 3 holds every train close-up and fold 4 most of the sun, so per-fold differences mix recipe with condition.
- The hard-negative switch adds both harder and more butt-free tiles; the empty-tile-share pair has not been run.
- All photos come from one session; no held-out session exists.
- No butt is under 20 px and only 2 are under 30 px, so nothing here measures the trained model's size floor.
- The test split has not been scored; it is reserved for the finalist.
