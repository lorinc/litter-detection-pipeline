# Labelling policy

This policy covers every image in `data/images/`, annotated with the tool (`pipeline/annotate.py`, then http://127.0.0.1:8712/). One box marks one butt.

## The rule

- **butt:** the annotator sees a butt.
- **maybe:** it is likely a butt, but not certain.
- Nothing else is labelled. One category rule is set: **cigar and cigarillo ends are `maybe`** (out of scope for now). They are tobacco litter, but look unlike cigarette butts (brown leaf wrapper, no filter), so they are kept out of both training signal and scoring rather than labelled background. The rule was applied to every image; the only case is one cigarillo end seen in three val images (`20261001_134111`, `20261001_134116`, `20261001_134120`). No other category rules (fragments, roach tips, heated-tobacco sticks, off-ground butts) were set; each case was judged by eye under the two definitions above.
- Boxes are drawn around the visible butt. Touching butts get one box each.

## How confidence behaved in practice

- Below a certain apparent size in pixels, and without clearly saturated orange and white colour, confidence drops from `butt` through `maybe` to "not a butt" (no box).
- The size at which this happens was not measured.

## What this means for the labels

- **Small, unsaturated butts are probably under-labelled.** A small white filter without an orange tip, or a dirty or faded butt, is more likely left unboxed than a large or orange one. So the labels have a floor set by human perception, and it is lower for orange butts than for white ones.
- **Evaluation can't measure a detector below that floor on native-size butts alone.** A detector box on a tiny unlabelled object may be a real butt the annotator couldn't confirm. The downscale test gets around this: it shrinks butts that were labelled at a confident size, so the small-size bins contain butts whose identity is known.
- **`maybe` boxes count as neither a find nor a miss** in evaluation.
- **Completeness:** the annotator estimates about 99.5% of butts are flagged, for butts above the perception floor. This is an estimate, not a measurement.

## Workflow facts

- Val and test images were annotated blind, with no proposals, before any train image.
- Train images were annotated with proposals. Proposals not accepted were stored as rejected; proposals whose centre lies inside an annotator box are stored as covered, not rejected.
- Status `done` means the image was fully scanned, including when it holds zero butts.
