# Capture protocol for the next photo session

How to shoot a dataset that the pipeline in `pipeline/` can use with less guesswork. Each rule names the weakness of the 2026-10-01 set that it removes and the config setting it affects. A new session gets its own config file (`pipeline/config.json` copied, with `source_glob` and `data_dir` changed) and runs through the same scripts.

## Rules

- **Fix the camera geometry and write it down.** Same leg, upright, phone clamped at one known angle for searching and one for inspecting. Note the leg length and both angles at the start of the session.
  - *Fixes:* the camera pitch is currently fitted from the butts. With known pitch, the perspective size filter and the pixel-size statistics apply directly to a robot camera at the same height.
- **Let orientation carry the shot type.** Landscape = searching (looking ahead over an area), portrait = inspecting (close look at one butt). Orientation is stored in each photo, so shot type is filled automatically and needs no manual labelling.
  - *Fixes:* the shot types in `meta.csv` are pre-filled by eye and unverified.
- **Separate spots in time.** Pause at least 60 s between spots, and keep shots at one spot less than 30 s apart.
  - *Fixes:* the 2026-10-01 walk has no gap longer than 49 s, so the split can only require 14 s between photos in different splits (`split.min_gap_s`). With this rhythm, `min_gap_s` can be set to 30–60 s, so no two splits share a scene.
- **At each spot, take the search shot first,** before walking up to any butt, then the inspection shots.
  - *Fixes:* search shots taken after approaching a butt are biased toward butts that were already spotted.
- **Shoot the common error sources on purpose** (next section).
  - *Fixes:* negatives and hard negatives currently come only from what happened to be in frame, so the model has seen few examples of what it confuses with butts.
- **Spread light conditions deliberately.** Roughly half the spots in sun, half in shade, plus some sun/shade edges.
  - *Fixes:* the sunny photos of 2026-10-01 sit in one 4-minute stretch (13:44–13:48), which limits how the split can place them (`split.require`).
- **Several sessions, different neighbourhoods and days.** Hold a whole session out as the test set.
  - *Fixes:* all current val and test photos come from the same 22-minute walk. A held-out session is the first truly independent test.
- **Keep location information.** Turn on GPS, or note the street at each spot. Preprocessing strips GPS from the copies and keeps it only in `meta.csv`.
  - *Fixes:* no GPS in the current set, so spots can only be separated by time.
- **Keep people and number plates out of frame where you can.** Anything that slips in is blurred by hand (`privacy_boxes.json`).
- **Same phone, full resolution, no digital zoom, hold still** (motion blur hides small butts). If a candidate robot camera exists by then, shoot with it instead.

## Common error sources

Shoot these deliberately, with the same camera geometry as the search shots (leg upright, recorded angle, landscape), spread over sun and shade. Each needs two kinds of frame:

- **Without butts:** the error source alone. These frames have no boxes and are labelled `done` with zero butts; they become "not a butt" training tiles.
- **With a butt next to it:** a butt beside or touching the error source, so the model learns the difference in one frame.

Write the error source into the street notes for each frame (e.g. `gum`, `crack`), so false positives can be counted per source later.

**False alarms seen on the trained model** (on the demo's validation photos):

- **Wall cracks and pavement cracks:** dark thin lines, crack junctions, crumbled mortar at the foot of walls.
- **Gum spots:** flattened grey, white and black gum discs, fresh and old.
- **Leaves:** single dry leaves, curled and flat, leaf fragments, piles in gutters.

**Likely false alarms, not yet confirmed on this model:**

- Bottle caps and can tabs.
- Paper scraps, receipts, tissue, cigarette-pack foil and cellophane.
- Matches, twigs, seed pods, toothpicks.
- Pebbles, light stones and chipped tile.
- Stains, paint marks, bird droppings.
- Filter-coloured litter: orange and white plastic bits, straw wrappers.

**Misses** (butts the model or the labels tend to lose):

- **White filters without an orange tip,** and aged, flattened, dirty or wet-then-dried butts: under-represented, and below the labeller's confidence when small.
- **Close-ups and large butts:** the trained models found 5–6 of 8 close-up butts and missed 2 of the 11 largest val butts; shoot inspection shots of single butts at several distances.
- **Partly hidden butts:** in cracks, between tiles, in drain grates and gutters, under leaves.
- **Butts on look-alike surfaces:** beige or white tiles, light brick, sun–shade edges.

## Known gaps

- The 60 s pause, the 30 s within-spot limit and the half-sun, half-shade target are working values, not measured requirements.
- The error-source lists come from one session: cracks, gum and leaves are seen on the trained model; the rest are candidates.
- The protocol has not been tried in the field.
