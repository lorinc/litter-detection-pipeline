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
- **Shoot empty ground and lookalikes too:** clean pavement, leaves, paper scraps, gum, bottle caps, filter-coloured litter.
  - *Fixes:* negatives and hard negatives currently come only from what happened to be in frame.
- **Spread light conditions deliberately.** Roughly half the spots in sun, half in shade, plus some sun/shade edges.
  - *Fixes:* the sunny photos of 2026-10-01 sit in one 4-minute stretch (13:44–13:48), which limits how the split can place them (`split.require`).
- **Several sessions, different neighbourhoods and days.** Hold a whole session out as the test set.
  - *Fixes:* all current val and test photos come from the same 22-minute walk. A held-out session is the first truly independent test.
- **Keep location information.** Turn on GPS, or note the street at each spot. Preprocessing strips GPS from the copies and keeps it only in `meta.csv`.
  - *Fixes:* no GPS in the current set, so spots can only be separated by time.
- **Keep people and number plates out of frame where you can.** Anything that slips in is blurred by hand (`privacy_boxes.json`).
- **Same phone, full resolution, no digital zoom, hold still** (motion blur hides small butts). If a candidate robot camera exists by then, shoot with it instead.

## Known gaps

- The 60 s pause, the 30 s within-spot limit and the half-sun, half-shade target are working values, not measured requirements.
- The protocol has not been tried in the field.
