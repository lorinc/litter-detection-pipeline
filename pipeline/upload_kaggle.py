"""Stage the dataset and upload it to Kaggle (a new version of the existing dataset, whose visibility it keeps; a new
dataset starts private).
usage: upload_kaggle.py [--config PATH] [--notes TEXT] [--dry-run]
Credentials: KAGGLE_API_TOKEN in the environment (source the private credentials file first).
Staged under <data_dir>/<kaggle.stage_dir> (images hard-linked, not copied): images/<stem>.png, labels/<stem>.json,
split.json, meta.csv, config.json, README.md. Photos in the config's "exclude" are left out entirely.
Code is not uploaded; it ships with each training run, so code changes need no new dataset version.
"""
import json,os,shutil,sys
from common import HERE,load_config,load_split
cfg=load_config(); data=cfg['_data']; K=cfg['kaggle']
stage=data/K['stage_dir']; shutil.rmtree(stage,ignore_errors=True); (stage/'images').mkdir(parents=True); (stage/'labels').mkdir()
split=load_split(cfg)
for s in sorted(split):
    os.link(data/'images'/f'{s}.png',stage/'images'/f'{s}.png')
    shutil.copy(data/'labels'/f'{s}.json',stage/'labels'/f'{s}.json')
for f in ('split.json','meta.csv'): shutil.copy(data/f,stage/f)
cp=HERE/'pipeline/config.json' if '--config' not in sys.argv else sys.argv[sys.argv.index('--config')+1]
shutil.copy(cp,stage/'config.json')
n={v:sum(1 for x in split.values() if x==v) for v in ('train','val','test')}
open(stage/'README.md','w').write(f"""# {cfg['dataset']}

Street photos of cigarette butts, one phone session, labelled by hand. Full frames at native resolution
(upright, metadata stripped, plates and faces blurred). Tiling happens at training time.

- images/: {len(split)} PNG frames
- labels/: one JSON per frame: boxes (x, y, w, h in px; class butt or maybe), rejected_proposals (hard negatives, train only), covered_proposals
- split.json: frozen time-group split (train {n['train']} / val {n['val']} / test {n['test']} in this upload); val and test labels are blind (made without detector proposals)
- meta.csv: timestamp, size, shot type, light, surface (shot type and light labelled by eye, unverified)
- config.json: the pipeline config this upload was made with (its "exclude" list names photos left out)
""")
print('staged',len(split),'frames',n,'at',stage)
if '--dry-run' in sys.argv: sys.exit()
import kagglehub
notes=sys.argv[sys.argv.index('--notes')+1] if '--notes' in sys.argv else ''
kagglehub.dataset_upload(K['handle'],str(stage),version_notes=notes)
