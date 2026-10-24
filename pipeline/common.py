"""Shared config loading. Every script takes --config PATH (default pipeline/config.json); paths in the
config are relative to the initiative directory. Photos listed in the config's "exclude" are dropped from the
split by load_split, so no script sees them; their label and split entries stay on disk (reversible).
With cv.fold set to k, load_split takes the fold of each photo from cv.assignment ({stem: fold}; run.py copies it
into each run's config) or else <data_dir>/folds.json (make_folds.py): fold k becomes 'val', the other folds 'train',
and test stays 'test', so every script runs one cross-validation fold unchanged."""
import json,sys,pathlib
HERE=pathlib.Path(__file__).resolve().parents[1]
def load_config(p=None):
    p=p or (pathlib.Path(sys.argv[sys.argv.index('--config')+1]) if '--config' in sys.argv else HERE/'pipeline/config.json')
    cfg=json.load(open(p)); cfg['_data']=HERE/cfg['data_dir']; return cfg
def load_split(cfg):
    """{stem: 'train'|'val'|'test'} from <data_dir>/split.json, without the config's excluded photos."""
    ex=set(cfg.get('exclude',[]))
    split={s:v for s,v in json.load(open(cfg['_data']/'split.json'))['split'].items() if s not in ex}
    k=cfg.get('cv',{}).get('fold')
    if k is None: return split
    folds=cfg['cv'].get('assignment') or json.load(open(cfg['_data']/'folds.json'))['folds']
    assert set(folds)=={s for s,v in split.items() if v!='test'},'folds.json does not cover the train and val photos'
    return {s:v if v=='test' else 'val' if folds[s]==k else 'train' for s,v in split.items()}
