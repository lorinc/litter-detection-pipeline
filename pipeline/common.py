"""Shared config loading. Every script takes --config PATH (default pipeline/config.json); paths in the
config are relative to the initiative directory. Photos listed in the config's "exclude" are dropped from the
split by load_split, so no script sees them; their label and split entries stay on disk (reversible)."""
import json,sys,pathlib
HERE=pathlib.Path(__file__).resolve().parents[1]
def load_config():
    p=pathlib.Path(sys.argv[sys.argv.index('--config')+1]) if '--config' in sys.argv else HERE/'pipeline/config.json'
    cfg=json.load(open(p)); cfg['_data']=HERE/cfg['data_dir']; return cfg
def load_split(cfg):
    """{stem: 'train'|'val'|'test'} from <data_dir>/split.json, without the config's excluded photos."""
    ex=set(cfg.get('exclude',[]))
    return {s:v for s,v in json.load(open(cfg['_data']/'split.json'))['split'].items() if s not in ex}
