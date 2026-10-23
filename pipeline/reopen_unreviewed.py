"""Set train images back to 'unseen' when they were marked done without their proposals: the label has no accepted
and no rejected proposal, but the proposal file now holds at least one box. Boxes are kept.
usage: reopen_unreviewed.py [--config PATH] [--apply]   (without --apply it only lists them)
"""
import json,sys
from common import load_config,load_split
cfg=load_config(); data=cfg['_data']
split=load_split(cfg); hits=[]
for s,v in sorted(split.items()):
    lf,pf=data/'labels'/f'{s}.json',data/'proposals'/f'{s}.json'
    if v!='train' or not lf.exists() or not pf.exists(): continue
    l=json.load(open(lf))
    if l['status']=='done' and not l.get('rejected_proposals') and not any(b.get('src')=='proposal' for b in l['boxes']) \
       and json.load(open(pf))['boxes']:
        hits.append(s)
        if '--apply' in sys.argv: l['status']='unseen'; json.dump(l,open(lf,'w'),indent=1)
print(len(hits),'train images','reopened' if '--apply' in sys.argv else 'to reopen',hits)
