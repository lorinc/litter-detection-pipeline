"""Cut the train and val photos into cross-validation folds and freeze <data_dir>/folds.json.
usage: make_folds.py [--config PATH]           print the folds the config produces (no file written)
       make_folds.py [--config PATH] --write   write folds.json once; refuses if it exists (never reshuffle)
Fold 0 is the frozen split's val (blind labels). The train photos are cut into time groups exactly as the split is
(split.min_gap_s, over the whole session), and the groups are shuffled (cv.seed) into folds 1..cv.folds-1, each
group to the currently smallest fold; the first shuffle whose folds are within cv.tolerance (share of the train and
val photos) of an equal size and each hold cv.require photos per value (empty: no minimum) wins. Test photos are in no fold.
With cv.fold set to k, load_split (common.py) makes fold k 'val' and the other folds 'train'.
"""
import csv,json,sys,random,datetime,collections
from common import load_config,load_split
cfg=load_config(); C=cfg['cv']; data=cfg['_data']
split=load_split(dict(cfg,cv={}))
rows=sorted(csv.DictReader(open(data/'meta.csv')),key=lambda r:r['timestamp']); meta={r['image']:r for r in rows}
ts=[datetime.datetime.strptime(r['timestamp'],'%Y:%m:%d %H:%M:%S') for r in rows]
groups=[[rows[0]['image']]]
for i in range(1,len(rows)):
    if (ts[i]-ts[i-1]).total_seconds()>=cfg['split']['min_gap_s']: groups.append([])
    groups[-1].append(rows[i]['image'])
tg=[[s for s in g if split.get(s)=='train'] for g in groups]; tg=[g for g in tg if g]
assert all(len({split.get(s) for s in g if s in split})==1 for g in groups),'a time group spans splits'
pool=[s for s,v in split.items() if v in ('train','val')]; n=len(pool); k=C['folds']-1; ntr=sum(map(len,tg))
def covers(idx):
    c={f:collections.Counter(meta[s][f] for s in idx) for f in C['require']}
    return all(c[f][v]>=m for f,mins in C['require'].items() for v,m in mins.items())
rng=random.Random(C['seed']); best=None
for _ in range(C['tries']):
    order=tg[:]; rng.shuffle(order); folds=[[] for _ in range(k)]
    for g in order: min(folds,key=len).extend(g)
    if all(abs(len(f)-ntr/k)<=C['tolerance']*n for f in folds) and all(map(covers,folds)): best=folds; break
if best is None: sys.exit(f"no folds found in {C['tries']} tries: loosen cv.tolerance/cv.require")
asg={s:0 for s,v in split.items() if v=='val'}|{s:i+1 for i,f in enumerate(best) for s in f}
t={s:ts[i] for i,s in enumerate(r['image'] for r in rows)}
gap=min(abs((t[a]-t[b]).total_seconds()) for a in asg for b in asg if asg[a]<asg[b])
print(f"{len(tg)} train time groups; closest photos in different folds: {gap:.0f} s")
for f in range(C['folds']):
    idx=[s for s in asg if asg[s]==f]
    print(f"fold {f} {len(idx):3}  shot_type {dict(collections.Counter(meta[s]['shot_type'] for s in idx))}  light {dict(collections.Counter(meta[s]['light'] for s in idx))}")
if '--write' in sys.argv:
    o=data/'folds.json'
    if o.exists(): sys.exit(f'{o} exists; the folds are frozen')
    json.dump({'config':{'cv':C,'split_min_gap_s':cfg['split']['min_gap_s'],'exclude':cfg.get('exclude',[])},'folds':dict(sorted(asg.items()))},open(o,'w'),indent=1); print('wrote',o)
