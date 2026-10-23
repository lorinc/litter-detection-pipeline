"""Shuffle time groups into train/val/test and freeze <data_dir>/split.json.
usage: make_split.py [--config PATH]           print the split the config produces (no file written)
       make_split.py [--config PATH] --write   write split.json once; refuses if it exists (never reshuffle)
Photos are sorted by capture time and cut into groups wherever the gap to the previous photo is >= min_gap_s,
so any two photos in different splits are at least min_gap_s apart. Groups are shuffled (seeded) and the first
assignment that hits the val/test fractions within tolerance and has at least the 'require' minimum count of photos per value in val and test
wins; config and seed make it reproducible.
"""
import csv,json,sys,random,datetime,collections
from common import load_config
cfg=load_config(); sc=cfg['split']; data=cfg['_data']
rows=sorted(csv.DictReader(open(data/'meta.csv')),key=lambda r:r['timestamp'])
ts=[datetime.datetime.strptime(r['timestamp'],'%Y:%m:%d %H:%M:%S') for r in rows]
groups=[[0]]
for i in range(1,len(rows)):
    if (ts[i]-ts[i-1]).total_seconds()>=sc['min_gap_s']: groups.append([])
    groups[-1].append(i)
n=len(rows); rng=random.Random(sc['seed'])
def covers(idx):
    c={k:collections.Counter(rows[i][k] for i in idx) for k in sc['require']}
    return all(c[k][v]>=m for k,mins in sc['require'].items() for v,m in mins.items())
best=None
for _ in range(sc['tries']):
    order=groups[:]; rng.shuffle(order); asg={}; fill={s:0 for s in sc['fractions']}
    for g in order:
        s=next((s for s,f in sc['fractions'].items() if fill[s]<f*n and fill[s]+len(g)<=(f+sc['tolerance'])*n),'train')
        for i in g: asg[i]=s
        if s!='train': fill[s]+=len(g)
    if all(abs(fill[s]/n-f)<=sc['tolerance'] for s,f in sc['fractions'].items()) and \
       all(covers([i for i in asg if asg[i]==s]) for s in sc['fractions']):
        best=asg; break
if best is None: sys.exit(f"no split found in {sc['tries']} tries: loosen tolerance/require or lower min_gap_s")
gap=min(abs((ts[i]-ts[j]).total_seconds()) for i in range(n) for j in range(i+1,n) if best[i]!=best[j])
print(f"{len(groups)} time groups at min_gap_s={sc['min_gap_s']}; closest photos in different splits: {gap:.0f} s")
for s in ('train','val','test'):
    idx=[i for i in range(n) if best[i]==s]
    print(f"{s:5} {len(idx):3} ({100*len(idx)/n:.0f}%)  groups {len({next(k for k,g in enumerate(groups) if i in g) for i in idx})}"
          f"  shot_type {dict(collections.Counter(rows[i]['shot_type'] for i in idx))}  light {dict(collections.Counter(rows[i]['light'] for i in idx))}")
if '--write' in sys.argv:
    f=data/'split.json'
    if f.exists(): sys.exit(f'{f} exists; the split is frozen')
    json.dump({'config':{k:v for k,v in cfg.items() if not k.startswith('_')},
               'split':{rows[i]['image']:best[i] for i in range(n)}},open(f,'w'),indent=1); print('wrote',f)
