"""Label-completeness check: re-annotate a few train photos blind and compare with their proposal-assisted labels.
usage: blind_check.py --pick [N] [--config PATH]      choose the photos and write the blind session (default N 10)
       blind_check.py [--blind DIR] [--config PATH]   compare (DIR default <data_dir>/blind_check/labels)
--pick writes <data_dir>/blind_check/: split.json (the picked photos, all marked 'val', so annotate.py serves no
proposals and the page opens them blind), config.json (data_dir data/blind_check, the annotate block copied) and
an images link to ../images. The session's labels go to <data_dir>/blind_check/labels/; <data_dir>/labels/ is never
read or written by it. Start it with: annotate.py --config ../data/blind_check/config.json
Pick rule: pool = train photos of cv folds 1.. (data/folds.json, fold 0 is val), minus the config's excluded ones,
that were reviewed with proposals (at least one rejected or covered proposal; the photos annotated before their
proposals existed are already blind). Score = labelled boxes x rejected proposals (a fair number of both). One
photo per (fold, light) cell, the highest score; then, in rounds over the cells (largest first), the next-highest
photo of a surface not yet picked, and once none is left, the next-highest of any surface, until N.
Ties are broken by a shuffle with seed 12. Refuses to re-pick once blind labels exist.
Compare: a blind box and an original box match when either's centre lies inside the other or IoU >= eval.match_iou,
paired one to one by descending IoU. Per photo and in total: matched pairs by class (original/blind), blind-only
and original-only boxes by class, and how many blind-only butts sit on a rejected proposal (seen and turned down)
or on none (never proposed, never drawn). Estimated missed share = blind butts with no original box of any class /
all blind butts, Wilson 95%; and the reverse, original butts with no blind box / all original butts. One annotator
re-labelling their own photos: an estimate of self-consistency, not of true completeness. Photos not marked done
in the blind pass are skipped. Writes compare.json beside DIR (<data_dir>/blind_check/compare.json).
"""
import json,sys,csv,random,pathlib
from common import HERE,load_config,load_split
import metrics
cfg=load_config(); data=cfg['_data']; out=data/'blind_check'
def arg(k,d=None): return sys.argv[sys.argv.index(k)+1] if k in sys.argv else d
def box(b): return (b['x'],b['y'],b['x']+b['w'],b['y']+b['h'])
def pick(n):
    assert not any((out/'labels').glob('*.json')),f'{out}/labels already holds blind labels; not re-picking'
    split=load_split(cfg); folds=json.load(open(data/'folds.json'))['folds']; meta={m['image']:m for m in csv.DictReader(open(data/'meta.csv'))}
    pool=[]
    for s in sorted(split):
        if split[s]!='train' or folds[s]==0: continue
        l=json.load(open(data/'labels'/f'{s}.json')); rej=len(l.get('rejected_proposals',[]))
        if rej+len(l.get('covered_proposals',[])): pool.append((s,folds[s],meta[s]['light'],meta[s]['surface'],len(l['boxes'])*rej))
    random.Random(12).shuffle(pool); pool.sort(key=lambda p:-p[4])
    cells={}
    for p in pool: cells.setdefault((p[1],p[2]),[]).append(p)
    got=[c[0] for c in cells.values()]
    for new in (True,True,False,False):
        for c in sorted(cells.values(),key=len,reverse=True):
            q=[p for p in c if p not in got and not (new and p[3] in {g[3] for g in got})]
            if q and len(got)<n: got.append(q[0])
    got=sorted(got[:n])
    out.mkdir(exist_ok=True)
    json.dump({'split':{p[0]:'val' for p in got},'rule':'blind_check.py --pick: see its docstring; seed 12',
               'photos':[{'image':p[0],'fold':p[1],'light':p[2],'surface':p[3],'score':p[4]} for p in got]},open(out/'split.json','w'),indent=1)
    json.dump({'data_dir':str(out.relative_to(HERE)),'exclude':[],'annotate':cfg['annotate']},open(out/'config.json','w'),indent=1)
    if not (out/'images').exists(): (out/'images').symlink_to('../images')
    for p in got: print(*p,sep='  ')
    print(f"{len(got)} photos from a pool of {len(pool)}; session: annotate.py --config ../data/blind_check/config.json")
def compare(bdir):
    E=cfg['eval']; m=E['match_iou']; rows=[]
    for s in json.load(open(out/'split.json'))['split']:
        f=bdir/f'{s}.json'
        if not f.exists() or json.load(open(f)).get('status')!='done': print(f'{s}: blind pass not done, skipped'); continue
        o=json.load(open(data/'labels'/f'{s}.json')); b=json.load(open(f))['boxes']; ob=o['boxes']
        pairs=sorted(((metrics.iou(box(x),box(y)),i,j) for i,x in enumerate(ob) for j,y in enumerate(b)
                      if metrics.hit(box(x),box(y),m) or metrics.hit(box(y),box(x),m)),reverse=True)
        ui,uj,r={*range(len(ob))},{*range(len(b))},{}
        for _,i,j in pairs:
            if i in ui and j in uj: ui.discard(i); uj.discard(j); k=f"{ob[i]['cls']}/{b[j]['cls']}"; r[k]=r.get(k,0)+1
        for i in ui: k=f"{ob[i]['cls']}/-"; r[k]=r.get(k,0)+1
        for j in uj: k=f"-/{b[j]['cls']}"; r[k]=r.get(k,0)+1
        rej=[box(p) for p in o.get('rejected_proposals',[])]
        r['-/butt on rejected proposal']=sum(b[j]['cls']=='butt' and any(metrics.hit(p,box(b[j]),m) or metrics.hit(box(b[j]),p,m) for p in rej) for j in uj)
        rows.append({'image':s,**r}); print(s,'  '.join(f'{k} {v}' for k,v in sorted(r.items()) if v))
    keys=sorted({k for r in rows for k in r if k!='image'}); tot={k:sum(r.get(k,0) for r in rows) for k in keys}; g=lambda k:tot.get(k,0)
    nb=g('butt/butt')+g('maybe/butt')+g('-/butt'); no=g('butt/butt')+g('butt/maybe')+g('butt/-')
    res={'photos':len(rows),'total':tot,'per_photo':rows,
         'missed_by_original':{'k':g('-/butt'),'n':nb,'share':round(g('-/butt')/nb,3) if nb else None,'ci95':metrics.wilson(g('-/butt'),nb)},
         'missed_by_blind':{'k':g('butt/-'),'n':no,'share':round(g('butt/-')/no,3) if no else None,'ci95':metrics.wilson(g('butt/-'),no)},
         'note':'one annotator re-labelling their own photos blind: an estimate of self-consistency, not of true completeness'}
    print('total (original/blind):','  '.join(f'{k} {v}' for k,v in tot.items() if v))
    for k,t in (('missed_by_original','blind butts with no original box'),('missed_by_blind','original butts with no blind box')):
        x=res[k]; print(f"{t}: {x['k']}/{x['n']} = {x['share']} (Wilson 95% {x['ci95']}), estimate")
    json.dump(res,open(bdir.parent/'compare.json','w'),indent=1)
if '--pick' in sys.argv:
    a=sys.argv[sys.argv.index('--pick')+1:]; pick(int(a[0]) if a and a[0].isdigit() else 10)
else: compare(pathlib.Path(arg('--blind',out/'labels')))
