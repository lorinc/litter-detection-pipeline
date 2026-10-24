"""Cross-validation report: pool every fold's held-out frames into one score per recipe, and compare two recipes.
usage: cv_report.py RECIPE [OTHER]
RECIPE's folds are the run directories runs/RECIPE_f0 .. runs/RECIPE_f<cv.folds-1> (run.py --cv), each scored by the
harness on its held-out fold (detections cached in <data_dir>/eval/RECIPE_fK/val/). Every train and val photo is
held out exactly once, so the pooled frames are all of them, each scored by the model that did not see it.
Pooled at the run's operating threshold: recall (Wilson 95%), false positives per frame (95% interval from
eval.bootstrap resamples of whole frames), recall per size bin and shot type, the recall-vs-false-positives curve,
and the same per fold. Fold 0 is the frozen val split (blind labels); folds 1.. hold train photos, whose labels were
made from OWLv2 proposals, so pooled scores compare recipes with each other, not with OWLv2.
With OTHER: a paired bootstrap over the same frames (eval.bootstrap resamples, eval.seed) of OTHER minus RECIPE in
recall and false positives per frame; a difference counts as real only when its 95% interval excludes zero.
Writes runs/cv_RECIPE.json, and with OTHER runs/cv_RECIPE_vs_OTHER.json.
"""
import json,sys,csv
import numpy as np
from common import HERE,load_config
import metrics
def pooled(name):
    k=0; frames=[]; folds=[]
    while (HERE/'runs'/f'{name}_f{k}'/'config.json').exists():
        cfg=load_config(HERE/'runs'/f'{name}_f{k}'/'config.json'); assert cfg['cv']['fold']==k
        r=json.load(open(HERE/'runs'/f'{name}_f{k}'/f'{name}_f{k}_val.json')); data=cfg['_data']; E=cfg['eval']
        meta={m['image']:m for m in csv.DictReader(open(data/'meta.csv'))}
        fr=[metrics.frame(json.load(open(data/'eval'/f'{name}_f{k}'/'val'/f'{s}.json')),json.load(open(data/'labels'/f'{s}.json')),
                          meta[s]['shot_type'],E['match_iou'])|{'fold':k} for s in r['images']]
        frames+=fr; folds.append({'fold':k,'images':len(fr),**metrics.at(fr,r['threshold'])}); thr=r['threshold']; k+=1
    assert k and k==cfg['cv']['folds'],f'{name}: {k} fold run directories, config says {cfg["cv"]["folds"]}'
    assert len({f['image'] for f in frames})==len(frames),'a frame is held out twice'
    return frames,folds,thr,E
def boot(frames,E,fn):
    rng=np.random.default_rng(E['seed']); v=[fn([frames[i] for i in rng.integers(0,len(frames),len(frames))]) for _ in range(E['bootstrap'])]
    return [round(float(np.percentile(v,2.5)),3),round(float(np.percentile(v,97.5)),3)]
names=[a for a in sys.argv[1:] if not a.startswith('--')]; out={}
for n in names:
    frames,folds,thr,E=pooled(n); bins=E['bins']; op=metrics.at(frames,thr)
    op['fp_per_frame_ci95']=boot(frames,E,lambda f:metrics.at(f,thr)['fp_per_frame'])
    by_bin=[]
    for i in range(len(bins)-1):
        r=[sc for f in frames for sz,sc in f['butts'] if bins[i]<=sz<bins[i+1]]; kk=sum(sc>=thr for sc in r)
        by_bin.append({'bin':f"{bins[i]}-{bins[i+1] if bins[i+1]<100000 else ''}",'n':len(r),'found':kk,'recall':round(kk/len(r),3) if r else None,'ci95':metrics.wilson(kk,len(r))})
    by_shot={sh:metrics.at([f for f in frames if f['shot']==sh],thr) for sh in sorted({f['shot'] for f in frames})}
    curve=[{'threshold':t,**{k:v for k,v in metrics.at(frames,t).items() if k in ('recall','fp_per_frame','precision')}}
           for t in sorted({sc for f in frames for sc,_ in f['dets']},reverse=True)]
    out[n]=(frames,thr,E)
    json.dump({'recipe':n,'threshold':thr,'operating_point':op,'folds':folds,'by_size_bin':by_bin,'by_shot':by_shot,'curve':curve,
               'images':sorted(f['image'] for f in frames)},open(HERE/'runs'/f'cv_{n}.json','w'),indent=1)
    print(f"{n}: {op['frames']} frames, {op['butts']} butts, threshold {thr}: recall {op['recall']} {op['recall_ci95']}, "
          f"fp/frame {op['fp_per_frame']} {op['fp_per_frame_ci95']}")
    for f in folds: print(f"  fold {f['fold']}: {f['images']} frames, recall {f['recall']} ({f['found']}/{f['butts']}), fp/frame {f['fp_per_frame']}")
if len(names)==2:
    (a,ta,E),(b,tb,_)=out[names[0]],out[names[1]]; bi={f['image']:f for f in b}
    assert set(bi)=={f['image'] for f in a},'the recipes were not scored on the same frames'
    pairs=[(f,bi[f['image']]) for f in a]
    def diff(p,key):
        x,y=metrics.at([q[0] for q in p],ta),metrics.at([q[1] for q in p],tb); return y[key]-x[key]
    res={'base':names[0],'other':names[1],'frames':len(pairs)}
    for key in ('recall','fp_per_frame'):
        res[key]={'base':metrics.at(a,ta)[key],'other':metrics.at(b,tb)[key],'diff':round(diff(pairs,key),3),'diff_ci95':boot(pairs,E,lambda p:diff(p,key))}
        res[key]['real']=not (res[key]['diff_ci95'][0]<=0<=res[key]['diff_ci95'][1])
        print(f"{names[1]} - {names[0]} {key}: {res[key]['diff']} {res[key]['diff_ci95']} {'real' if res[key]['real'] else 'inside noise'}")
    json.dump(res,open(HERE/'runs'/f'cv_{names[0]}_vs_{names[1]}.json','w'),indent=1)
