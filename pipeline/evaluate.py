"""Evaluation harness: score any detector on whole frames of one split, the same way for every detector.
usage: evaluate.py NAME [--config PATH] [--split val] [--out DIR]
NAME is a key of eval.detectors in the config; its 'adapter' names a class in detectors.py and its 'threshold' is
the detector's operating score threshold.
Detection pass (resumable): each frame is run through the adapter (tiling, mapping back and merging happen inside
it), timed after a warm-up, and every kept box is cached to <data_dir>/eval/<NAME>/<split>/<image>.json.
Scoring: ground truth = annotator boxes of class 'butt'; 'maybe' boxes are neither finds nor misses, and detections
on them are not false positives. A detection hits a box by the task 5 rule (centre inside, or IoU >= eval.match_iou).
A butt is found when any detection at or above the threshold hits it; a false positive is a detection that hits no
labelled box (several detections on one butt are not penalised). Reported at the operating threshold: recall
(Wilson 95%), false positives per frame (95% interval from eval.bootstrap resamples of whole frames), boxes per
frame, recall per size bin (box long side, native px) and per shot type; a sweep over all scores gives the
precision-recall and recall-vs-false-positives curves; plus ms per frame on this machine, parameters and FLOPs.
Writes <NAME>_<split>.json (with config, git commit and a hash of the split's label files) and
eval_<NAME>_<split>.png: to --out DIR (a run directory) if given, else to output/complete/eval/ and output/complete/fig/.
"""
import json,sys,csv,time,hashlib,subprocess,platform,statistics,pathlib
import numpy as np,torch
from PIL import Image
from common import HERE,load_config,load_split
import metrics,detectors
cfg=load_config(); data=cfg['_data']; E=cfg['eval']
name=next(a for i,a in enumerate(sys.argv[1:],1) if not a.startswith('--') and sys.argv[i-1] not in ('--config','--split','--out'))
which=sys.argv[sys.argv.index('--split')+1] if '--split' in sys.argv else E['split']
entry=E['detectors'][name]; thr=entry['threshold']; bins=E['bins']
split=load_split(cfg); meta={r['image']:r for r in csv.DictReader(open(data/'meta.csv'))}
imgs=sorted(s for s,v in split.items() if v==which)
det=detectors.ADAPTERS[entry['adapter']](cfg,entry)
# detection pass
cache=data/'eval'/name/which; cache.mkdir(parents=True,exist_ok=True); warm=False
for s in imgs:
    o=cache/f'{s}.json'
    if o.exists(): continue
    if not warm: det.warmup(); warm=True
    im=Image.open(data/'images'/f'{s}.png').convert('RGB')
    t0=time.perf_counter(); hits=det.detect(im); ms=(time.perf_counter()-t0)*1000
    json.dump({'image':s,'size':[im.width,im.height],'ms':round(ms),'hits':[[round(h[0],4)]+[round(v,1) for v in h[1:]] for h in hits],
               'settings':det.settings},open(o,'w'))
    print(f'{s}: {len(hits)} boxes, {ms/1000:.0f} s',flush=True)
# scoring
frames=[metrics.frame(json.load(open(cache/f'{s}.json')),json.load(open(data/'labels'/f'{s}.json')),meta[s]['shot_type'],E['match_iou']) for s in imgs]
at=metrics.at
rng=np.random.default_rng(E['seed'])
boot=[at([frames[i] for i in rng.integers(0,len(frames),len(frames))],thr)['fp_per_frame'] for _ in range(E['bootstrap'])]
op=at(frames,thr); op['fp_per_frame_ci95']=[round(float(np.percentile(boot,2.5)),2),round(float(np.percentile(boot,97.5)),2)]
by_bin=[]
for i in range(len(bins)-1):
    r=[sc for f in frames for sz,sc in f['butts'] if bins[i]<=sz<bins[i+1]]; k=sum(sc>=thr for sc in r)
    by_bin.append({'bin':f"{bins[i]}-{bins[i+1] if bins[i+1]<100000 else ''}",'n':len(r),'found':k,
                   'recall':round(k/len(r),3) if r else None,'ci95':metrics.wilson(k,len(r))})
by_shot={sh:at([f for f in frames if f['shot']==sh],thr) for sh in sorted({f['shot'] for f in frames})}
ts=sorted({sc for f in frames for sc,_ in f['dets']},reverse=True)
curve=[{'threshold':t,**{k:v for k,v in at(frames,t).items() if k in ('recall','fp_per_frame','precision','boxes_per_frame')}} for t in ts]
ms=[f['ms'] for f in frames]; w,h=frames[0]['size']; inf=det.info(w,h)
lh=hashlib.sha256(); [lh.update(s.encode()+open(data/'labels'/f'{s}.json','rb').read()) for s in imgs]
git=lambda *a:subprocess.run(['git',*a],cwd=HERE,capture_output=True,text=True).stdout.strip()
res={'detector':name,'split':which,'threshold':thr,'match_iou':E['match_iou'],'operating_point':op,'by_size_bin':by_bin,'by_shot':by_shot,
     'speed':{'ms_per_frame_mean':round(statistics.mean(ms)),'ms_per_frame_median':round(statistics.median(ms)),
              'machine':platform.processor() or platform.machine(),'cpu':next((l.split(':',1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')),None),
              'torch':torch.__version__,'frame_size':[w,h]},
     'compute':{**inf,'gflops_per_frame':round(inf['flops_per_frame']/1e9,1)},'curve':curve,
     'labels_sha256':lh.hexdigest(),'git_commit':git('rev-parse','HEAD'),'git_dirty':bool(git('status','--porcelain','--','pipeline')),
     'images':imgs,'settings':det.settings,'config':{k:v for k,v in cfg.items() if not k.startswith('_')}}
od=pathlib.Path(sys.argv[sys.argv.index('--out')+1]) if '--out' in sys.argv else HERE/'output/complete/eval'
fd=od if '--out' in sys.argv else HERE/'output/complete/fig'; od.mkdir(parents=True,exist_ok=True); json.dump(res,open(od/f'{name}_{which}.json','w'),indent=1)
print(f"{name} on {which}: {len(imgs)} frames, {op['butts']} butts, threshold {thr}")
print(f"  recall {op['recall']} {op['recall_ci95']}, false positives/frame {op['fp_per_frame']} {op['fp_per_frame_ci95']}, boxes/frame {op['boxes_per_frame']}, precision {op['precision']}")
for b in by_bin: print(f"  {b['bin']:>9} px  n={b['n']:3}  found={b['found']:3}  recall={b['recall']}  CI95={b['ci95']}")
for sh,v in by_shot.items(): print(f"  {sh}: {v['frames']} frames, recall {v['recall']} ({v['found']}/{v['butts']}), fp/frame {v['fp_per_frame']}")
print(f"  {res['speed']['ms_per_frame_mean']} ms/frame (median {res['speed']['ms_per_frame_median']}), {inf['params']/1e6:.1f} M params, {res['compute']['gflops_per_frame']} GFLOPs/frame")
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig,(a1,a2)=plt.subplots(1,2,figsize=(10,4))
a1.plot([c['recall'] for c in curve if c['precision'] is not None],[c['precision'] for c in curve if c['precision'] is not None],color='#d2691e')
a1.plot([op['recall']],[op['precision']],'ko',ms=4,label=f'threshold {thr}'); a1.set_xlim(0,1.02); a1.set_ylim(0,1.02)
a1.set_xlabel('recall'); a1.set_ylabel('precision'); a1.legend(frameon=False,fontsize=8); a1.set_title(f'{name}: precision-recall ({which}, whole frames)',fontsize=9)
a2.plot([c['fp_per_frame'] for c in curve],[c['recall'] for c in curve],color='#d2691e'); a2.plot([op['fp_per_frame']],[op['recall']],'ko',ms=4)
a2.set_xlabel('false positives per frame'); a2.set_ylabel('recall'); a2.set_ylim(0,1.02); a2.set_title(f'{name}: recall vs false positives per frame',fontsize=9)
fig.tight_layout(); fig.savefig(fd/f'eval_{name}_{which}.png',dpi=130)
