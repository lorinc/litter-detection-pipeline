"""Task 5, report: recall per butt size at inference scale, pooled over downscale factors, from the cached detections.
usage: downscale_report.py [--config PATH] [--split val]   writes output/complete/downscale/<split>.json and a plot
Ground truth: annotator boxes of class 'butt'; 'maybe' boxes are neither finds nor misses (detections on them are
not false positives). A butt is found at a factor when a detection with score >= recall_threshold has its centre
inside the butt box or IoU >= match_iou with it. Size = box long side at native scale / factor. Confidence intervals:
Wilson 95%. Floor = lower edge of the smallest size bin from which every bin upward has recall >= floor_recall.
Curve floor = size where a logistic fit of found ~ log(size) crosses floor_recall, with a 95% interval from 1000
bootstrap resamples of whole images; reported as null when the crossing lies below the smallest size tested.
"""
import json,sys,csv,math,collections
from common import HERE,load_config,load_split
import metrics
cfg=load_config(); data=cfg['_data']; D=cfg['downscale']
which=sys.argv[sys.argv.index('--split')+1] if '--split' in sys.argv else 'val'
split=load_split(cfg); meta={r['image']:r for r in csv.DictReader(open(data/'meta.csv'))}
imgs=sorted(s for s,v in split.items() if v==which); bins=D['bins']
def hit(d,g): return metrics.hit(d,g,D['match_iou'])
wilson=metrics.wilson
def binof(x): return next(i for i in range(len(bins)-1) if bins[i]<=x<bins[i+1])
def run(thr):
    rows=[]; frames=collections.defaultdict(lambda:{'frames':0,'boxes':0,'fp':0})
    for f in D['factors']:
        for s in imgs:
            cf=data/'downscale'/f'f{f}'/f'{s}.json'
            if not cf.exists(): continue
            lab=json.load(open(data/'labels'/f'{s}.json'))
            gt=[(b['x'],b['y'],b['x']+b['w'],b['y']+b['h'],b['cls']) for b in lab['boxes']]
            dets=[h[1:] for h in json.load(open(cf))['hits'] if h[0]>=thr]
            fr=frames[f]; fr['frames']+=1; fr['boxes']+=len(dets)
            fr['fp']+=sum(1 for d in dets if not any(hit(d,g[:4]) for g in gt))
            for g in gt:
                if g[4]!='butt': continue
                rows.append({'factor':f,'image':s,'shot':meta[s]['shot_type'],'size':max(g[2]-g[0],g[3]-g[1])/f,
                             'found':any(hit(d,g[:4]) for d in dets)})
    return rows,frames
def table(rows):
    out=[]
    for i in range(len(bins)-1):
        r=[x for x in rows if binof(x['size'])==i]; k=sum(x['found'] for x in r)
        out.append({'bin':f"{bins[i]}-{bins[i+1] if bins[i+1]<100000 else ''}",'lo':bins[i],'n':len(r),'found':k,
                    'recall':round(k/len(r),3) if r else None,'ci95':wilson(k,len(r))})
    return out
def floor(tab):
    fl=None
    for t in reversed(tab):
        if t['n']==0: continue
        if t['recall']>=D['floor_recall']: fl=t['lo']
        else: break
    return fl
rows,frames=run(D['recall_threshold']); pooled=table(rows)
res={'split':which,'images':len(imgs),'threshold':D['recall_threshold'],'butts_per_factor':{str(f):sum(1 for x in rows if x['factor']==f) for f in D['factors']},
     'pooled':pooled,'floor_px':floor(pooled),
     'by_shot':{sh:table([x for x in rows if x['shot']==sh]) for sh in sorted({x['shot'] for x in rows})},
     'by_factor':{str(f):{'recall':round(sum(x['found'] for x in rows if x['factor']==f)/max(1,sum(1 for x in rows if x['factor']==f)),3),
                          'boxes_per_frame':round(v['boxes']/v['frames'],1),'unmatched_per_frame':round(v['fp']/v['frames'],1),'frames':v['frames']}
                  for f,v in frames.items()}}
rows2,frames2=run(D['sensitivity_threshold']); t2=table(rows2)
res['sensitivity']={'threshold':D['sensitivity_threshold'],'floor_px':floor(t2),'pooled':t2,
                    'boxes_per_frame':{str(f):round(v['boxes']/v['frames'],1) for f,v in frames2.items()}}
res['floor_px_by_shot']={sh:floor(t) for sh,t in res['by_shot'].items()}
import numpy as np
from sklearn.linear_model import LogisticRegression
def curve_floor(rs):
    X=np.log([[x['size']] for x in rs]); y=np.array([x['found'] for x in rs])
    if len(set(y))<2: return None
    m=LogisticRegression(C=1e6,max_iter=1000).fit(X,y); a,b=m.coef_[0][0],m.intercept_[0]
    v=float(np.exp((math.log(D['floor_recall']/(1-D['floor_recall']))-b)/a)) if a>0 else None
    return v if v is not None and v>=min(x['size'] for x in rs) else None  # extrapolated below the tested sizes: not estimable
def boot(rs,n=1000):
    rng=np.random.default_rng(0); by=collections.defaultdict(list)
    for x in rs: by[x['image']].append(x)
    keys=list(by); vals=[]
    for _ in range(n):
        v=curve_floor([x for k in rng.choice(keys,len(keys)) for x in by[k]])
        if v is not None: vals.append(v)
    return ([round(float(np.percentile(vals,2.5)),1),round(float(np.percentile(vals,97.5)),1)] if vals else None),round(len(vals)/n,3)
cf=curve_floor(rows); res['curve_floor_px']=round(cf,1) if cf else None
ci,share=boot(rows) if cf else (None,None); res['curve_floor_ci95']=ci; res['curve_floor_estimable_share']=share
res['curve_floor_by_shot']={sh:(round(v,1) if (v:=curve_floor([x for x in rows if x['shot']==sh])) else None) for sh in res['by_shot']}
od=HERE/'output/complete/downscale'; od.mkdir(parents=True,exist_ok=True); json.dump(res,open(od/f'{which}.json','w'),indent=1)
print(f"{which}: {len(imgs)} images, threshold {D['recall_threshold']}, butts per factor {res['butts_per_factor']}")
for t in pooled: print(f"  {t['bin']:>9} px  n={t['n']:3}  found={t['found']:3}  recall={t['recall']}  CI95={t['ci95']}")
print('  floor (recall >= %.2f from here up): %s px'%(D['floor_recall'],res['floor_px']),'| by shot',res['floor_px_by_shot'],'| at %.2f: %s px'%(D['sensitivity_threshold'],res['sensitivity']['floor_px']))
print('  curve floor:',res['curve_floor_px'],'px, 95% interval',res['curve_floor_ci95'],'(estimable in',res['curve_floor_estimable_share'],'of resamples)','| by shot',res['curve_floor_by_shot'])
for f,v in res['by_factor'].items(): print(f"  factor {f}: recall {v['recall']}, boxes/frame {v['boxes_per_frame']}, unmatched/frame {v['unmatched_per_frame']} ({v['frames']} frames)")
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig,(a1,a2)=plt.subplots(1,2,figsize=(10,4))
def curve(ax,tab,lab,col,ls='-'):
    xs=[math.sqrt(max(t['lo'],5)*min(bins[i+1],400)) for i,t in enumerate(tab) if t['n']]
    ys=[t['recall'] for t in tab if t['n']]; lo=[t['recall']-t['ci95'][0] for t in tab if t['n']]; hi=[t['ci95'][1]-t['recall'] for t in tab if t['n']]
    ax.errorbar(xs,ys,yerr=[lo,hi],label=lab,color=col,ls=ls,marker='o',ms=4,capsize=2,lw=1.5)
curve(a1,pooled,f"all shots (score ≥ {D['recall_threshold']})",'#d2691e')
curve(a1,t2,f"all shots (score ≥ {D['sensitivity_threshold']})",'#999','--')
a1.axhline(D['floor_recall'],color='k',lw=.5); a1.set_xscale('log'); a1.set_ylim(0,1.05)
a1.set_xlabel('butt long side at inference scale (px)'); a1.set_ylabel('recall'); a1.legend(frameon=False,fontsize=8)
a1.set_title(f'OWLv2 recall by butt size ({which}, pooled over factors)',fontsize=9)
fs=sorted(res['by_factor'],key=float); a2.plot([float(f) for f in fs],[res['by_factor'][f]['boxes_per_frame'] for f in fs],'o-',color='#d2691e',label='all boxes')
a2.plot([float(f) for f in fs],[res['by_factor'][f]['unmatched_per_frame'] for f in fs],'o--',color='#999',label='not on a labelled butt')
a2.set_xlabel('downscale factor'); a2.set_ylabel('boxes per frame'); a2.legend(frameon=False,fontsize=8); a2.set_title(f"boxes per frame (score ≥ {D['recall_threshold']})",fontsize=9)
fig.tight_layout(); fig.savefig(HERE/f'output/complete/fig/downscale_{which}.png',dpi=130)
