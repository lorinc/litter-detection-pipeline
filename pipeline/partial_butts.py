"""Partly visible butts: how often training tiles hold a cut-off butt left as background (export.min_visible), and
whether trained models' scores depend on where a butt sits relative to the 640 px tile grid. No training.
usage: partial_butts.py train                     training-data counts (export planning only, no pixels)
       partial_butts.py infer RECIPE [RECIPE ...]  per-tile raw detections of each fold model on its held-out photos
       partial_butts.py report RECIPE [RECIPE ...]  inference-side tables
train: for run 02 and each fold of 04-cv-hard-negatives, the train tiles exactly as exported (tile names from the run's
kaggle/export.json, contents from export.plan, target counts checked against export.json). Per kept tile, every butt
box that overlaps it has a visible share v = area inside the tile / box area; v >= min_visible is a target, a smaller
v is a fragment trained as background (all fragments are cut by an inner tile edge). Shares are of the box, not of the
butt's pixels.
infer: grid tiles (tiling.starts, fit=True) through the run's YOLO adapter at its raw threshold, cached per photo to
<data_dir>/eval_tiles/RUN/<image>.json (resumable; only tiles with a box are stored). Merging the cached tiles with
tiling.tiled reproduces the harness cache in <data_dir>/eval/RUN/val/ (as a set: boxes with tied scores may come
out in another order); the report checks that.
report: butt-in-tile pairs, with a detection credited to a butt by the harness rule (metrics.hit, full-frame coords,
'maybe' boxes ignored): per visible-share bin, the share of pairs with a detection >= the threshold and score quantiles;
for butts wholly inside a tile, the same by margin = distance to the nearest inner tile edge (edges on the frame border
do not cut anything and are not counted); a within-butt paired comparison (score in the tile where the butt is nearest
an inner edge minus the tile where it is farthest, for butts wholly inside two or more tiles), with a bootstrap over
photos; photo-level recall (merged harness cache) by the butt's best margin and by how many tiles hold it wholly, with
Wilson intervals, also split by size and for wide shots only. Writes <data_dir>/eval_tiles/partial_butts.json
(train) and <data_dir>/eval_tiles/partial_butts_RECIPE.json (report).
"""
import json,sys,csv,collections
import numpy as np
from common import HERE,load_config,load_split
import tiling,metrics
mode=sys.argv[1]; args=[a for a in sys.argv[2:] if not a.startswith('--')]
OUT=HERE/'data'/'eval_tiles'; OUT.mkdir(parents=True,exist_ok=True)
VBINS=[0,0.25,0.5,0.8,1.0+1e-9]; MBINS=[0,16,48,96,160,10**6]
def runs(recipe):
    k=0
    while (HERE/'runs'/f'{recipe}_f{k}'/'config.json').exists(): yield f'{recipe}_f{k}',load_config(HERE/'runs'/f'{recipe}_f{k}'/'config.json'); k+=1
def butts(lab): return [(b['x'],b['y'],b['x']+b['w'],b['y']+b['h']) for b in lab['boxes'] if b['cls']=='butt']
def share(g,tx,ty,T):
    ix=min(g[2],tx+T)-max(g[0],tx); iy=min(g[3],ty+T)-max(g[1],ty)
    return ix*iy/((g[2]-g[0])*(g[3]-g[1])) if ix>0 and iy>0 else 0
def margin(g,tx,ty,T,W,H):
    """Distance from a box wholly inside the tile to the nearest tile edge that is not on the frame border."""
    d=[g[0]-tx if tx>0 else None,g[1]-ty if ty>0 else None,tx+T-g[2] if tx+T<W else None,ty+T-g[3] if ty+T<H else None]
    return min((v for v in d if v is not None),default=10**6)
def vbin(v): return next(i for i in range(len(VBINS)-1) if VBINS[i]<=v<VBINS[i+1])
def label(lo,hi,u=''): return f"{lo}-{hi}{u}" if hi<10**5 else f"{lo}{u} and up"
if mode=='train':
    import export
    res={}
    for name,cfgp in [('02-maybe-masked',HERE/'runs/02-maybe-masked/config.json')]+[(n,HERE/'runs'/n/'config.json') for n,_ in runs('04-cv-hard-negatives')]:
        cfg=load_config(cfgp); X=cfg['export']; export.X,export.f,export.T,export.S,export.data=X,X['downscale'],X['tile'],X['stride'],cfg['_data']
        ej=[t for t in json.load(open(HERE/'runs'/name/'kaggle/export.json'))['tiles'] if t['split']=='train']
        want=collections.Counter(t['tile'] for t in ej); ntg={t['tile']:t['targets'] for t in ej}
        split=load_split(cfg); kept=[]; grid=[]
        for s in sorted(s for s,v in split.items() if v=='train'):
            for t in export.plan(s,'train'):
                if not t['hard']: grid.append(t)
                if want[t['tile']]>0: want[t['tile']]-=1; kept.append(t)
        assert not +want,f'{name}: tiles in export.json not planned: {+want}'
        assert all(len(t['targets'])==ntg[t['tile']] for t in kept),f'{name}: target counts differ from export.json'
        labs={}; T=X['tile']; mv=X['min_visible']; frag=[]; tg=0; tiles_with=0; frag_only=0; hn_with=0
        for t in kept:
            lab=labs.setdefault(t['image'],json.load(open(cfg['_data']/'labels'/f"{t['image']}.json")))
            vs=[share(g,t['x'],t['y'],T) for g in butts(lab)]; fr=[v for v in vs if 0<v<mv]
            assert sum(v>=mv for v in vs)==len(t['targets'])
            tg+=len(t['targets']); frag+=fr; tiles_with+=bool(fr); frag_only+=bool(fr) and not t['targets']; hn_with+=bool(fr) and t['hard']
        h=[sum(VBINS[i]<=v<VBINS[i+1] for v in frag) for i in range(3)]
        r={'train_photos':sum(v=='train' for v in split.values()),'tiles':len(kept),'tiles_with_target':sum(bool(t['targets']) for t in kept),
           'targets':tg,'tiles_with_fragment':tiles_with,'tiles_with_fragment_no_target':frag_only,'hard_tiles_with_fragment':hn_with,
           'fragments':len(frag),'fragments_by_share':{label(VBINS[i],VBINS[i+1]):h[i] for i in range(3)},
           'fragments_ge_0.25':h[1]+h[2],'tiles_with_fragment_ge_0.25':sum(any(0.25<=share(g,t['x'],t['y'],T)<mv for g in butts(labs[t['image']])) for t in kept),
           'fragments_ge_0.25_per_target':round((h[1]+h[2])/tg,3),'fragment_share_quartiles':[round(float(q),3) for q in np.percentile(frag,[25,50,75])]}
        gf=[share(g,t['x'],t['y'],T) for t in grid for g in butts(labs.setdefault(t['image'],json.load(open(cfg['_data']/'labels'/f"{t['image']}.json"))))]
        nb=sum(len(butts(labs[s])) for s in labs)
        bf=sum(any(0.25<=share(g,t['x'],t['y'],T)<mv for t in grid if t['image']==s) for s in labs for g in butts(labs[s]))
        ba=sum(any(0<share(g,t['x'],t['y'],T)<mv for t in grid if t['image']==s) for s in labs for g in butts(labs[s]))
        r['full_grid']={'butts_with_any_fragment':ba,'tiles':len(grid),'targets':sum(v>=mv for v in gf),'fragments_ge_0.25':sum(0.25<=v<mv for v in gf),'butts':nb,'butts_with_a_fragment_ge_0.25':bf}
        res[name]=r; print(name,r)
    json.dump(res,open(OUT/'partial_butts.json','w'),indent=1)
elif mode=='infer':
    import torch,detectors
    from PIL import Image
    for recipe in args:
        for name,cfg in runs(recipe):
            e=cfg['eval']['detectors'][name]; T,S=e['tile'],e['stride']; det=None
            od=OUT/name; od.mkdir(exist_ok=True)
            for s in sorted(s for s,v in load_split(cfg).items() if v=='val'):
                if (od/f'{s}.json').exists(): continue
                if det is None: det=detectors.YOLO(cfg,e); det.warmup()
                im=Image.open(cfg['_data']/'images'/f'{s}.png').convert('RGB'); tl=[]
                for ty in tiling.starts(im.height,T,S,True):
                    for tx in tiling.starts(im.width,T,S,True):
                        d=det._tile(im.crop((tx,ty,tx+T,ty+T)))
                        if d: tl.append([tx,ty,[[round(h[0],4)]+[round(v,1) for v in h[1:]] for h in d]])
                json.dump({'image':s,'run':name,'size':[im.width,im.height],'tile':T,'stride':S,'raw_threshold':e['raw_threshold'],'tiles':tl},open(od/f'{s}.json','w'))
                print(name,s,len(tl),'tiles with boxes',flush=True)
elif mode=='report':
    for recipe in args:
        pairs=[]; bt=[]; check=[0,0]
        for name,cfg in runs(recipe):
            E=cfg['eval']; e=E['detectors'][name]; thr=e['threshold']; T,S=e['tile'],e['stride']; mi=E['match_iou']
            meta={m['image']:m for m in csv.DictReader(open(cfg['_data']/'meta.csv'))}
            for s in sorted(s for s,v in load_split(cfg).items() if v=='val'):
                c=json.load(open(OUT/name/f'{s}.json')); lab=json.load(open(cfg['_data']/'labels'/f'{s}.json')); W,H=c['size']
                tl={(tx,ty):[(h[0],h[1]+tx,h[2]+ty,h[3]+tx,h[4]+ty) for h in d] for tx,ty,d in c['tiles']}
                grid=[(tx,ty) for ty in tiling.starts(H,T,S,True) for tx in tiling.starts(W,T,S,True)]
                it=iter(grid); merged=tiling.tiled(type('F',(),{'width':W,'height':H,'crop':lambda self,b:None})(),T,S,e['nms_iou'],
                                                   lambda _:[(h[0],h[1]-tx,h[2]-ty,h[3]-tx,h[4]-ty) for tx,ty in [next(it)] for h in tl.get((tx,ty),[])],fit=True)
                hc=json.load(open(cfg['_data']/'eval'/name/'val'/f'{s}.json'))['hits']
                check[0]+=1; check[1]+=len(merged)==len(hc) and all(abs(a[0]-b[0])<2e-3 and max(abs(x-y) for x,y in zip(a[1:],b[1:]))<1 for a,b in zip(sorted(merged),sorted(map(tuple,hc))))
                for g in butts(lab):
                    sz=max(g[2]-g[0],g[3]-g[1]); full=[]
                    for tx,ty in grid:
                        v=share(g,tx,ty,T)
                        if not v: continue
                        sc=max([h[0] for h in tl.get((tx,ty),[]) if metrics.hit(h[1:],g,mi)],default=0.0)
                        m=margin(g,tx,ty,T,W,H) if v>=1-1e-9 else None
                        pairs.append({'run':name,'image':s,'size':sz,'shot':meta[s]['shot_type'],'v':v,'margin':m,'score':sc})
                        if m is not None: full.append((m,sc))
                    best=max([h[0] for h in hc if metrics.hit(h[1:],g,mi)],default=-1)
                    bt.append({'run':name,'image':s,'size':sz,'shot':meta[s]['shot_type'],'n_full':len(full),'best_margin':max(m for m,_ in full) if full else None,
                               'score':best,'found':best>=thr,'tile_scores':sorted(full)})
        print(f"{recipe}: merged per-tile cache reproduces the harness cache in {check[1]} of {check[0]} photos")
        q=lambda x:[round(float(v),3) for v in np.percentile(x,[25,50,75])] if len(x) else None
        def row(p,key):
            k=sum(x['score']>=thr for x in p); return {key:None,'pairs':len(p),'butts':len({(x['image'],x['size']) for x in p}),'fired':k,'fired_share':round(k/len(p),3) if p else None,
                                                     'ci95':metrics.wilson(k,len(p)),'score_quartiles':q([x['score'] for x in p])}
        by_v=[]
        for i in range(len(VBINS)-1):
            p=[x for x in pairs if VBINS[i]<=x['v']<VBINS[i+1] and (x['v']<1-1e-9 or i<3)]
            by_v.append(row(p,'v')|{'v':label(VBINS[i],round(min(VBINS[i+1],1),2))})
        whole=[x for x in pairs if x['margin'] is not None]; by_v.append(row(whole,'v')|{'v':'1 (wholly inside)'})
        by_m=[]
        for lo,hi in zip(MBINS,MBINS[1:]):
            p=[x for x in whole if lo<=x['margin']<hi]; r=row(p,'margin')|{'margin':label(lo,hi,' px')}
            r['size_median']=float(np.median([x['size'] for x in p])) if p else None
            r['by_size']={b:row([x for x in p if (x['size']<60)==(b=='<60 px')],'s') for b in ('<60 px','>=60 px')}; by_m.append(r)
        # within-butt paired: nearest-inner-edge tile minus farthest, butts wholly inside >= 2 tiles
        pb=[b for b in bt if b['n_full']>=2]; d=np.array([b['tile_scores'][0][1]-b['tile_scores'][-1][1] for b in pb])
        md=[b['tile_scores'][-1][0]-b['tile_scores'][0][0] for b in pb]
        near=[(b,dd) for b,dd in zip(pb,d) if b['tile_scores'][0][0]<48 and b['tile_scores'][-1][0]>=96]
        imgs=sorted({b['image'] for b in bt}); rng=np.random.default_rng(0)
        def boot(items):
            byi=collections.defaultdict(list)
            for b,dd in items: byi[b['image']].append(dd)
            ks=list(byi); v=[]
            for _ in range(1000):
                x=[dd for i in rng.integers(0,len(ks),len(ks)) for dd in byi[ks[i]]]; v.append(np.mean(x) if x else 0)
            return [round(float(np.percentile(v,2.5)),3),round(float(np.percentile(v,97.5)),3)]
        def paired(items):
            dd=np.array([x for _,x in items]); fl=[(b['tile_scores'][0][1]>=thr)-(b['tile_scores'][-1][1]>=thr) for b,_ in items]
            return {'butts':len(items),'mean_diff':round(float(dd.mean()),3) if len(dd) else None,'ci95':boot(items) if len(dd) else None,
                    'median_diff':round(float(np.median(dd)),3) if len(dd) else None,'near_lower':int((dd<-0.05).sum()),'near_higher':int((dd>0.05).sum()),
                    'fired_far_not_near':fl.count(-1),'fired_near_not_far':fl.count(1)}
        pr={'all':paired(list(zip(pb,d)))|{'margin_gap_median':float(np.median(md)) if md else None},'near_lt48_far_ge96':paired(near),'small_lt60':paired([(b,dd) for b,dd in zip(pb,d) if b['size']<60]),
            'near_lt16_small_lt60':paired([(b,dd) for b,dd in zip(pb,d) if b['size']<60 and b['tile_scores'][0][0]<16])}
        def rec(p,key,val):
            k=sum(b['found'] for b in p); return {key:val,'butts':len(p),'found':k,'recall':round(k/len(p),3) if p else None,'ci95':metrics.wilson(k,len(p)),
                                                 'size_median':float(np.median([b['size'] for b in p])) if p else None}
        r_m=[rec([b for b in bt if lo<=b['best_margin']<hi],'best_margin',label(lo,hi,' px')) for lo,hi in zip(MBINS,MBINS[1:])]
        r_m_small=[rec([b for b in bt if lo<=b['best_margin']<hi and b['size']<60],'best_margin',label(lo,hi,' px')) for lo,hi in zip(MBINS,MBINS[1:])]
        r_m_wide=[rec([b for b in bt if lo<=b['best_margin']<hi and b['shot']=='wide'],'best_margin',label(lo,hi,' px')) for lo,hi in zip(MBINS,MBINS[1:])]
        r_n=[rec([b for b in bt if b['n_full']==n],'n_full',n) for n in sorted({b['n_full'] for b in bt})]
        r_fold=[rec([b for b in bt if b['run']==n and b['best_margin']<48],'run',n)|{'all':rec([b for b in bt if b['run']==n],'run',n)['butts']} for n in sorted({b['run'] for b in bt})]
        misses=[b for b in bt if not b['found']]
        res={'recipe':recipe,'threshold':thr,'merge_check':{'photos':check[0],'reproduced':check[1]},'butts':len(bt),'found':sum(b['found'] for b in bt),
             'pairs_by_share':by_v,'whole_by_margin':by_m,'paired_within_butt':pr,'recall_by_best_margin':r_m,'recall_by_best_margin_small':r_m_small,
             'recall_by_best_margin_wide':r_m_wide,'recall_by_n_full':r_n,'near_edge_by_fold':r_fold,
             'misses_best_margin_lt48':sum(b['best_margin']<48 for b in misses),'misses':len(misses),'all_best_margin_lt48':sum(b['best_margin']<48 for b in bt)}
        json.dump(res,open(OUT/f'partial_butts_{recipe}.json','w'),indent=1)
        print(json.dumps({k:v for k,v in res.items()},indent=None)[:200])
        for k in ('pairs_by_share','whole_by_margin','recall_by_best_margin','recall_by_best_margin_small','recall_by_best_margin_wide','recall_by_n_full','near_edge_by_fold'):
            print(k); [print(' ',{kk:vv for kk,vv in x.items() if kk!='by_size'}) for x in res[k]]
        print('paired',pr); print('misses with best margin < 48 px:',res['misses_best_margin_lt48'],'of',res['misses'],'; all butts:',res['all_best_margin_lt48'],'of',len(bt))
