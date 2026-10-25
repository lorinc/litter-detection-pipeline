"""Error listing of a cross-validation recipe: every false positive and every missed butt, pooled over its folds.
usage: error_analysis.py RECIPE [--threshold T] [--sheets]
Folds, held-out frames and cached detections as in cv_report.py; matching as the harness (metrics.hit: centre inside
the box or IoU >= eval.match_iou; a detection on a maybe box is neither a hit nor a false positive). T defaults to the
runs' operating threshold. Each row: kind (fp|miss), photo, fold, box (x0 y0 x1 y1, native px), score (fp: the
detection's; miss: the best score of a hitting detection below T, empty when none above raw_threshold), long side,
shot type, light, surface, and the box's place in the detector's tile grid (tiling.starts, fit=True, tile and stride
from the cache): tiles_full (grid tiles holding the box entirely), max_visible (largest share of its area inside one
tile), margin (in the tile that holds it best, px from the box to the nearest tile edge; negative when cut).
fp rows also give on_butt / on_maybe (the box overlaps a labelled butt / maybe box without matching it) and
tile_edge (a box side lies within 2 px of a tile edge that is not the frame border: the box was cut by its tile).
Writes output/complete/eval/errors_RECIPE.csv and errors_RECIPE.json: counts (tile_edge also for detections on
butts, as a control), recall of all butts by size, best-tile margin, shot type and light (misses against found
butts), and counts by cause when output/complete/eval/errors_RECIPE_causes.csv (idx, cause, note: a classification
by eye) exists. --sheets: contact sheets of crops with context (box outlined, numbered as the csv's idx, 'edge' when
tile_edge) to output/complete/fig/errors_RECIPE_{fp,miss}_N.jpg.
"""
import json,sys,csv
from PIL import Image,ImageDraw
from common import HERE,load_config
import metrics,tiling
recipe=sys.argv[1]; data=None; rows=[]
thr=float(sys.argv[sys.argv.index('--threshold')+1]) if '--threshold' in sys.argv else None
def grid(b,W,H,T,S):
    best=None; full=0
    for ty in tiling.starts(H,T,S,True):
        for tx in tiling.starts(W,T,S,True):
            a,c=max(b[0],tx),min(b[2],tx+T); e,g=max(b[1],ty),min(b[3],ty+T)
            vis=max(0,c-a)*max(0,g-e)/((b[2]-b[0])*(b[3]-b[1])); m=min(b[0]-tx,b[1]-ty,tx+T-b[2],ty+T-b[3])
            full+=vis>=1
            if best is None or (vis,m)>best: best=(vis,m)
    return full,round(best[0],3),round(best[1])
def tile_edge(b,W,H,T,S):
    xs={e for t in tiling.starts(W,T,S,True) for e in (t,t+T)}-{0,W}; ys={e for t in tiling.starts(H,T,S,True) for e in (t,t+T)}-{0,H}
    return int(any(abs(b[i]-e)<=2 for i in (0,2) for e in xs) or any(abs(b[i]-e)<=2 for i in (1,3) for e in ys))
butts=[]; tp_edge=[]
k=0
while (HERE/'runs'/f'{recipe}_f{k}'/'config.json').exists():
    cfg=load_config(HERE/'runs'/f'{recipe}_f{k}'/'config.json'); data=cfg['_data']; mi=cfg['eval']['match_iou']
    r=json.load(open(HERE/'runs'/f'{recipe}_f{k}'/f'{recipe}_f{k}_val.json')); t=thr if thr is not None else r['threshold']
    meta={m['image']:m for m in csv.DictReader(open(data/'meta.csv'))}
    for s in r['images']:
        c=json.load(open(data/'eval'/f'{recipe}_f{k}'/'val'/f'{s}.json')); lab=json.load(open(data/'labels'/f'{s}.json'))
        st=c['settings']['yolo']; W,H=c['size']; m=meta[s]
        gt=[((b['x'],b['y'],b['x']+b['w'],b['y']+b['h']),b['cls']) for b in lab['boxes']]
        dets=[(h[0],tuple(h[1:])) for h in c['hits']]
        base={'photo':s,'fold':k,'shot':m['shot_type'],'light':m['light'],'surface':m['surface']}
        def row(kind,b,sc,**x):
            rows.append({'kind':kind,**base,'box':' '.join(str(round(v)) for v in b),'score':'' if sc is None else round(sc,3),
                         'long_side':round(max(b[2]-b[0],b[3]-b[1])),**dict(zip(('tiles_full','max_visible','margin'),grid(b,W,H,st['tile'],st['stride']))),**x})
        for sc,d in dets:
            if sc>=t and any(metrics.hit(d,g,mi) for g,kk in gt if kk=='butt'): tp_edge.append(tile_edge(d,W,H,st['tile'],st['stride']))
            if sc<t or any(metrics.hit(d,g,mi) for g,_ in gt): continue
            ov=lambda cls:int(any(metrics.iou(d,g)>0 for g,kk in gt if kk==cls))
            row('fp',d,sc,on_butt=ov('butt'),on_maybe=ov('maybe'),tile_edge=tile_edge(d,W,H,st['tile'],st['stride']))
        for g,kk in gt:
            if kk!='butt': continue
            sc=max([x for x,d in dets if metrics.hit(d,g,mi)],default=None)
            if sc is None or sc<t: row('miss',g,sc)
            butts.append({'found':sc is not None and sc>=t,'long_side':max(g[2]-g[0],g[3]-g[1]),'margin':grid(g,W,H,st['tile'],st['stride'])[2],
                          'shot':m['shot_type'],'light':m['light']})
    k+=1
assert k,f'no run directories for {recipe}'
for kind in ('fp','miss'):
    for i,r in enumerate(sorted([r for r in rows if r['kind']==kind],key=lambda r:(r['photo'],r['box']))): r['idx']=f"{kind}{i+1}"
rows.sort(key=lambda r:(r['kind'],int(r['idx'][len(r['kind']):])))
cols=['idx','kind','photo','fold','box','score','long_side','shot','light','surface','tiles_full','max_visible','margin','on_butt','on_maybe','tile_edge']
out=HERE/'output/complete/eval'/f'errors_{recipe}.csv'
with open(out,'w',newline='') as fh: w=csv.DictWriter(fh,cols,extrasaction='ignore'); w.writeheader(); w.writerows(rows)
def recall_by(key,edges=None):
    g={}
    for b in butts:
        v=b[key] if edges is None else next(f"{lo}-{hi if hi<1e9 else ''}" for lo,hi in zip(edges,edges[1:]) if lo<=b[key]<hi)
        g.setdefault(v,[0,0]); g[v][0]+=1; g[v][1]+=b['found']
    return {v:{'butts':n,'found':k,'missed':n-k,'recall':round(k/n,3),'ci95':metrics.wilson(k,n)} for v,(n,k) in sorted(g.items(),key=lambda x:(len(x[0]),x[0]))}
fps=[r for r in rows if r['kind']=='fp']; ms=[r for r in rows if r['kind']=='miss']
summ={'recipe':recipe,'threshold':t,'false_positives':len(fps),'misses':len(ms),'butts':len(butts),
      'fp_tile_edge':sum(r['tile_edge'] for r in fps),'tp_detections':len(tp_edge),'tp_tile_edge':sum(tp_edge),'fp_on_butt':sum(r['on_butt'] for r in fps),'fp_on_maybe':sum(r['on_maybe'] for r in fps),
      'miss_no_detection':sum(r['score']=='' for r in ms),'miss_cut_in_every_tile':sum(r['tiles_full']==0 for r in ms),
      'recall_by_size':recall_by('long_side',[0,30,45,60,90,120,1e9]),'recall_by_margin':recall_by('margin',[-1e9,0,32,64,128,1e9]),
      'recall_by_shot':recall_by('shot'),'recall_by_light':recall_by('light')}
cf=out.with_name(f'errors_{recipe}_causes.csv')       # hand classification by cause (idx, cause, note), when it exists
if cf.exists():
    cause={r['idx']:r['cause'] for r in csv.DictReader(open(cf))}; assert set(cause)=={r['idx'] for r in rows},f'{cf} does not match the error list'
    for kind,extra in (('fp','tile_edge'),('miss','no_detection')):
        g={}
        for r in rows:
            if r['kind']!=kind: continue
            e=g.setdefault(cause[r['idx']],{'n':0,extra:0}); e['n']+=1; e[extra]+=r['tile_edge'] if kind=='fp' else r['score']==''
        summ[f'{kind}_by_cause']=dict(sorted(g.items(),key=lambda x:-x[1]['n']))
json.dump(summ,open(out.with_suffix('.json'),'w'),indent=1)
print(f"{recipe} at {t}: {sum(r['kind']=='fp' for r in rows)} false positives, {sum(r['kind']=='miss' for r in rows)} misses -> {out}")
if '--sheets' in sys.argv:
    C,R,P=6,4,240                       # columns, rows per sheet, crop px
    for kind in ('fp','miss'):
        rs=[r for r in rows if r['kind']==kind]; ims={}
        for n in range(0,len(rs),C*R):
            sheet=Image.new('RGB',(C*P,R*(P+18)),'white'); dr=ImageDraw.Draw(sheet)
            for j,r in enumerate(rs[n:n+C*R]):
                if r['photo'] not in ims: ims={r['photo']:Image.open(data/'images'/f"{r['photo']}.png").convert('RGB')}
                im=ims[r['photo']]; b=[float(v) for v in r['box'].split()]; side=max(5*r['long_side'],256)
                cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2; x0,y0=cx-side/2,cy-side/2
                cr=im.crop((round(x0),round(y0),round(x0+side),round(y0+side))).resize((P,P),Image.LANCZOS); f=P/side
                ImageDraw.Draw(cr).rectangle([(b[0]-x0)*f-3,(b[1]-y0)*f-3,(b[2]-x0)*f+3,(b[3]-y0)*f+3],outline=(255,0,255),width=1)
                X,Y=j%C*P,j//C*(P+18); sheet.paste(cr,(X,Y))
                dr.text((X+3,Y+P+3),f"{r['idx']} {r['photo'][9:]} s{r['score'] or '-'} {r['long_side']}px{' edge' if r.get('tile_edge') else ''}",fill='black')
            sheet.save(HERE/'output/complete/fig'/f'errors_{recipe}_{kind}_{n//(C*R)+1}.jpg',quality=80)
