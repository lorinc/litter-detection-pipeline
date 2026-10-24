"""Export tiles and boxes for training (task 6). YOLO format only.
usage: export.py [--config PATH] [--data DIR] [--out DIR]
--data: a directory laid out like <data_dir> (images/<stem>.png, labels/<stem>.json, split.json), e.g. the Kaggle
dataset mount; default <data_dir>. --out: default <data_dir>/<export.out_dir>; it is wiped and rewritten.
Settings (config 'export'):
  downscale    frames are shrunk by 1/downscale (area averaging) before tiling; 1 = native resolution
  tile, stride square tiles at this size and spacing (tiling.starts with fit=True: every tile is a full square)
  min_visible  a box is a target in a tile when at least this share of its area is inside; it is clipped to the
               tile. A smaller share leaves it unlabelled (background)
  maybe        'background' (unlabelled), 'butt' (target like a butt), 'mask' (the visible part painted with
               mask_fill), 'drop_tile' (tiles where a maybe counts by min_visible are left out)
  background_share  share of train tiles that hold no target: that many empty train tiles are drawn at random
               (seed) from the whole train grid; null keeps every empty tile. Val and test keep the full grid
  hard_negatives  true adds, for train photos only, tiles centred on the label file's rejected proposals (OWLv2
               boxes the annotator rejected; covered ones excluded via labels_util.covered): proposals by descending
               detector score, each not yet fully inside a hard-negative tile gets one, shifted inward at the frame
               border. Their boxes are labelled like grid tiles; they come on top of background_share
  workers      parallel processes, one frame each (0 = all cores); png_compress  zlib level for tiles (lossless)
Tiles are planned from the labels first (no pixels read), empty train tiles are drawn, then frames are cut and
written in parallel. Writes images/<split>/<stem>_x<tx>_y<ty>.png, labels/<split>/<same>.txt (class 0 = butt;
cx cy w h, normalised), data.yaml for Ultralytics, and export.json (settings, counts, each tile's source frame and
offset in native px).
"""
import json,sys,os,shutil,random,pathlib,multiprocessing
from PIL import Image,ImageDraw
from common import load_config,load_split
import tiling,labels_util
cfg=load_config(); X=cfg['export']
if '--data' in sys.argv: cfg['_data']=pathlib.Path(sys.argv[sys.argv.index('--data')+1])
data=cfg['_data']; out=pathlib.Path(sys.argv[sys.argv.index('--out')+1]) if '--out' in sys.argv else data/X['out_dir']
assert X['maybe'] in ('background','butt','mask','drop_tile'),X['maybe']
f=X['downscale']; T=X['tile']; S=X['stride']
def plan(s,sp):
    """Tiles of one frame from its labels: [{tile, split, image, x, y (native px), targets, masks, dropped, hard}]."""
    lab=json.load(open(data/'labels'/f'{s}.json')); W,H=round(lab['w']/f),round(lab['h']/f)
    boxes=[(b['x']/f,b['y']/f,(b['x']+b['w'])/f,(b['y']+b['h'])/f,b['cls']) for b in lab['boxes']]
    def tile(tx,ty,hard):
        tw,th=min(T,W-tx),min(T,H-ty); tgt=[]; masks=[]; drop=False
        for a,b,c,e,cls in boxes:
            ca,cb,cc,ce=max(a,tx),max(b,ty),min(c,tx+tw),min(e,ty+th)
            if cc<=ca or ce<=cb: continue
            if cls=='maybe' and X['maybe']=='mask': masks.append((ca-tx,cb-ty,cc-tx,ce-ty)); continue
            if (cc-ca)*(ce-cb)/((c-a)*(e-b))<X['min_visible']: continue
            if cls=='butt' or X['maybe']=='butt': tgt.append((ca-tx,cb-ty,cc-tx,ce-ty))
            elif X['maybe']=='drop_tile': drop=True
        return {'tile':f"{s}{'_hn' if hard else ''}_x{round(tx*f)}_y{round(ty*f)}",'split':sp,'image':s,'x':round(tx*f),'y':round(ty*f),
                'px':(tx,ty,tw,th),'targets':tgt,'masks':masks,'dropped':drop,'hard':hard}
    tiles=[tile(tx,ty,False) for ty in tiling.starts(H,T,S,True) for tx in tiling.starts(W,T,S,True)]
    if X.get('hard_negatives') and sp=='train':
        props=sorted((p for p in lab.get('rejected_proposals',[]) if not labels_util.covered(p,lab['boxes'])),key=lambda p:-p['det']); done=[]
        for p in props:
            a,b,c,e=p['x']/f,p['y']/f,(p['x']+p['w'])/f,(p['y']+p['h'])/f
            if any(tx<=a and ty<=b and c<=tx+T and e<=ty+T for tx,ty in done): continue
            tx=round(min(max((a+c)/2-T/2,0),max(W-T,0))); ty=round(min(max((b+e)/2-T/2,0),max(H-T,0)))
            done.append((tx,ty)); tiles.append(tile(tx,ty,True))
    return tiles
def write(s,tiles):
    """Cut and write the kept tiles of one frame."""
    im=Image.open(data/'images'/f'{s}.png').convert('RGB')
    if f!=1: im=im.resize((round(im.width/f),round(im.height/f)),Image.BOX)
    for t in tiles:
        tx,ty,tw,th=t['px']; tile=im.crop((tx,ty,tx+tw,ty+th))
        if t['masks']:
            dr=ImageDraw.Draw(tile)
            for m in t['masks']: dr.rectangle(m,fill=tuple(X['mask_fill']))
        tile.save(out/'images'/t['split']/f"{t['tile']}.png",compress_level=X['png_compress'])
        open(out/'labels'/t['split']/f"{t['tile']}.txt",'w').write(''.join(f"0 {(a+c)/2/tw:.6f} {(b+e)/2/th:.6f} {(c-a)/tw:.6f} {(e-b)/th:.6f}\n" for a,b,c,e in t['targets']))
if __name__=='__main__':
    split=load_split(cfg); tiles=[t for s in sorted(split) for t in plan(s,split[s])]
    kept=[t for t in tiles if not t['dropped']]
    empty=[t for t in kept if t['split']=='train' and not t['targets'] and not t['hard']]
    if X['background_share'] is not None:
        full=sum(1 for t in kept if t['split']=='train' and t['targets'] and not t['hard'])
        n=min(len(empty),round(X['background_share']/(1-X['background_share'])*full))
        drop={t['tile'] for t in empty}-{t['tile'] for t in random.Random(X['seed']).sample(empty,n)}
        kept=[t for t in kept if t['tile'] not in drop]
    shutil.rmtree(out,ignore_errors=True)
    for sp in set(split.values()):
        for d in ('images','labels'): (out/d/sp).mkdir(parents=True,exist_ok=True)
    by={}
    for t in kept: by.setdefault(t['image'],[]).append(t)
    with multiprocessing.Pool(X['workers'] or os.cpu_count()) as pool: pool.starmap(write,sorted(by.items()),chunksize=1)
    counts={}
    for sp in sorted(set(split.values())):
        k=[t for t in kept if t['split']==sp]
        counts[sp]={'frames':sum(1 for v in split.values() if v==sp),'tiles':len(k),'with_target':sum(1 for t in k if t['targets']),
                    'targets':sum(len(t['targets']) for t in k),'empty':sum(1 for t in k if not t['targets']),
                    'empty_dropped':sum(1 for t in tiles if t['split']==sp and not t['dropped'] and not t['targets'] and not t['hard'])-sum(1 for t in k if not t['targets'] and not t['hard']),
                    'maybe_dropped':sum(1 for t in tiles if t['split']==sp and t['dropped']),
                    'hard_negative':sum(1 for t in k if t['hard']),'hard_negative_with_target':sum(1 for t in k if t['hard'] and t['targets'])}
    open(out/'data.yaml','w').write(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: butt\n")
    index=[{k:t[k] for k in ('tile','split','image','x','y','hard')}|{'size':round(T*f),'targets':len(t['targets'])} for t in kept]
    json.dump({'settings':X,'counts':counts,'tiles':index},open(out/'export.json','w'),indent=1)
    for sp,c in counts.items(): print(f"{sp}: {c}")
