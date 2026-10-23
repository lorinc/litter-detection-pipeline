"""Task 5, detection pass: run OWLv2 on whole frames shrunk by each downscale factor and cache the raw boxes.
usage: downscale_detect.py [--config PATH] [--splits val,train]   resumable; skips cached (factor, image) pairs
Each frame is shrunk by 1/f (downscale.resample, 'box' = area average, the closest stand-in for a lower-resolution
sensor), tiled with detector.tile / detector.stride, and every box >= detector.raw_threshold after NMS is saved in
native px to <data_dir>/downscale/f<factor>/<image>.json. Thresholds are applied later by downscale_report.py.
"""
import json,sys,time
from PIL import Image
from common import load_config,load_split
import owl
cfg=load_config(); data=cfg['_data']; D=cfg['downscale']
splits=sys.argv[sys.argv.index('--splits')+1].split(',') if '--splits' in sys.argv else D['splits']
split=load_split(cfg); imgs=sorted(s for s,v in split.items() if v in splits)
RS={'box':Image.BOX,'bilinear':Image.BILINEAR,'lanczos':Image.LANCZOS}[D['resample']]
for f in D['factors']:
    out=data/'downscale'/f'f{f}'; out.mkdir(parents=True,exist_ok=True)
    for s in imgs:
        o=out/f'{s}.json'
        if o.exists(): continue
        t0=time.time(); im=Image.open(data/'images'/f'{s}.png').convert('RGB')
        sm=im if f==1 else im.resize((round(im.width/f),round(im.height/f)),RS)
        hits=[(sc,a*im.width/sm.width,b*im.height/sm.height,c*im.width/sm.width,e*im.height/sm.height) for sc,a,b,c,e in owl.detect(sm,cfg)]
        json.dump({'image':s,'factor':f,'size':[sm.width,sm.height],'hits':[[round(h[0],4)]+[round(v,1) for v in h[1:]] for h in hits],
                   'settings':{'detector':cfg['detector'],'resample':D['resample']}},open(o,'w'))
        print(f'f{f} {s}: {len(hits)} boxes, {time.time()-t0:.0f} s',flush=True)
