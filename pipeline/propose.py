"""Proposal cache for the annotation tool: OWLv2 + box filter on train images only (blind rule: never val or test).
usage: propose.py [--config PATH]   resumable: images with an existing <data_dir>/proposals/<image>.json are skipped
Writes {image, w, h, boxes:[{x,y,w,h,det,filter}], settings} with boxes >= detector.keep_threshold after NMS, in native px.
The filter score only sorts and colours proposals; nothing is dropped by it.
"""
import json,time,numpy as np
from PIL import Image
from common import load_config,load_split
import owl
cfg=load_config(); data=cfg['_data']; out=data/'proposals'; out.mkdir(exist_ok=True)
split=load_split(cfg); bf=np.load(data/'box_filter.npz')
todo=[s for s,v in sorted(split.items()) if v=='train' and not (out/f'{s}.json').exists()]
print(len(todo),'train images to process',flush=True)
for k,s in enumerate(todo,1):
    t0=time.time(); im=Image.open(data/'images'/f'{s}.png').convert('RGB')
    hits=[h for h in owl.detect(im,cfg) if h[0]>=cfg['detector']['keep_threshold']]
    p=1/(1+np.exp(-(owl.embed(im,[h[1:] for h in hits],cfg)@bf['coef']+bf['intercept'][0]))) if hits else []
    boxes=[{'x':round(a,1),'y':round(b,1),'w':round(c-a,1),'h':round(e-b,1),'det':round(sc,4),'filter':round(float(q),4)}
           for (sc,a,b,c,e),q in zip(hits,p)]
    rec={'image':s,'w':im.width,'h':im.height,'boxes':boxes,'settings':{'detector':cfg['detector'],'box_filter_frames':bf['frames'].tolist()}}
    tmp=out/f'{s}.json.tmp'; json.dump(rec,open(tmp,'w')); tmp.rename(out/f'{s}.json')
    print(f'{k}/{len(todo)} {s}: {len(boxes)} boxes, {time.time()-t0:.0f} s',flush=True)
