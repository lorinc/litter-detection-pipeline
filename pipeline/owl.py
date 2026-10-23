"""Detector and box-filter building blocks (same method as initiative 04's detect_tiled.py and train_probe.py).
detect(im, cfg)        -> [(score, x0, y0, x1, y1)] in image px: OWLv2 on overlapping square tiles, NMS across tiles.
embed(im, boxes, cfg)  -> DINOv2-small pooled features of a square crop around each box (box filter input).
"""
import numpy as np,torch
import tiling
from transformers import Owlv2Processor,Owlv2ForObjectDetection,AutoImageProcessor,AutoModel
_m={}
def _owl(name):
    if name not in _m: _m[name]=(Owlv2Processor.from_pretrained(name),Owlv2ForObjectDetection.from_pretrained(name).eval())
    return _m[name]
def detect_tile(tile,cfg):
    """OWLv2 on one tile -> [(score, x0, y0, x1, y1)] in tile px, every box >= detector.raw_threshold."""
    d=cfg['detector']; torch.set_num_threads(d['threads']); proc,model=_owl(d['model'])
    with torch.no_grad(): out=model(**proc(text=[d['prompts']],images=tile,return_tensors='pt'))
    side=max(tile.size)  # owlv2 pads to square
    res=proc.post_process_grounded_object_detection(out,threshold=d['raw_threshold'],target_sizes=torch.tensor([[side,side]]))[0]
    return [(float(s),*b.tolist()) for s,b in zip(res['scores'],res['boxes'])]
def detect(im,cfg):
    d=cfg['detector']; return tiling.tiled(im,d['tile'],d['stride'],d['nms_iou'],lambda t:detect_tile(t,cfg))
def embed(im,boxes,cfg):
    f=cfg['box_filter']; key=('emb',f['backbone'])
    if key not in _m: _m[key]=(AutoImageProcessor.from_pretrained(f['backbone']),AutoModel.from_pretrained(f['backbone']).eval())
    proc,net=_m[key]; X=[]
    for a,b,c,e in boxes:
        cx,cy=(a+c)/2,(b+e)/2; half=max(f['crop_min_half'],max(c-a,e-b)*f['crop_scale'])
        crop=im.crop((int(cx-half),int(cy-half),int(cx+half),int(cy+half)))
        with torch.no_grad(): X.append(net(**proc(images=crop,return_tensors='pt')).pooler_output[0].numpy())
    return np.array(X)
