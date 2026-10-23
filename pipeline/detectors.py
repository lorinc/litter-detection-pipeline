"""Detector adapters for the evaluation harness. Each adapter takes the config and its own eval.detectors entry and
exposes:
  warmup()            load weights, so loading is not timed as part of a frame
  detect(im)          -> [(score, x0, y0, x1, y1)] in full-frame px, after tiling, mapping back and merging;
                         every box the detector keeps, down to its own raw threshold (thresholds are applied later)
  info(w, h)          -> {'params', 'flops_per_frame', ...} for a w x h frame; FLOPs as counted by
                         torch.utils.flop_counter (2 per multiply-add), a hardware-independent speed measure
  settings            dict written into the result so the run is reproducible
ADAPTERS maps the 'adapter' name used in config.json to the class.
"""
import torch
from PIL import Image
from torch.utils.flop_counter import FlopCounterMode
import tiling
class OWLv2:
    """Zero-shot OWLv2 exactly as in proposals and task 5 (owl.detect): config 'detector' block, no box filter."""
    def __init__(self,cfg,entry):
        import owl; self.owl=owl; self.cfg=cfg; self.d=cfg['detector']; self.settings={'detector':self.d}
    def warmup(self): self.owl.detect(Image.new('RGB',(64,64)),self.cfg)
    def detect(self,im): return self.owl.detect(im,self.cfg)
    def info(self,w,h):
        proc,model=self.owl._owl(self.d['model'])
        # every tile is padded and resized to the model's input square, so one tile's count holds for edge tiles too
        inp=proc(text=[self.d['prompts']],images=Image.new('RGB',(self.d['tile'],self.d['tile'])),return_tensors='pt')
        with torch.no_grad(),FlopCounterMode(display=False) as fc: model(**inp)
        n=tiling.tiles(w,h,self.d['tile'],self.d['stride'])
        return {'params':sum(p.numel() for p in model.parameters()),'flops_per_tile':fc.get_total_flops(),
                'tiles_per_frame':n,'flops_per_frame':fc.get_total_flops()*n,
                'note':'FLOPs include the text encoder, which owl.detect re-runs on every tile'}
ADAPTERS={'owlv2':OWLv2}
class YOLO:
    """An Ultralytics detector (trained on export tiles) run on the export's square grid (tiling with fit=True):
    entry keys weights (path, relative to the initiative directory), tile, stride, raw_threshold (per-tile conf
    floor kept for the sweep), nms_iou (merge across tiles), imgsz, threads. Within a tile Ultralytics' own NMS runs."""
    def __init__(self,cfg,entry):
        from ultralytics import YOLO as Y
        from common import HERE
        self.e=entry; self.model=Y(str(HERE/entry['weights'])); self.settings={'yolo':entry}
    def _tile(self,t):
        r=self.model.predict(t,imgsz=self.e['imgsz'],conf=self.e['raw_threshold'],verbose=False)[0].boxes
        return [(float(s),*b) for s,b in zip(r.conf.tolist(),r.xyxy.tolist())]
    def warmup(self): torch.set_num_threads(self.e['threads']); self._tile(Image.new('RGB',(self.e['tile'],self.e['tile'])))
    def detect(self,im): return tiling.tiled(im,self.e['tile'],self.e['stride'],self.e['nms_iou'],self._tile,fit=True)
    def info(self,w,h):
        net=self.model.model; x=torch.zeros(1,3,self.e['imgsz'],self.e['imgsz'])
        with torch.no_grad(),FlopCounterMode(display=False) as fc: net(x)
        n=tiling.tiles(w,h,self.e['tile'],self.e['stride'],True)
        return {'params':sum(p.numel() for p in net.parameters()),'flops_per_tile':fc.get_total_flops(),
                'tiles_per_frame':n,'flops_per_frame':fc.get_total_flops()*n}
ADAPTERS['yolo']=YOLO
