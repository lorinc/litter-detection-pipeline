"""Shared tiling for every detector: cut a frame into overlapping square tiles, run a per-tile detector, map boxes
back to full-frame px, merge across tiles with NMS.
tiled(im, tile, stride, nms_iou, fn, fit=False) -> [(score, x0, y0, x1, y1)] sorted by score; fn(tile_image) returns
the same tuples in tile px. fit=False (OWLv2): edge tiles are cropped at the frame border, not padded. fit=True (the
export and trained models): the last tile in each row and column is shifted inward to end at the border, so every
tile is a full square. A frame needs tiles(w, h, tile, stride, fit) calls.
"""
import torch,torchvision
def starts(n,tile,stride,fit=False):
    if not fit or n<=tile: return list(range(0,max(1,n-tile+stride),stride))
    s=list(range(0,n-tile+1,stride)); return s+[n-tile] if s[-1]+tile<n else s
def tiles(w,h,tile,stride,fit=False): return len(starts(w,tile,stride,fit))*len(starts(h,tile,stride,fit))
def tiled(im,tile,stride,nms_iou,fn,fit=False):
    hits=[]
    for ty in starts(im.height,tile,stride,fit):
        for tx in starts(im.width,tile,stride,fit):
            for s,a,b,c,e in fn(im.crop((tx,ty,min(tx+tile,im.width),min(ty+tile,im.height)))):
                hits.append((s,a+tx,b+ty,c+tx,e+ty))
    if hits:
        keep=torchvision.ops.nms(torch.tensor([h[1:] for h in hits]),torch.tensor([h[0] for h in hits]),nms_iou).tolist()
        hits=[hits[i] for i in keep]
    return sorted(hits,reverse=True)
