"""Matching and interval helpers shared by task 5 (downscale_report.py) and the evaluation harness (evaluate.py).
Boxes are (x0, y0, x1, y1). A detection hits a labelled box when its centre lies inside the box or IoU >= match_iou.
"""
import math
def iou(a,b):
    ix=max(0,min(a[2],b[2])-max(a[0],b[0])); iy=max(0,min(a[3],b[3])-max(a[1],b[1])); i=ix*iy
    return i/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i) if i else 0
def hit(d,g,match_iou): cx,cy=(d[0]+d[2])/2,(d[1]+d[3])/2; return (g[0]<=cx<=g[2] and g[1]<=cy<=g[3]) or iou(d,g)>=match_iou
def wilson(k,n,z=1.96):
    if n==0: return (None,None)
    p=k/n; den=1+z*z/n; c=(p+z*z/(2*n))/den; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den; return (round(c-h,3),round(c+h,3))
def frame(c,lab,shot,match_iou):
    """One frame's scoring record from a harness cache entry c and its label file: butts [(long side, best score
    of a hitting detection or -1)], dets [(score, kind)] with kind butt|maybe|fp."""
    gt=[((b['x'],b['y'],b['x']+b['w'],b['y']+b['h']),b['cls']) for b in lab['boxes']]
    dets=[(h[0],h[1:]) for h in c['hits']]
    butts=[(max(g[2]-g[0],g[3]-g[1]),max([sc for sc,d in dets if hit(d,g,match_iou)],default=-1)) for g,k in gt if k=='butt']
    kinds=[(sc,'butt' if any(k=='butt' and hit(d,g,match_iou) for g,k in gt) else
                'maybe' if any(hit(d,g,match_iou) for g,k in gt) else 'fp') for sc,d in dets]
    return {'image':c['image'],'shot':shot,'size':c['size'],'ms':c['ms'],'butts':butts,'dets':kinds}
def at(fr,t):
    """Scores of frames fr at threshold t."""
    n=sum(len(f['butts']) for f in fr); k=sum(sc>=t for f in fr for _,sc in f['butts'])
    fp=sum(sc>=t and kd=='fp' for f in fr for sc,kd in f['dets']); tp=sum(sc>=t and kd=='butt' for f in fr for sc,kd in f['dets'])
    nb=sum(sc>=t for f in fr for sc,_ in f['dets'])
    return {'frames':len(fr),'butts':n,'found':k,'recall':round(k/n,3) if n else None,'recall_ci95':wilson(k,n),
            'fp_per_frame':round(fp/len(fr),2),'boxes_per_frame':round(nb/len(fr),2),'precision':round(tp/(tp+fp),3) if tp+fp else None}
