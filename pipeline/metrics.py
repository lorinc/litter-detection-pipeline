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
