"""Perspective check: are detector boxes the right pixel size for their image row, given a ground plane?
usage: perspective_check.py [TOL=1.5]   (reads initiative 04's work/*.hits.json + circles.json; writes output/complete/fig/perspective_check.png)
Camera model: pinhole, f = 2889 px (25 mm equivalent on the 4000x3000 sensor frame), principal point at the frame
centre, camera height H above flat ground, pitch theta below horizontal, no roll. A butt of length L at depth Z along the
optical axis spans f*L/Z px lying sideways and about f*L*sin(alpha)/Z pointing at the camera (alpha = ray depression).
Expected box long side: from f*max(D, Lmin*sin(alpha))/Z to f*Lmax/Z. Boxes are labelled from the operator's circles
(centre inside circle padded 20 px), boxes >= 0.10 only. Pitch is fitted, so in-sample.
"""
import json,math,pathlib,sys,numpy as np
F=2889; H=0.49; LMIN,LMAX,D=0.015,0.040,0.007; TOL=float(sys.argv[1]) if len(sys.argv)>1 else 1.5; PAD=20; THR=0.10
root=pathlib.Path(__file__).resolve().parents[2]/'04-detection-demo-v0'
circ=json.load(open(root/'images/local_photos/circles.json'))
def band(v,th):  # v = row offset from centre (px, down positive); returns (lo,hi) px or None above horizon
    a=th+math.atan(v/F)
    if a<=0.01: return None
    R=H/math.sin(a); Z=R*math.cos(a-th)
    return F*max(D,LMIN*math.sin(a))/Z, F*LMAX/Z
def mid(v,th):
    b=band(v,th); return None if b is None else math.sqrt(b[0]*b[1])
imgs=[]
for hf in sorted((root/'images/local_photos/work').glob('*.hits.json')):
    s=hf.name[:-10]
    if s not in circ: continue
    W,Hh=circ[s]['size']; cs=circ[s]['circles']; boxes=[]
    for sc,a,b,c,e in json.load(open(hf)):
        if sc<THR: continue
        cx,cy=(a+c)/2,(b+e)/2
        pos=any(x-PAD<=cx<=x+w+PAD and y-PAD<=cy<=y+h+PAD for x,y,w,h in cs)
        boxes.append(dict(pos=pos,v=cy-Hh/2,L=max(c-a,e-b),s=sc))
    imgs.append(dict(stem=s,boxes=boxes))
ths=np.radians(np.arange(1,85,0.5))
def cost(bs,th):
    r=[math.log(b['L']/m) for b in bs if (m:=mid(b['v'],th))]
    return sum(x*x for x in r)+10*sum(1 for b in bs if mid(b['v'],th) is None)
def classify(b,th):
    bd=band(b['v'],th)
    if bd is None: return 'above horizon'
    if b['L']<bd[0]/TOL: return 'too small'
    if b['L']>bd[1]*TOL: return 'too big'
    return 'fits'
def run(name,pick):
    out={True:{},False:{}}
    for im in imgs:
        th=pick(im)
        for b in im['boxes']: k=classify(b,th); out[b['pos']][k]=out[b['pos']].get(k,0)+1
    print(name)
    for p in (True,False): print('  ','butts' if p else 'non-butts',sum(out[p].values()),out[p])
    return out
# model A: free pitch per image, fitted on that image's butts
fitA={im['stem']:min(ths,key=lambda t:cost([b for b in im['boxes'] if b['pos']],t)) for im in imgs}
run('A: pitch fitted per image from its own butts',lambda im:fitA[im['stem']])
# model B: one session pitch theta0, each image either theta0 (leg upright) or theta0+15 deg (leaning)
best=None
for t0 in ths:
    c=sum(min(cost([b for b in im['boxes'] if b['pos']],t0+d) for d in (0,math.radians(15))) for im in imgs)
    if best is None or c<best[0]: best=(c,t0)
t0=best[1]
pickB=lambda im:min((t0,t0+math.radians(15)),key=lambda t:cost([b for b in im['boxes'] if b['pos']],t))
print(f'B: session pitch {math.degrees(t0):.1f} deg or +15 deg; per-image choice:',
      sum(1 for im in imgs if pickB(im)>t0+0.01),'of',len(imgs),'leaning')
run('B: two-state pitch',pickB)
print('A pitches (deg):',sorted(round(math.degrees(t)) for t in fitA.values()))
# plot: observed long side / expected mid (model B) against row, butts vs non-butts
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(7,4.5))
for p,col,lab in ((False,'#b0b0b0','non-butt boxes'),(True,'#d2691e','marked butts')):
    xs,ys=[],[]
    for im in imgs:
        th=pickB(im)
        for b in im['boxes']:
            m=mid(b['v'],th)
            if b['pos']==p and m: xs.append(math.degrees(th+math.atan(b['v']/F))); ys.append(b['L']/m)
    ax.scatter(xs,ys,s=18,c=col,label=lab,edgecolors='none')
ax.set_yscale('log'); ax.axhline(1,color='k',lw=.5); ax.set_xlabel('ray angle below horizontal (deg) — larger = nearer the camera')
ax.set_ylabel('box long side / expected size'); ax.legend(frameon=False)
ax.set_title('Box size against ground-plane expectation (two-state pitch, in-sample)',fontsize=10)
fig.tight_layout(); fig.savefig(pathlib.Path(__file__).resolve().parents[1]/'output/complete/fig/perspective_check.png',dpi=130)
