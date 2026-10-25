"""Pooled cross-validation curves of several recipes in one chart: recall against false positives per frame, from
runs/cv_RECIPE.json (cv_report.py), with each recipe's operating point at its threshold marked.
usage: cv_plot.py OUT.png RECIPE [RECIPE ...]     OUT.png is relative to output/complete/fig/
"""
import json,sys
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import HERE
COLORS=['#2a78d6','#eb6834','#1baf7a','#eda100']     # categorical slots 1-4, fixed order
out,recipes=sys.argv[1],sys.argv[2:]
fig,ax=plt.subplots(figsize=(6.4,4.2))
for r,c in zip(recipes,COLORS):
    d=json.load(open(HERE/'runs'/f'cv_{r}.json')); op=d['operating_point']
    ax.plot([p['fp_per_frame'] for p in d['curve']],[p['recall'] for p in d['curve']],color=c,lw=2,label=r)
    ax.plot(op['fp_per_frame'],op['recall'],'o',ms=8,color=c,mec='white',mew=2)
ax.set_xlim(0,4); ax.set_ylim(0.6,1.0); ax.grid(alpha=0.25,lw=0.6)
for s in ('top','right'): ax.spines[s].set_visible(False)
ax.set_xlabel('false positives per photo'); ax.set_ylabel('recall')
ax.set_title(f'Cross-validation, pooled over {op["frames"]} photos ({op["butts"]} butts); dots at threshold {d["threshold"]}',fontsize=9)
ax.legend(frameon=False,fontsize=8,loc='lower right')
fig.tight_layout(); fig.savefig(HERE/'output/complete/fig'/out,dpi=130)
