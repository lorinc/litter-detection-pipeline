"""The project's progress in one chart: the zero-shot baseline and the trained models on the same blind val photos,
recall against false positives per photo from each harness result (evaluate.py / run.py), operating point marked,
speed per photo on this PC's CPU in the legend. All results must carry the same label hash.
usage: progress_plot.py [OUT.png]     OUT.png is relative to output/complete/fig/ (default progress_val.png)
"""
import json,sys
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import HERE
SERIES=[('OWLv2, zero-shot baseline','output/complete/eval/owlv2_val.json','#2a78d6'),
        ('YOLO11s, 04-cv-hard-negatives fold 0','runs/04-cv-hard-negatives_f0/04-cv-hard-negatives_f0_val.json','#eb6834'),
        ('YOLO11n, 06-cv-yolo11n fold 0','runs/06-cv-yolo11n_f0/06-cv-yolo11n_f0_val.json','#eda100')]  # slots 1, 2, 4: 04 and 06 keep their cv_plot colours
out=sys.argv[1] if len(sys.argv)>1 else 'progress_val.png'
ds=[json.load(open(HERE/p)) for _,p,_ in SERIES]
assert len({d['labels_sha256'] for d in ds})==1,'results are scored on different labels'
fig,ax=plt.subplots(figsize=(6.4,4.4))
for (name,_,c),d in zip(SERIES,ds):
    op,s=d['operating_point'],d['speed']['ms_per_frame_mean']/1000
    ax.plot([p['fp_per_frame'] for p in d['curve']],[p['recall'] for p in d['curve']],color=c,lw=2,
            label=f'{name}: {s:.0f} s per photo' if s>=10 else f'{name}: {s:.1f} s per photo')
    ax.plot(op['fp_per_frame'],op['recall'],'o',ms=8,color=c,mec='white',mew=2)
ax.set_xlim(0,8); ax.set_ylim(0.6,1.0); ax.grid(alpha=0.25,lw=0.6)
for s in ('top','right'): ax.spines[s].set_visible(False)
ax.set_xlabel('false positives per photo'); ax.set_ylabel('recall')
ax.set_title(f'{op["frames"]} validation photos ({op["butts"]} butts, labelled blind), one session; no intervals on the curves\n'
             f'dots at the operating threshold (OWLv2 {ds[0]["threshold"]:.2f}, YOLO {ds[1]["threshold"]:.2f}); speed on a laptop CPU (i5-1145G7)',fontsize=8.5)
ax.legend(frameon=False,fontsize=8,loc='lower right')
fig.tight_layout(); fig.savefig(HERE/'output/complete/fig'/out,dpi=130)
