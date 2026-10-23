"""Fit the box filter (logistic regression on DINOv2-small crop features) on train-split frames only; save <data_dir>/box_filter.npz.
usage: train_filter.py [--config PATH]
Training labels: initiative 04's circle-derived box labels (box_filter.train_labels / train_boxes), restricted to
images that split.json puts in train, so no val or test frame shapes the filter. In-sample; used only to sort proposals.
"""
import json,numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from common import HERE,load_config,load_split
import owl
cfg=load_config(); data=cfg['_data']; f=cfg['box_filter']
split=load_split(cfg); X,y,used=[],[],[]
for lf in sorted((HERE/f['train_labels']).glob('*.json')):
    lab=json.load(open(lf)); stem=lf.stem
    if lab.get('weak') or split.get(stem)!='train': continue
    hits=json.load(open(HERE/f['train_boxes']/f'{stem}_boxes.json')); pos=set(lab['butts'])
    im=Image.open(data/'images'/f'{stem}.png').convert('RGB')
    X.extend(owl.embed(im,[h[1:] for h in hits],cfg)); y.extend(int(i in pos) for i in range(1,len(hits)+1)); used.append(stem)
X,y=np.array(X),np.array(y)
m=LogisticRegression(C=f['C'],class_weight='balanced',max_iter=2000).fit(X,y)
np.savez(data/'box_filter.npz',coef=m.coef_[0],intercept=m.intercept_,frames=np.array(used))
print(f'{len(used)} train frames, {len(y)} boxes, {y.sum()} butt boxes -> {data/"box_filter.npz"}')
