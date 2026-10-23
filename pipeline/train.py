"""Train an Ultralytics model on an export (runs on Kaggle; works anywhere with a GPU or, slowly, a CPU).
usage: train.py --data EXPORT_DIR --project DIR [--config PATH]
Settings (config 'train'): model (Ultralytics weights name, e.g. yolo11s.pt), args (passed to model.train as is;
anything not set there is Ultralytics' default, including its augmentation). imgsz is the export tile size.
Writes Ultralytics' run (weights/best.pt, weights/last.pt, results.csv, plots) to DIR/train.
"""
import sys,pathlib
from ultralytics import YOLO
from common import load_config
cfg=load_config(); A=cfg['train']
data=pathlib.Path(sys.argv[sys.argv.index('--data')+1]); project=sys.argv[sys.argv.index('--project')+1]
YOLO(A['model']).train(data=str(data/'data.yaml'),imgsz=cfg['export']['tile'],project=project,name='train',exist_ok=True,**A['args'])
