"""One command per run: publish the pipeline, export and train on Kaggle, download, evaluate locally.
usage: run.py NAME [--config PATH]      resumable: each step is skipped when its output already exists
Run directory runs/NAME/ (initiative directory): config.json (the config the run used, with its eval entry),
meta.json (private and public commit, label hash, Kaggle kernel), kernel/ (the script pushed to Kaggle),
kaggle/ (downloaded training output: weights, results.csv, plots, log), NAME_val.json and eval_NAME_val.png
(harness result). The Kaggle kernel clones the public repo at the published commit, installs the pinned
Ultralytics, runs export.py on the Kaggle dataset and train.py, and keeps only the training output.
Needs KAGGLE_API_TOKEN in the environment.
"""
import json,sys,time,shutil,subprocess,hashlib,pathlib
from common import HERE,load_config,load_split
import publish
cfg=load_config(); R=cfg['run']; name=sys.argv[1]
rd=HERE/'runs'/name; rd.mkdir(parents=True,exist_ok=True); kid=f"{R['kaggle_user']}/butt-train-{name}".lower().replace('_','-')
kg=lambda *a,check=True:subprocess.run([str(pathlib.Path(sys.executable).parent/'kaggle'),*a],capture_output=True,text=True,check=check).stdout
def step(msg): print(f'[{name}] {msg}',flush=True)
if not (rd/'meta.json').exists():
    pub=publish.publish(); split=load_split(cfg); lh=hashlib.sha256()
    for s in sorted(split): lh.update(s.encode()+open(cfg['_data']/'labels'/f'{s}.json','rb').read())
    rc={k:v for k,v in cfg.items() if not k.startswith('_')}
    rc.setdefault('eval',{}).setdefault('detectors',{})[name]={**R['eval'],'weights':f'runs/{name}/kaggle/train/weights/best.pt',
        'tile':cfg['export']['tile'],'stride':cfg['export']['stride'],'imgsz':cfg['export']['tile']}
    json.dump(rc,open(rd/'config.json','w'),indent=2)
    kd=rd/'kernel'; kd.mkdir(exist_ok=True)
    open(kd/'run.py','w').write(f'''import subprocess,glob,json,os,shutil
sh=lambda c:subprocess.run(c,shell=True,check=True)
sh("git clone {R['public_repo']} /tmp/code && git -C /tmp/code checkout {pub}")
sh("pip install -q ultralytics=={cfg['train']['ultralytics']}")
data=os.path.dirname(glob.glob('/kaggle/input/**/split.json',recursive=True)[0])
open('/tmp/config.json','w').write({json.dumps(rc)!r})
sh(f"cd /tmp/code/pipeline && python export.py --config /tmp/config.json --data {{data}} --out /tmp/export")
sh("cd /tmp/code/pipeline && python train.py --config /tmp/config.json --data /tmp/export --project /kaggle/working")
shutil.copy('/tmp/export/export.json','/kaggle/working/export.json')
''')
    json.dump({'id':kid,'title':f'butt-train-{name}','code_file':'run.py','language':'python','kernel_type':'script',
               'is_private':True,'enable_gpu':True,'enable_internet':True,'dataset_sources':[R['kaggle_dataset']]},open(kd/'kernel-metadata.json','w'),indent=1)
    step('pushing kernel '+kid); print(kg('kernels','push','-p',str(kd)))
    json.dump({'run':name,'private_commit':publish.git('rev-parse','HEAD'),'public_repo':R['public_repo'],'public_commit':pub,
               'labels_sha256':lh.hexdigest(),'kernel':kid,'pushed':time.strftime('%Y-%m-%d %H:%M:%S')},open(rd/'meta.json','w'),indent=1)
if not (rd/'kaggle'/'train'/'weights'/'best.pt').exists():
    while True:
        st=kg('kernels','status',kid,check=False).strip(); step(st)
        if 'COMPLETE' in st.upper(): break
        if 'ERROR' in st.upper() or 'CANCEL' in st.upper():
            kg('kernels','output',kid,'-p',str(rd/'kaggle'),check=False); sys.exit(f'kernel failed; log in {rd/"kaggle"}')
        time.sleep(120)
    step('downloading output'); kg('kernels','output',kid,'-p',str(rd/'kaggle'))
if not (rd/f'{name}_val.json').exists():
    step('evaluating'); subprocess.run([sys.executable,str(HERE/'pipeline/evaluate.py'),name,'--config',str(rd/'config.json'),'--out',str(rd)],check=True)
step('done: '+str(rd/f'{name}_val.json'))
