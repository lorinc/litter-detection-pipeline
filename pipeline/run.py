"""One command per batch of runs: publish the pipeline, export and train on Kaggle (one GPU per run, in parallel),
download, evaluate locally.
usage: run.py [--cv] NAME[=CONFIG] [NAME[=CONFIG] ...]     a bare NAME uses pipeline/config.json
--cv: each NAME is a recipe trained once per cross-validation fold (config cv.folds, folds.json from make_folds.py),
as runs NAME_f0 .. NAME_f<folds-1> with cv.fold set; each is scored on its held-out fold, then cv_report.py pools them
(with exactly two recipes it compares the second against the first).
resumable: each step is skipped when its output already exists. The runs of one call share one Kaggle kernel,
butt-train-NAME1-NAME2... (with -cv for --cv), which trains them in rounds of as many runs as the machine has GPUs
(Kaggle's T4 machine has 2), each round exported just before it trains.
Run directory runs/NAME/ (initiative directory): config.json (the config the run used, with its eval entry),
meta.json (private and public commit, label hash, Kaggle kernel, order in the batch), kernel/ (the script pushed to Kaggle),
kaggle/ (downloaded training output: weights, results.csv, plots, train.log, kernel log), NAME_val.json and
eval_NAME_val.png (harness result, scoring last.pt: the final epoch, so no scored photo picks the checkpoint). The kernel clones the public repo at the published commit, installs the pinned
Ultralytics, and per round runs export.py on the Kaggle dataset for each run in turn, then train.py for the round's
runs at once, each seeing only its own GPU, and keeps only the training output.
Needs KAGGLE_API_TOKEN in the environment.
"""
import json,sys,copy,time,shutil,subprocess,hashlib,pathlib,tempfile
from common import HERE,load_config,load_split
import publish
cv='--cv' in sys.argv; runs={}; recipes=[]
for a in sys.argv[1:]:
    if a=='--cv': continue
    n,_,p=a.partition('='); cfg=load_config(pathlib.Path(p).resolve() if p else None); recipes.append(n)
    for k in range(cfg['cv']['folds']) if cv else [None]:
        runs[f'{n}_f{k}' if cv else n]=copy.deepcopy(cfg)
        if cv: runs[f'{n}_f{k}']['cv']|={'fold':k,'assignment':json.load(open(cfg['_data']/'folds.json'))['folds']}
assert runs,'usage: run.py [--cv] NAME[=CONFIG] [NAME[=CONFIG] ...]'
names=list(runs); R=runs[names[0]]['run']; tag='+'.join(recipes)+(' cv' if cv else '')
rds={n:HERE/'runs'/n for n in names}; kid=f"{R['kaggle_user']}/butt-train-{'-'.join(recipes)}{'-cv' if cv else ''}".lower().replace('_','-')
kg=lambda *a,check=True:subprocess.run([str(pathlib.Path(sys.executable).parent/'kaggle'),*a],capture_output=True,text=True,check=check).stdout
def step(msg): print(f'[{tag}] {msg}',flush=True)
pushed=[n for n in names if (rds[n]/'meta.json').exists()]
assert not pushed or len(pushed)==len(names),f'runs {pushed} were pushed before; run them alone or delete their directories'
if not pushed:
    pub=publish.publish(); rcs={}
    for n,cfg in runs.items():
        split=load_split(cfg); lh=hashlib.sha256()
        for s in sorted(split): lh.update(s.encode()+open(cfg['_data']/'labels'/f'{s}.json','rb').read())
        rc={k:v for k,v in cfg.items() if not k.startswith('_')}
        rc.setdefault('eval',{}).setdefault('detectors',{})[n]={**cfg['run']['eval'],'weights':f'runs/{n}/kaggle/train/weights/last.pt',
            'tile':cfg['export']['tile'],'stride':cfg['export']['stride'],'imgsz':cfg['export']['tile']}
        rds[n].mkdir(parents=True,exist_ok=True); json.dump(rc,open(rds[n]/'config.json','w'),indent=2); rcs[n]=(rc,lh.hexdigest())
    models=sorted({rc['train']['model'] for rc,_ in rcs.values()}|{'yolo11n.pt'}); ver={rc['train']['ultralytics'] for rc,_ in rcs.values()}
    assert len(ver)==1,f'runs of one batch need the same Ultralytics version, got {ver}'
    kd=pathlib.Path(tempfile.mkdtemp())/'kernel'; kd.mkdir()
    open(kd/'run.py','w').write(f'''import subprocess,glob,json,os,shutil,sys
sh=lambda c:subprocess.run(c,shell=True,check=True)
runs=json.loads({json.dumps({n:rc for n,(rc,_) in rcs.items()})!r})
gpus=subprocess.run("nvidia-smi -L",shell=True,capture_output=True,text=True).stdout.splitlines()
assert gpus,"no GPU found"
sh("git clone {R['public_repo']} /tmp/code && git -C /tmp/code checkout {pub}")
sh("pip install -q ultralytics=={ver.pop()}")
# fetch shared downloads (weights, the AMP check's yolo11n.pt, the plot font) once, so parallel runs do not race
sh("""cd /tmp/code/pipeline && python -c "from ultralytics import YOLO; from ultralytics.utils.checks import check_font; [YOLO(m) for m in {models}]; check_font('Arial.ttf')" """)
data=os.path.dirname(glob.glob('/kaggle/input/**/split.json',recursive=True)[0])
names=list(runs); codes={{}}
for r0 in range(0,len(names),len(gpus)):
    rnd=names[r0:r0+len(gpus)]; print(f"===== round {{r0//len(gpus)+1}}: {{rnd}}",flush=True)
    for n in rnd:
        open(f'/tmp/{{n}}.json','w').write(json.dumps(runs[n])); os.makedirs(f'/kaggle/working/{{n}}',exist_ok=True)
        sh(f"cd /tmp/code/pipeline && python export.py --config /tmp/{{n}}.json --data {{data}} --out /tmp/export_{{n}}")
        shutil.copy(f'/tmp/export_{{n}}/export.json',f'/kaggle/working/{{n}}/export.json')
    procs={{n:subprocess.Popen(f"cd /tmp/code/pipeline && exec python train.py --config /tmp/{{n}}.json --data /tmp/export_{{n}} --project /kaggle/working/{{n}}",
        shell=True,env={{**os.environ,'CUDA_VISIBLE_DEVICES':str(i)}},stdout=open(f'/kaggle/working/{{n}}/train.log','w'),stderr=subprocess.STDOUT)
        for i,n in enumerate(rnd)}}
    codes|={{n:p.wait() for n,p in procs.items()}}
    for n in rnd: shutil.rmtree(f'/tmp/export_{{n}}',ignore_errors=True)
for n in runs: print(f"===== {{n}}: exit {{codes[n]}}"); print(''.join(open(f'/kaggle/working/{{n}}/train.log').readlines()[-30:]))
sys.exit(1 if any(codes.values()) else 0)
''')
    json.dump({'id':kid,'title':kid.split('/')[1],'code_file':'run.py','language':'python','kernel_type':'script',
               'is_private':True,'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4',
               'dataset_sources':[R['kaggle_dataset']]},open(kd/'kernel-metadata.json','w'),indent=1)
    step('pushing kernel '+kid); print(kg('kernels','push','-p',str(kd)))
    for i,n in enumerate(names):
        shutil.copytree(kd,rds[n]/'kernel',dirs_exist_ok=True)
        json.dump({'run':n,'batch':names,'order':i,'private_commit':publish.git('rev-parse','HEAD'),'public_repo':R['public_repo'],
                   'public_commit':pub,'labels_sha256':rcs[n][1],'kernel':kid,'pushed':time.strftime('%Y-%m-%d %H:%M:%S')},
                  open(rds[n]/'meta.json','w'),indent=1)
    shutil.rmtree(kd.parent)
best=lambda n:rds[n]/'kaggle'/'train'/'weights'/'last.pt'
failed=False
if not all(best(n).exists() for n in names):
    while True:
        st=kg('kernels','status',kid,check=False).strip(); step(st)
        if 'COMPLETE' in st.upper(): break
        if 'ERROR' in st.upper() or 'CANCEL' in st.upper(): failed=True; break
        time.sleep(120)
    step('downloading output'); dl=pathlib.Path(tempfile.mkdtemp()); kg('kernels','output',kid,'-p',str(dl),check=False)
    for n in names:
        kd=rds[n]/'kaggle'; kd.mkdir(exist_ok=True)
        if (dl/n).exists(): shutil.copytree(dl/n,kd,dirs_exist_ok=True)
        for f in dl.glob('*.log'): shutil.copy(f,kd)
    shutil.rmtree(dl)
for n in names:
    if not best(n).exists(): step(f'{n}: no weights; log in {rds[n]/"kaggle"}'); failed=True; continue
    if not (rds[n]/f'{n}_val.json').exists():
        step(f'evaluating {n}'); subprocess.run([sys.executable,str(HERE/'pipeline/evaluate.py'),n,'--config',str(rds[n]/'config.json'),'--out',str(rds[n])],check=True)
    step('done: '+str(rds[n]/f'{n}_val.json'))
if failed: sys.exit(f'kernel {kid} failed or a run has no weights')
if cv:
    for grp in [recipes] if len(recipes)==2 else [[r] for r in recipes]: subprocess.run([sys.executable,str(HERE/'pipeline/cv_report.py'),*grp],check=True)
