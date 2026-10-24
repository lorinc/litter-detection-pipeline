"""Sync this pipeline to the public repository: a snapshot of the git-tracked files under pipeline/ and
annotation_tool/ (the annotation tool's page, labelling policy and example images), no history, plus the
files and folders listed in config run.publish_docs ({public path: private path}, e.g. docs/capture-protocol.md;
a folder copies its git-tracked files).
usage: publish.py [--config PATH]     prints the public commit
Config 'run': public_repo (URL), public_clone (local working copy, relative to the initiative directory or
absolute; cloned if missing). The private tree must have no uncommitted pipeline changes, so every public commit
names the private commit it was cut from. Nothing is pushed when the snapshot is unchanged.
"""
import subprocess,shutil,pathlib,sys
from common import HERE,load_config
R=load_config()['run']
def git(*a,cwd=HERE,check=True): return subprocess.run(['git',*a],cwd=cwd,capture_output=True,text=True,check=check).stdout.strip()
def publish():
    assert R['public_repo'] and R['public_clone'],'set run.public_repo and run.public_clone in the config'
    docs=R.get('publish_docs',{}); dirs=['pipeline','annotation_tool']+sorted({d.split('/')[0] for d in docs})
    assert not git('status','--porcelain','--','pipeline','annotation_tool',*docs.values()),'commit the pipeline, tool and published documents first'
    pc=(HERE/R['public_clone']).resolve(); src=git('rev-parse','HEAD')
    if not pc.exists(): subprocess.run(['git','clone',R['public_repo'],str(pc)],check=True)
    git('pull','--ff-only',cwd=pc,check=False)
    for d in dirs: shutil.rmtree(pc/d,ignore_errors=True)
    for f in git('ls-files','pipeline','annotation_tool').splitlines():
        (pc/f).parent.mkdir(parents=True,exist_ok=True); shutil.copy2(HERE/f,pc/f)
    for d,f in docs.items():
        for g in git('ls-files',f).splitlines():
            t=pc/d/pathlib.Path(g).relative_to(f) if (HERE/f).is_dir() else pc/d
            t.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(HERE/g,t)
    git('add','-A',cwd=pc)
    if git('status','--porcelain',cwd=pc):
        git('commit','-m',f'Sync pipeline from private commit {src[:7]}',cwd=pc); git('push',cwd=pc)
    return git('rev-parse','HEAD',cwd=pc)
if __name__=='__main__': print(publish())
