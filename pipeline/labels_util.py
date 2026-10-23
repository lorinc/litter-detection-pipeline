"""Label-record helpers shared by the tool's rule and the export.
covered(p, boxes): a proposal whose centre lies inside an annotator box sits on a labelled butt; it is 'covered' and
is never a hard negative. Run directly to move covered entries out of rejected_proposals in existing labels:
usage: labels_util.py [--config PATH] [--apply]   (without --apply it only counts)
"""
import json,sys
def covered(p,boxes):
    cx,cy=p['x']+p['w']/2,p['y']+p['h']/2
    return any(b['x']<=cx<=b['x']+b['w'] and b['y']<=cy<=b['y']+b['h'] for b in boxes)
def split_covered(rec):
    """Return rec with rejected proposals that are covered moved to covered_proposals (implicit flag dropped)."""
    rej,cov=[],list(rec.get('covered_proposals',[]))
    for r in rec.get('rejected_proposals',[]):
        if covered(r,rec['boxes']): cov.append({k:v for k,v in r.items() if k!='implicit'})
        else: rej.append(r)
    return {**rec,'rejected_proposals':rej,'covered_proposals':cov},len(rec.get('rejected_proposals',[]))-len(rej)
if __name__=='__main__':
    from common import load_config
    data=load_config()['_data']; moved=files=0
    for lf in sorted((data/'labels').glob('*.json')):
        rec,n=split_covered(json.load(open(lf)))
        if n:
            moved+=n; files+=1
            if '--apply' in sys.argv: json.dump(rec,open(lf,'w'),indent=1)
    print(f"{moved} rejected proposals in {files} label files are covered","(moved)" if '--apply' in sys.argv else "(dry run)")
