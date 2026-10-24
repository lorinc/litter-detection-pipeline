"""Local annotation server (Python standard library only, no network beyond localhost).
usage: annotate.py [--config PATH]   then open http://127.0.0.1:<annotate.port>/
Serves annotation_tool/index.html, the image copies (PNG, or JPEG as in annotation_tool/example/), labels and proposals. Proposals are served for train images only; val and
test requests are refused here (blind rule), whatever the page asks. Labels autosave to <data_dir>/labels/<image>.json.
"""
import json,os,re
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from common import HERE,load_config,load_split
cfg=load_config(); data=cfg['_data']; A=cfg['annotate']
split=load_split(cfg); labels=data/'labels'; labels.mkdir(exist_ok=True)
STEM=re.compile(r'^[0-9_]+$')
def label_of(s):
    f=labels/f'{s}.json'
    return json.load(open(f)) if f.exists() else None
def summary():
    out=[]
    for s in sorted(split):
        l=label_of(s); out.append({'image':s,'split':split[s],'status':l['status'] if l else 'unseen','n':len(l['boxes']) if l else 0})
    return out
class H(BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def send(self,code,body,ctype='application/json'):
        if isinstance(body,(dict,list)): body=json.dumps(body).encode()
        self.send_response(code); self.send_header('Content-Type',ctype); self.send_header('Content-Length',str(len(body)))
        self.send_header('Cache-Control','no-store'); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        p=self.path.split('?')[0].strip('/').split('/')
        if p==['']: return self.send(200,(HERE/'annotation_tool/index.html').read_bytes(),'text/html; charset=utf-8')
        if p==['api','config']: return self.send(200,{k:A[k] for k in ('annotator_pass','loupe_px','loupe_zooms','accept_all_default')})
        if p==['api','list']: return self.send(200,summary())
        if len(p)==3 and p[0]=='api' and STEM.match(p[2]) and p[2] in split:
            s=p[2]
            if p[1]=='image':
                f=data/'images'/f'{s}.png'
                if f.exists(): return self.send(200,f.read_bytes(),'image/png')
                return self.send(200,(data/'images'/f'{s}.jpg').read_bytes(),'image/jpeg')
            if p[1]=='labels': return self.send(200,label_of(s) or {})
            if p[1]=='proposals':
                if split[s]!='train': return self.send(403,{'error':'blind image: proposals are never served for val/test'})
                f=data/'proposals'/f'{s}.json'
                return self.send(200,json.load(open(f)) if f.exists() else {'boxes':[],'missing':True})
        self.send(404,{'error':'not found'})
    def do_POST(self):
        p=self.path.strip('/').split('/')
        if not (len(p)==3 and p[:2]==['api','labels'] and STEM.match(p[2]) and p[2] in split): return self.send(404,{'error':'not found'})
        rec=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        ok=rec.get('image')==p[2] and rec.get('status') in ('unseen','done') and isinstance(rec.get('boxes'),list) \
           and all(b.get('cls') in ('butt','maybe') and all(isinstance(b.get(k),(int,float)) for k in 'xywh') for b in rec['boxes'])
        if not ok: return self.send(400,{'error':'invalid label record'})
        if split[p[2]]!='train' and (rec.get('rejected_proposals') or rec.get('covered_proposals')): return self.send(400,{'error':'blind image cannot carry proposals'})
        tmp=labels/f'{p[2]}.json.tmp'; json.dump(rec,open(tmp,'w'),indent=1); os.replace(tmp,labels/f'{p[2]}.json')
        self.send(200,{'saved':True})
print(f"annotation tool on http://{A['host']}:{A['port']}/  ({len(split)} images)",flush=True)
ThreadingHTTPServer((A['host'],A['port']),H).serve_forever()
