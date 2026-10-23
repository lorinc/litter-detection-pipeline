"""Make upright, metadata-free, lossless copies of the source photos and keep <data_dir>/meta.csv current.
usage: preprocess.py [--config PATH] [--force]   (never modifies the source photos)
- Orientation from EXIF is baked into the pixels; copies are PNG with no EXIF or text chunks (<data_dir>/images/<stem>.png).
- Photos in the config's "exclude" get no image copy; their meta.csv row is kept.
- Regions in <data_dir>/privacy_boxes.json ({stem: [[x,y,w,h], ...]}, upright native px) are Gaussian-blurred.
- meta.csv: extracted columns (timestamp, gps as "lat,lon", w, h, privacy) are rewritten; manual columns
  (shot_type, light, surface) are kept as they are. Skips images already written unless --force.
"""
import csv,json,sys
from PIL import Image,ImageOps,ImageFilter
from common import HERE,load_config
cfg=load_config(); data=cfg['_data']
out=data/'images'; out.mkdir(parents=True,exist_ok=True)
pf=data/'privacy_boxes.json'; priv=json.load(open(pf)) if pf.exists() else {}
metaf=data/'meta.csv'; MANUAL=['shot_type','light','surface']
old={r['image']:r for r in csv.DictReader(open(metaf))} if metaf.exists() else {}
def latlon(g):  # EXIF GPS IFD -> "lat,lon" in decimal degrees, '' if absent
    if not g or 2 not in g or 4 not in g: return ''
    dec=lambda v,ref:(1 if ref in ('N','E') else -1)*(float(v[0])+float(v[1])/60+float(v[2])/3600)
    return f"{dec(g[2],g.get(1,'N')):.6f},{dec(g[4],g.get(3,'E')):.6f}"
force='--force' in sys.argv; rows=[]; excl=set(cfg.get('exclude',[]))
for f in sorted(HERE.glob(cfg['source_glob'])):
    stem=f.stem; im=Image.open(f); ex=im.getexif()
    ts=ex.get_ifd(0x8769).get(36867,''); gps=ex.get_ifd(0x8825)
    up=ImageOps.exif_transpose(im).convert('RGB')
    o=out/f'{stem}.png'
    if stem not in excl and (force or not o.exists()):
        for x,y,w,h in priv.get(stem,[]):
            box=(x,y,x+w,y+h); up.paste(up.crop(box).filter(ImageFilter.GaussianBlur(25)),box)
        up.save(o,optimize=False)
    r={'image':stem,'timestamp':ts,'gps':latlon(gps),'w':up.width,'h':up.height,
       'privacy':'blurred' if stem in priv else ''}
    r.update({k:old.get(stem,{}).get(k,'') for k in MANUAL}); rows.append(r)
with open(metaf,'w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows),'images;',sum(1 for r in rows if r['gps']),'with GPS;',sum(1 for r in rows if r['privacy']),'blurred')
