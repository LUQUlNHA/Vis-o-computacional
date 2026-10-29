"""Verificação independente dos TXT efetivamente gravados e das divisões."""
from collections import Counter
import hashlib
import numpy as np
from PIL import Image, ImageOps
from common import ROOT, load_json, save_json

def validate():
    manifest=load_json(ROOT/'relatorios/anotacoes_convertidas.json')
    names=load_json(ROOT/'relatorios/preparacao.json')['classes']
    hashes={s:set() for s in ['train','val','test']}
    counts=Counter()
    duplicate_boxes=[]
    for r in manifest:
        hashes[r['split']].add(r['sha256'])
        for task in ['detection','segmentation']:
            folder=ROOT/'dataset'/task
            matches=list((folder/'images'/r['split']).glob(r['stem']+'.*'))
            assert len(matches)==1,(task,r['stem'],'imagem ausente/ambígua')
            image=matches[0]
            with image.open('rb') as f:
                assert hashlib.file_digest(f,'sha256').hexdigest()==r['sha256']
            with Image.open(image) as im:
                oriented_size=im.size[::-1] if im.getexif().get(274,1) in {5,6,7,8} else im.size
                assert oriented_size==(r['width'],r['height'])
            lines=(folder/'labels'/r['split']/(r['stem']+'.txt')).read_text().splitlines()
            assert len(lines)==len(r['objects'])
            seen=set()
            for i,line in enumerate(lines):
                vals=np.asarray([float(x) for x in line.split()])
                cid=vals[0]; coords=vals[1:]
                assert np.isfinite(vals).all() and cid==int(cid) and 0<=cid<len(names)
                assert (coords>=0).all() and (coords<=1).all()
                assert int(cid)==r['objects'][i]['class_id']
                if task=='detection':
                    assert len(coords)==4 and (coords[2:]>0).all()
                    assert (coords[:2]-coords[2:]/2>=-1e-8).all()
                    assert (coords[:2]+coords[2:]/2<=1+1e-8).all()
                    key=tuple(np.asarray(vals,dtype=np.float32))
                    if key in seen:
                        duplicate_boxes.append(dict(image=r['stem'],line=i,class_id=int(cid)))
                    seen.add(key)
                else:
                    assert len(coords)>=6 and len(coords)%2==0
                    p=coords.reshape(-1,2)
                    assert len(np.unique(p,axis=0))>=3
                    x1,y1=p.min(0); x2,y2=p.max(0)
                    box=np.array([(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1])
                    assert np.allclose(box,r['objects'][i]['bbox'],atol=3e-8)
                counts[task]+=1
    assert all(not hashes[a]&hashes[b] for a,b in [('train','val'),('train','test'),('val','test')])
    assert counts['detection']==counts['segmentation']
    report=dict(images=len(manifest),objects_per_task=dict(counts),exact_leakage=False,duplicate_boxes=duplicate_boxes)
    save_json(ROOT/'relatorios/validacao_dados.json',report)
    if duplicate_boxes:
        raise RuntimeError('Caixas duplicadas que o Ultralytics pode remover; revisão necessária. Consulte validacao_dados.json')
    print('Validação independente aprovada:',report,flush=True)

if __name__=='__main__':
    validate()
