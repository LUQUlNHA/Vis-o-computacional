"""Inspeção e conversão LabelMe → YOLO, sem alterar os dados originais."""
from collections import Counter, defaultdict
import hashlib
import random
import shutil
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from shapely import make_valid
from shapely.geometry import Polygon, box
import yaml
from common import ROOT, load_json, save_json, log
from download import download

def polygon_parts(geom):
    if geom.is_empty:
        return []
    if geom.geom_type=='Polygon':
        return [geom] if geom.area>0 else []
    return [p for g in getattr(geom,'geoms',[]) for p in polygon_parts(g)]

def prepare():
    marker=ROOT/'relatorios/preparacao.json'
    if marker.exists():
        log('Conversão já concluída; preservada')
        return load_json(marker)
    base=download()
    classes=[s.strip() for s in (base/'classes.txt').read_text(encoding='utf-8-sig').splitlines() if s.strip() and s.strip().lower()!='background']
    ids={c:i for i,c in enumerate(classes)}
    assert len(classes)==15
    images={}
    for p in (base/'images').iterdir():
        if p.suffix.lower() in {'.jpg','.jpeg','.png'}:
            if p.stem in images:
                raise ValueError('Nome de imagem duplicado: '+p.stem)
            images[p.stem]=p
    annotations={p.stem:p for p in (base/'labels_json').glob('*.json')}
    masks={}
    for directory in sorted(base.glob('*masks')):
        files={p.stem:p for p in directory.glob('*.png')}
        samples=[]
        for stem in sorted(files)[:3]:
            with Image.open(files[stem]) as mask:
                samples.append(dict(file=files[stem].name,mode=mask.mode,size=list(mask.size),colors=mask.getcolors(maxcolors=32)))
        masks[directory.name]=dict(files=len(files),missing_for_images=sorted(images.keys()-files.keys()),orphan_masks=sorted(files.keys()-images.keys()),samples=samples,
                                   usage='Inspeção semântica; não usado como identidade de instâncias')
    save_json(ROOT/'relatorios/inspecao_mascaras.json',masks)
    issues=[]
    records=[]
    types=Counter()
    grouped=0
    for stem,p in sorted(images.items()):
        if stem not in annotations:
            issues.append(dict(image=stem,action='imagem excluída',reason='JSON ausente')); continue
        try:
            with Image.open(p) as im:
                im.load()
                w,h=im.size
                orientation=im.getexif().get(274,1)
                if orientation in {5,6,7,8}:
                    w,h=h,w
                if orientation!=1:
                    issues.append(dict(image=stem,action='interpretação EXIF',reason=f'Orientação {orientation}: dimensões orientadas, como JSON e OpenCV/Ultralytics; bytes originais preservados'))
            j=load_json(annotations[stem])
            if (j['imageWidth'],j['imageHeight'])!=(w,h):
                raise ValueError('Dimensão JSON difere da imagem')
        except Exception as exc:
            issues.append(dict(image=stem,action='imagem excluída',reason=str(exc))); continue
        groups=defaultdict(list)
        for k,s in enumerate(j['shapes']):
            types[s.get('shape_type','polygon')]+=1
            key=(s['label'].strip(), 'shape', k) if s.get('group_id') is None else (s['label'].strip(),'group',s['group_id'])
            groups[key].append((k,s))
        objects=[]
        seen_polygons=set()
        for key,shapes in groups.items():
            label=key[0]
            parts=[]
            if label.lower()=='background':
                issues.append(dict(image=stem,group=str(key),action='fundo ignorado',reason='Background não é uma classe de equipamento YOLO'))
                continue
            try:
                if label not in ids:
                    raise ValueError('Classe desconhecida: '+label)
                for k,s in shapes:
                    if s.get('shape_type','polygon')!='polygon':
                        raise ValueError('Tipo não poligonal: '+str(s.get('shape_type')))
                    a=np.asarray(s['points'],dtype=float)
                    if a.ndim!=2 or a.shape[1]!=2 or not np.isfinite(a).all() or len(np.unique(a,axis=0))<3:
                        raise ValueError('Vértices insuficientes ou não finitos')
                    geom=Polygon(a)
                    if not geom.is_valid:
                        geom=make_valid(geom)
                        issues.append(dict(image=stem,shape=k,action='corrigida',reason='make_valid: polígono inválido'))
                    if (a[:,0]<0).any() or (a[:,0]>w).any() or (a[:,1]<0).any() or (a[:,1]>h).any():
                        geom=geom.intersection(box(0,0,w,h))
                        issues.append(dict(image=stem,shape=k,action='corrigida',reason='recorte geométrico nos limites da imagem'))
                    parts.extend(polygon_parts(geom))
                if not parts:
                    raise ValueError('Área nula após validação')
                # Partes de um group_id permanecem UMA instância; pontes de largura zero
                # são necessárias porque o TXT YOLO não armazena MultiPolygon ou furos.
                from shapely.ops import unary_union
                union=unary_union(parts)
                parts=polygon_parts(union)
                if any(len(p.interiors) for p in parts):
                    issues.append(dict(image=stem,group=str(key),action='limitação',reason='formato YOLO não representa furos; usa contorno externo'))
                contours=[np.asarray(p.exterior.coords)[:-1] for p in parts]
                if len(contours)>1:
                    from ultralytics.data.converter import merge_multi_segment
                    a=np.concatenate(merge_multi_segment(contours),axis=0)
                    issues.append(dict(image=stem,group=str(key),action='corrigida',reason='partes da mesma instância unidas com pontes de largura zero'))
                else:
                    a=contours[0]
                normalized=np.clip(a/np.array([w,h]),0,1)
                x1,y1,x2,y2=union.bounds
                bbox=np.array([(x1+x2)/2/w,(y1+y2)/2/h,(x2-x1)/w,(y2-y1)/h])
                assert np.isfinite(normalized).all() and (bbox[2:]>0).all()
                # Duplicatas literalmente idênticas não representam outra instância.
                signature=(ids[label],normalized.tobytes())
                if signature in seen_polygons:
                    issues.append(dict(image=stem,group=str(key),action='duplicata excluída',reason='Mesma classe e mesmos vértices normalizados de outro objeto'))
                    continue
                seen_polygons.add(signature)
                objects.append(dict(class_id=ids[label],bbox=bbox.tolist(),polygon=normalized.tolist(),area_fraction=union.area/(w*h)))
                grouped+=len(shapes)>1
            except Exception as exc:
                issues.append(dict(image=stem,group=str(key),action='objeto excluído',reason=str(exc)))
        with p.open('rb') as stream:
            digest=hashlib.file_digest(stream,'sha256').hexdigest()
        records.append(dict(stem=stem,image=str(p.relative_to(ROOT)),json=str(annotations[stem].relative_to(ROOT)),width=w,height=h,sha256=digest,objects=objects))
        if len(records)%100==0:
            log(f'Inspecionadas {len(records)}/{len(images)} imagens')
    for stem in annotations.keys()-images.keys():
        issues.append(dict(image=stem,action='JSON órfão',reason='imagem ausente'))
    present={o['class_id'] for r in records for o in r['objects']}
    if present!=set(range(len(classes))):
        raise RuntimeError('Uma ou mais classes não têm nenhum objeto válido: revise a conversão antes de treinar')
    # Imagens byte a byte idênticas são agrupadas para evitar vazamento exato.
    unique=defaultdict(list)
    for r in records:
        unique[r['sha256']].append(r)
    groups=list(unique.values())
    random.Random(42).shuffle(groups)
    n=len(records); targets=[round(n*.7),round(n*.15)]
    split_names=['train','val','test']; split=0; used=0
    for group in groups:
        if split<2 and used>=targets[split]:
            split+=1; used=0
        for r in group:
            r['split']=split_names[split]
        used+=len(group)
    assert not (set(r['sha256'] for r in records if r['split']=='train') & set(r['sha256'] for r in records if r['split']!='train'))
    rows=[]
    for r in records:
        stem=r['stem']; subset=r['split']
        for task in ['detection','segmentation']:
            dest=ROOT/'dataset'/task
            image_dest=dest/'images'/subset/(stem+Path(r['image']).suffix.lower())
            image_dest.parent.mkdir(parents=True,exist_ok=True)
            if not image_dest.exists():
                # Cópias independentes: nenhum reparo JPEG posterior pode tocar o original.
                shutil.copy2(ROOT/r['image'],image_dest)
            label_dest=dest/'labels'/subset/(stem+'.txt')
            label_dest.parent.mkdir(parents=True,exist_ok=True)
            lines=[]
            for o in r['objects']:
                coords=o['bbox'] if task=='detection' else np.asarray(o['polygon']).flatten()
                lines.append(str(o['class_id'])+' '+' '.join(f'{v:.9f}' for v in coords))
            label_dest.write_text('\n'.join(lines)+'\n' if lines else '',encoding='utf-8')
        counts=Counter(o['class_id'] for o in r['objects'])
        for cid,name in enumerate(classes):
            rows.append(dict(image=stem,split=subset,class_id=cid,classe=name,objetos=counts[cid]))
    for task in ['detection','segmentation']:
        cfg=dict(path=(ROOT/'dataset'/task).as_posix(),train='images/train',val='images/val',test='images/test',names=dict(enumerate(classes)))
        (ROOT/'config'/f'{task}.yaml').write_text(yaml.safe_dump(cfg,allow_unicode=True),encoding='utf-8')
    df=pd.DataFrame(rows)
    df.to_csv(ROOT/'relatorios/distribuicao_imagens_classes.csv',index=False)
    df.groupby(['split','class_id','classe']).objetos.sum().to_csv(ROOT/'relatorios/distribuicao_classes.csv')
    save_json(ROOT/'relatorios/anotacoes_convertidas.json',records)
    save_json(ROOT/'relatorios/correcoes_descartes.json',issues)
    inventory={p.name:len(list(p.iterdir())) for p in base.iterdir() if p.is_dir()}
    report=dict(classes=classes,images_original=len(images),json_original=len(annotations),valid_images=n,
                objects=sum(len(r['objects']) for r in records),splits=dict(Counter(r['split'] for r in records)),
                annotation_types=dict(types),grouped_instances=grouped,exact_duplicate_images=n-len(unique),
                issues=len(issues),inventory=inventory,seed=42,split_method='shuffle de grupos SHA256; 70/15/15 com arredondamento',
                limitation='Uma única subestação; divisão aleatória não garante independência de pontos de vista ou datas. Polígonos sem group_id são tratados como instâncias anotadas, não identidade física rastreada.')
    visualize(records,classes)
    save_json(marker,report)
    log(str(report))
    return report

def visualize(records,classes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as PatchPolygon, Rectangle
    from matplotlib.lines import Line2D
    colors=plt.get_cmap('tab20')
    for idx,r in enumerate(random.Random(42).sample(records,3)):
        image=ImageOps.exif_transpose(Image.open(ROOT/r['image'])); image.thumbnail((1200,1200))
        w,h=image.size
        fig,axes=plt.subplots(1,3,figsize=(19,7))
        for ax,title in zip(axes,['Original','Caixas derivadas dos polígonos','Instâncias anotadas']):
            ax.imshow(image); ax.set_title(title); ax.axis('off')
        for o in r['objects']:
            c=colors(o['class_id']); x,y,bw,bh=o['bbox']
            axes[1].add_patch(Rectangle(((x-bw/2)*w,(y-bh/2)*h),bw*w,bh*h,fill=False,edgecolor=c,lw=1))
            axes[2].add_patch(PatchPolygon(np.array(o['polygon'])*[w,h],facecolor=c,edgecolor=c,alpha=.35,lw=.7))
        present=sorted(set(o['class_id'] for o in r['objects']))
        fig.legend(handles=[Line2D([0],[0],color=colors(c),lw=4,label=classes[c]) for c in present],loc='lower center',ncol=4,fontsize=8)
        fig.suptitle(f"{r['stem']} | {len(r['objects'])} instâncias | conjunto: {r['split']}")
        fig.tight_layout(rect=[0,.15,1,.95]); fig.savefig(ROOT/'graficos'/f'validacao_{idx+1}.png',dpi=140); plt.close(fig)
        for kind,title in [('original','Original'),('boxes','Caixas derivadas'),('polygons','Polígonos anotados')]:
            fig,ax=plt.subplots(figsize=(11,9))
            ax.imshow(image); ax.axis('off'); ax.set_title(f"{r['stem']} — {title}",fontsize=15)
            if kind!='original':
                for o in r['objects']:
                    c=colors(o['class_id']); x,y,bw,bh=o['bbox']
                    if kind=='boxes':
                        ax.add_patch(Rectangle(((x-bw/2)*w,(y-bh/2)*h),bw*w,bh*h,fill=False,edgecolor=c,lw=1))
                    else:
                        ax.add_patch(PatchPolygon(np.array(o['polygon'])*[w,h],facecolor=c,edgecolor=c,alpha=.35,lw=.7))
                fig.legend(handles=[Line2D([0],[0],color=colors(c),lw=4,label=classes[c]) for c in present],loc='lower center',ncol=2,fontsize=13)
            fig.tight_layout(rect=[0,.20,1,.96]); fig.savefig(ROOT/'graficos'/f'validacao_{idx+1}_{kind}.png',dpi=140); plt.close(fig)
    df=pd.read_csv(ROOT/'relatorios/distribuicao_classes.csv')
    table=df.pivot(index='classe',columns='split',values='objetos').fillna(0)
    table.plot.barh(stacked=True,figsize=(12,8)); plt.xlabel('Objetos anotados'); plt.ylabel('Classe'); plt.title('Distribuição das instâncias por classe e divisão'); plt.tight_layout(); plt.savefig(ROOT/'graficos/distribuicao.png',dpi=150); plt.close()

from pathlib import Path
if __name__=='__main__':
    prepare()
