"""Avaliação final dos melhores checkpoints, sem ajustar com o conjunto de teste."""
import gc
import time
import hashlib
from common import ROOT, MODELS, load_json, save_json, log
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

def evaluate():
    if not torch.cuda.is_available():
        raise RuntimeError('Avaliação CUDA exigida para comparação de velocidade')
    rows=[]
    for task,arch,_ in MODELS:
        folder=ROOT/'resultados'/task/arch
        completed=load_json(folder/'concluido.json')
        if completed['epochs']!=20:
            raise RuntimeError('Experimento final incompleto')
        ckpt=folder/'weights/best.pt'
        digest=hashlib.sha256(ckpt.read_bytes()).hexdigest()
        for split in ['val','test']:
            output=folder/f'avaliacao_{split}'
            marker=output/'metricas.json'
            if marker.exists():
                record=load_json(marker)
                if record['checkpoint_sha256']!=digest:
                    raise RuntimeError('Checkpoint alterado após avaliação; resultados preservados')
                rows.extend(record['rows']); continue
            model=YOLO(str(ckpt))
            # Contagem antes da fusão de convoluções e BatchNorm.
            parameters=sum(p.numel() for p in model.model.parameters())
            metrics=model.val(data=str(ROOT/'config'/f'{task}.yaml'),split=split,imgsz=640,
                              batch=completed['batch'],device=0,workers=0,half=True,plots=True,
                              project=str(folder),name=f'avaliacao_{split}',exist_ok=True,
                              conf=.001,iou=.7,max_det=300,overlap_mask=False,mask_ratio=4,verbose=False)
            output.mkdir(exist_ok=True)
            selected=[]
            for kind,metric in [('box',metrics.box)]+([('mask',metrics.seg)] if task=='segmentation' else []):
                row=dict(task=task,model=arch,split=split,metric_type=kind,precision=float(metric.mp),recall=float(metric.mr),
                         map50=float(metric.map50),map50_95=float(metric.map),parameters=parameters,
                         inference_ms=float(metrics.speed['inference']),preprocess_ms=float(metrics.speed['preprocess']),
                         postprocess_ms=float(metrics.speed['postprocess']),training_seconds=completed.get('epoch_seconds_total'),batch=completed['batch'])
                selected.append(row)
                perclass=[]
                for idx,cid in enumerate(metric.ap_class_index):
                    perclass.append(dict(class_id=int(cid),classe=model.names[int(cid)],precision=float(metric.p[idx]),
                                         recall=float(metric.r[idx]),map50=float(metric.ap50[idx]),map50_95=float(metric.ap[idx])))
                pd.DataFrame(perclass).to_csv(output/f'classes_{kind}.csv',index=False)
            # Benchmark separado: mesma imagem de validação, lote 1, aquecimento e 30 repetições.
            source=next((ROOT/'dataset'/task/'images/val').iterdir())
            for _ in range(5):
                model.predict(str(source),imgsz=640,device=0,half=True,verbose=False)
            elapsed=[]
            for _ in range(30):
                torch.cuda.synchronize(); start=time.perf_counter()
                model.predict(str(source),imgsz=640,device=0,half=True,verbose=False)
                torch.cuda.synchronize(); elapsed.append((time.perf_counter()-start)*1000)
            for row in selected:
                row.update(latency_e2e_mean_ms=float(np.mean(elapsed)),latency_e2e_std_ms=float(np.std(elapsed)))
            # Exemplos escolhidos por ordem de nome, nunca por qualidade da predição.
            samples=sorted((ROOT/'dataset'/task/'images'/split).iterdir())[:3]
            predictions=model.predict([str(p) for p in samples],imgsz=640,device=0,half=True,conf=.25,verbose=False)
            for i,pred in enumerate(predictions):
                pred.save(filename=str(output/f'predicao_{i+1}.jpg'))
            save_json(marker,dict(checkpoint_sha256=digest,rows=selected,speed=metrics.speed,
                                  benchmark=dict(image=str(source),warmup=5,repeats=30,batch=1,includes='leitura, pré-processamento, modelo e NMS; API predict'),
                                  metric_definition='AP IoU=0.50 e média IoU=0.50:0.95; precision/recall no ponto F1 ótimo da curva.'))
            rows.extend(selected)
            del model,metrics,predictions
            gc.collect(); torch.cuda.empty_cache()
    df=pd.DataFrame(rows)
    df.to_csv(ROOT/'relatorios/metricas_comparativas.csv',index=False)
    return df

if __name__=='__main__':
    evaluate()
