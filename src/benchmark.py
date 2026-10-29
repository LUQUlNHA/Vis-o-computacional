"""Latência adicional em validação; preserva as avaliações originais."""
import gc
import time
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from common import ROOT,MODELS,load_json,save_json

def benchmark():
    path=ROOT/'relatorios/benchmark_padronizado.json'
    if path.exists():
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA necessária para benchmark comparável')
    # Mesma imagem real em todas as arquiteturas; nenhum uso do teste para ajuste.
    source=sorted((ROOT/'dataset/detection/images/val').glob('*.jpg'))[0]
    results=[]
    observations=[]
    for task,arch,_ in MODELS:
        ckpt=ROOT/'resultados'/task/arch/'weights/best.pt'
        model=YOLO(str(ckpt))
        for _ in range(10):
            model.predict(str(source),imgsz=640,device=0,half=True,verbose=False)
        elapsed=[]
        for round_id in range(3):
            for sample in range(30):
                torch.cuda.synchronize(); start=time.perf_counter()
                model.predict(str(source),imgsz=640,device=0,half=True,verbose=False)
                torch.cuda.synchronize()
                ms=(time.perf_counter()-start)*1000
                elapsed.append(ms)
                observations.append(dict(task=task,model=arch,round=round_id+1,sample=sample+1,e2e_ms=ms))
        results.append(dict(task=task,model=arch,mean_ms=float(np.mean(elapsed)),std_ms=float(np.std(elapsed)),
                            median_ms=float(np.median(elapsed)),p95_ms=float(np.percentile(elapsed,95)),samples=len(elapsed)))
        del model
        gc.collect(); torch.cuda.empty_cache()
    pd.DataFrame(results).to_csv(ROOT/'relatorios/benchmark_padronizado.csv',index=False)
    pd.DataFrame(observations).to_csv(ROOT/'relatorios/benchmark_observacoes.csv',index=False)
    save_json(path,dict(image=str(source),warmup=10,rounds=3,repeats_per_round=30,batch=1,rows=results,
                        scope='API predict E2E: leitura + processamento + modelo + NMS, FP16 CUDA, imagem real de validação. Todos os valores mantidos, sem exclusão de outliers. Benchmark adicional motivado por dispersão nas medições originais, que foram preservadas.'))
    print(results,flush=True)

if __name__=='__main__':
    benchmark()
