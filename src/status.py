"""Progresso real, sem carregar modelos ou depender de GPU."""
import csv
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
from common import ROOT,MODELS,load_json

if __name__=='__main__':
    print(datetime.now(ZoneInfo('America/Sao_Paulo')).isoformat(timespec='seconds'))
    for task,arch,_ in MODELS:
        path=ROOT/'resultados'/task/arch
        csvpath=path/'results.csv'
        rows=[]
        if csvpath.exists():
            with csvpath.open(newline='',encoding='utf-8-sig') as stream:
                rows=list(csv.DictReader(stream))
        timings=load_json(path/'tempos_epocas.json') if (path/'tempos_epocas.json').exists() else []
        print(f'{task:13} {arch:6}: {len(rows):2}/20 épocas; concluído={(path/"concluido.json").exists()}',end='')
        if timings:
            print(f'; última época={timings[-1]["seconds"]:.1f}s; pico CUDA={timings[-1]["cuda_peak_allocated_bytes"]/1e9:.2f}GB',end='')
        print()
    estimate=ROOT/'relatorios/estimativa_duracao.json'
    if estimate.exists() and '--estimate' in sys.argv:
        print('Estimativa de referência (não contagem regressiva):',load_json(estimate))
