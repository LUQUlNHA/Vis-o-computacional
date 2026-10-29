"""Ajuste explícito após OOM comprovado; não altera resultados existentes."""
import argparse
from common import ROOT, load_json, save_json
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('task',choices=['detection','segmentation']); p.add_argument('--reason',required=True); a=p.parse_args()
    path=ROOT/'config/batches.json'
    cfg=load_json(path) if path.exists() else {'detection':8,'segmentation':8}
    for arch in ['YOLOv8','YOLO11']:
        if (ROOT/'resultados'/a.task/arch/'weights/last.pt').exists():
            raise RuntimeError('Checkpoint final existente: não alterar batch de um experimento em andamento ou concluído')
    cfg[a.task]=4; save_json(path,cfg)
    save_json(ROOT/'relatorios'/f'ajuste_batch_{a.task}.json',dict(batch=4,reason=a.reason,applies_to=['YOLOv8','YOLO11']))
