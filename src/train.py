"""Um modelo por processo CUDA; testes funcionais isolados dos resultados finais."""
import argparse
import gc
import time
import shutil
from common import ROOT, TRAIN, MODELS, load_json, save_json, log
import numpy as np
import pandas as pd
import torch
import psutil
import yaml
from ultralytics import YOLO

def smoke_yaml(task):
    records=load_json(ROOT/'relatorios/anotacoes_convertidas.json')
    candidates=[r for r in records if r['split']=='train']
    # Amostra inclui imagens densas, objetos pequenos e dimensões grandes.
    ordered=sorted(candidates,key=lambda r:len(r['objects']),reverse=True)
    selected=ordered[:8]+sorted(candidates,key=lambda r:r['width']*r['height'],reverse=True)[:8]
    selected=list({r['stem']:r for r in selected}.values())
    images=ROOT/'dataset'/task/'images/train'
    lookup={p.stem:p for p in images.iterdir()}
    listing=ROOT/'config'/f'smoke_{task}.txt'
    listing.write_text('\n'.join(lookup[r['stem']].as_posix() for r in selected),encoding='utf-8')
    cfg=yaml.safe_load((ROOT/'config'/f'{task}.yaml').read_text(encoding='utf-8'))
    cfg.update(train=listing.as_posix(),val=listing.as_posix())
    cfg.pop('test',None)
    path=ROOT/'config'/f'smoke_{task}.yaml'
    path.write_text(yaml.safe_dump(cfg,allow_unicode=True),encoding='utf-8')
    return path

def run(task,arch,batch=8,smoke=False):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA indisponível: treinamento completo em CPU bloqueado.')
    weight=next(m for t,a,m in MODELS if (t,a)==(task,arch))
    folder=ROOT/'resultados'/task/arch
    if smoke:
        folder=ROOT/'resultados/funcional'/task/f'{arch}_batch{batch}'
    complete=folder/'concluido.json'
    if complete.exists():
        log(f'Experimento concluído preservado: {folder}')
        return
    csv=folder/'results.csv'
    if csv.exists() and len(pd.read_csv(csv))==(1 if smoke else 20) and (folder/'weights/best.pt').exists():
        # Recupera a finalização interrompida após a última época.
        args=yaml.safe_load((folder/'args.yaml').read_text(encoding='utf-8'))
        if args['batch']!=batch or args['epochs']!=(1 if smoke else 20):
            raise RuntimeError('Resultados existentes com configuração diferente; preservados')
        save_json(complete,dict(recovered=True,batch=batch,epochs=len(pd.read_csv(csv))))
        return
    ckpt=folder/'weights/last.pt'
    if folder.exists() and any(folder.iterdir()) and not ckpt.exists():
        # Tentativa sem checkpoint é mantida para auditoria, nunca sobrescrita.
        backup=folder.with_name(folder.name+'_interrompido_'+time.strftime('%Y%m%d_%H%M%S'))
        folder.rename(backup)
    data=smoke_yaml(task) if smoke else ROOT/'config'/f'{task}.yaml'
    model=YOLO(str(ckpt) if ckpt.exists() else str(ROOT/weight))
    timings_path=folder/'tempos_epocas.json'
    timings=load_json(timings_path) if timings_path.exists() else []
    clock={}
    expected_epochs=1 if smoke else 20
    def epoch_start(trainer):
        clock['start']=time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        if trainer.batch_size!=batch:
            raise RuntimeError('BATCH_ALTERADO_POR_OOM: necessário ajustar ambos os modelos da tarefa')
    def epoch_end(trainer):
        # Ultralytics também chama este evento na validação final de best.pt,
        # com epoch=epochs; esse evento não constitui uma época adicional.
        if trainer.epoch>=expected_epochs:
            return
        if trainer.batch_size!=batch:
            raise RuntimeError('BATCH_ALTERADO_POR_OOM: necessário ajustar ambos os modelos da tarefa')
        elapsed=time.perf_counter()-clock['start']
        timings.append(dict(epoch=trainer.epoch+1,seconds=elapsed,
                            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            cuda_free_bytes=torch.cuda.mem_get_info()[0],batch=trainer.batch_size,
                            ram_available_bytes=psutil.virtual_memory().available,
                            process_rss_bytes=psutil.Process().memory_info().rss))
        save_json(timings_path,timings)
        log(f'Época {trainer.epoch+1}/{expected_epochs}: {elapsed:.1f} s; estimativa restante deste modelo: {elapsed*(expected_epochs-trainer.epoch-1)/60:.1f} min')
        if not smoke and trainer.epoch==0:
            position=next(i for i,(t,a,_) in enumerate(MODELS) if (t,a)==(task,arch))
            remaining=(20-1)+20*(len(MODELS)-position-1)
            estimate=dict(reference_model=f'{task}/{arch}',first_epoch_seconds=elapsed,
                          remaining_epochs=remaining,estimated_remaining_seconds=elapsed*remaining,
                          caveat='Estimativa preliminar: supõe custo por época igual ao atual; segmentação, aquecimento e fim do mosaico alteram a duração. Exclui avaliações e exportação.')
            save_json(ROOT/'relatorios/estimativa_duracao.json',estimate)
            log(f'Estimativa preliminar dos treinos restantes: {elapsed*remaining/3600:.2f} h, supondo o custo desta primeira época; segmentação pode ter custo diferente.')
    model.add_callback('on_train_epoch_start',epoch_start)
    model.add_callback('on_fit_epoch_end',epoch_end)
    started=time.perf_counter()
    cfg=dict(TRAIN,batch=batch,data=str(data),project=str(folder.parent),name=folder.name,exist_ok=True)
    if smoke:
        cfg.update(epochs=1,close_mosaic=0)
    if ckpt.exists():
        cfg.update(resume=str(ckpt))
    save_json(ROOT/'relatorios'/f'config_{task}_{arch}{"_funcional" if smoke else ""}.json',cfg)
    model.train(**cfg)
    torch.cuda.synchronize()
    if len(pd.read_csv(folder/'results.csv'))!=expected_epochs:
        raise RuntimeError('Número de épocas diferente do exigido')
    save_json(complete,dict(epochs=expected_epochs,batch=batch,seconds_this_session=time.perf_counter()-started,
                            epoch_seconds_total=sum(t['seconds'] for t in timings),
                            first_epoch_seconds=timings[0]['seconds'],functional=smoke,
                            effective_amp=bool(model.trainer.amp),effective_batch=model.trainer.batch_size,
                            optimizer_class=type(model.trainer.optimizer).__name__,
                            note='Tempos por época incluem validação; tempo de sessão inclui inicialização e avaliação final.'))
    del model
    gc.collect(); torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--task',choices=['detection','segmentation'],required=True)
    p.add_argument('--arch',choices=['YOLOv8','YOLO11'],required=True); p.add_argument('--batch',type=int,choices=[4,8],default=8)
    p.add_argument('--smoke',action='store_true'); a=p.parse_args()
    run(a.task,a.arch,a.batch,a.smoke)
