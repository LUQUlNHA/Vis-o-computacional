"""Orquestrador sequencial, retomável e com bloqueio de execução concorrente."""
import argparse
import os
import subprocess
import sys
import time
import msvcrt
from common import ROOT, MODELS, save_json, load_json, log

def call(script,*args):
    log('Executando: '+script+' '+' '.join(args))
    cmd=[sys.executable,'-u',str(ROOT/'src'/script),*args]
    result=subprocess.run(cmd,cwd=ROOT)
    if result.returncode:
        raise RuntimeError(f'Processo falhou ({result.returncode}): {script}; consulte relatorios/pipeline.log')

def main():
    p=argparse.ArgumentParser(); p.add_argument('--report-only',action='store_true'); a=p.parse_args()
    lockpath=ROOT/'relatorios/pipeline.lock'
    lockpath.parent.mkdir(exist_ok=True)
    with lockpath.open('a+b') as lock:
        lock.seek(0); lock.write(b'1'); lock.flush(); lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:
            raise RuntimeError('Outro pipeline está ativo; execução concorrente bloqueada')
        statepath=ROOT/'relatorios/execucao.json'
        origin=ROOT/'dataset/original/zenodo_record.json'
        initial=origin.stat().st_birthtime if origin.exists() else time.time()
        state=load_json(statepath) if statepath.exists() else dict(started=initial,sessions=[],time_scope='Tempo corrido desde registro inicial do download; inclui download, preparação e execução.')
        started=time.time()
        state['status']='em execução'
        state.pop('error',None)
        save_json(statepath,state)
        try:
            if not a.report_only:
                call('environment.py')
                call('prepare.py')
                call('audit_source.py')
                call('validate_data.py')
                batches=load_json(ROOT/'config/batches.json') if (ROOT/'config/batches.json').exists() else {'detection':8,'segmentation':8}
                # Todos os testes funcionais precedem qualquer experimento final.
                for task in ['detection','segmentation']:
                    while True:
                        try:
                            for t,arch,_ in MODELS:
                                if t==task:
                                    call('train.py','--task',t,'--arch',arch,'--batch',str(batches[task]),'--smoke')
                            break
                        except RuntimeError:
                            # Um erro não identificado não autoriza alterar a metodologia.
                            # O operador verifica o log: somente CUDA OOM permite batch 4.
                            raise RuntimeError('Teste funcional falhou. Verifique o erro antes do treino final. Se CUDA OOM, use src/set_batch.py para batch 4 em AMBAS as arquiteturas da tarefa.')
                save_json(ROOT/'config/batches.json',batches)
                for task,arch,_ in MODELS:
                    call('train.py','--task',task,'--arch',arch,'--batch',str(batches[task]))
                call('evaluate.py')
                call('benchmark.py')
            call('report.py')
            state['status']='concluido'
            state.pop('error',None)
        except BaseException as exc:
            state['status']='interrompido'; state['error']=str(exc)
            raise
        finally:
            state['sessions'].append(dict(started=started,finished=time.time(),seconds=time.time()-started))
            state['active_seconds_total']=sum(s['seconds'] for s in state['sessions'])
            state['elapsed_wall_seconds']=time.time()-state['started']
            save_json(statepath,state)
            lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)

if __name__=='__main__':
    main()
