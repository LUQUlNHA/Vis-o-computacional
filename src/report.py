"""Notebook executado e PDF a partir de artefatos reais; nunca reexecuta treinos."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'config/matplotlib'))
import ast
import base64
import io
import json
import re
import textwrap
from xml.sax.saxutils import escape
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nbformat as nbf
from nbclient import NotebookClient
from common import ROOT, MODELS, TRAIN, load_json, save_json

COVER = ('# Prova prática de Visão Computacional com YOLO\n\n'
         '**Comparação YOLOv8n × YOLO11n: detecção e segmentação de equipamentos elétricos**\n\n'
         '**Aluno:** Lucas Ferreira Neto\n\n'
         '**Matrícula:** 223026\n\n'
         '**Instituição:** Centro Universitário Facens (UniFacens)\n\n'
         'Execução local em Windows / RTX 4050. Quatro modelos, **20 épocas por modelo**. Dataset: Zenodo 7884270.')

def charts(df):
    plt.rcParams.update({'font.size':12,'axes.titlesize':13,'axes.labelsize':12,'legend.fontsize':11})
    distribution=pd.read_csv(ROOT/'relatorios/distribuicao_classes.csv').pivot(index='classe',columns='split',values='objetos').fillna(0)
    ax=distribution.plot.barh(stacked=True,figsize=(10,8))
    ax.set(xlabel='Instâncias anotadas',ylabel='Classe',title='Distribuição por classe e conjunto')
    plt.tight_layout(); plt.savefig(ROOT/'graficos/distribuicao.png',dpi=150); plt.close()
    for task in ['detection','segmentation']:
        frames={arch:pd.read_csv(ROOT/'resultados'/task/arch/'results.csv').rename(columns=lambda s:s.strip()) for arch in ['YOLOv8','YOLO11']}
        for stage,title in [('train','treinamento'),('val','validação')]:
            cols=[c for c in frames['YOLOv8'] if c.startswith(stage+'/') and 'loss' in c]
            fig,axes=plt.subplots(int(np.ceil(len(cols)/2)),2,figsize=(10,4*int(np.ceil(len(cols)/2))),squeeze=False)
            for ax,col in zip(axes.flat,cols):
                for arch,f in frames.items():
                    ax.plot(f.epoch,f[col],label=arch)
                ax.set(title=col,xlabel='Época',ylabel='Perda'); ax.grid(alpha=.2); ax.legend()
            for ax in list(axes.flat)[len(cols):]: ax.axis('off')
            fig.suptitle(f'{task}: perdas de {title}'); fig.tight_layout()
            fig.savefig(ROOT/'graficos'/f'{task}_{stage}_loss.png',dpi=150); plt.close(fig)
    for metric,label in [('map50','mAP@0.5'),('map50_95','mAP@0.5:0.95'),('inference_ms','Inferência do modelo (ms/imagem)')]:
        fig,axes=plt.subplots(2,1,figsize=(10,10))
        for ax,split in zip(axes,['val','test']):
            f=df[df.split==split].copy()
            f['Experimento']=f.task+' / '+f.metric_type+' / '+f.model
            ax.barh(f.Experimento,f[metric]); ax.set_title('Validação' if split=='val' else 'Teste'); ax.set_xlabel(label); ax.grid(axis='x',alpha=.2)
        fig.suptitle(label); fig.tight_layout(); fig.savefig(ROOT/'graficos'/f'comparativo_{metric}.png',dpi=150); plt.close(fig)
    f=df[(df.split=='test')&(df.metric_type=='box')].copy()
    bench=pd.read_csv(ROOT/'relatorios/benchmark_padronizado.csv')
    f=f.merge(bench,on=['task','model'],validate='one_to_one')
    names=f.task+' / '+f.model
    fig,axes=plt.subplots(3,1,figsize=(10,12))
    axes[0].barh(names,f.parameters/1e6); axes[0].set_xlabel('Parâmetros (milhões; antes da fusão)')
    axes[1].barh(names,f.mean_ms,xerr=f.std_ms); axes[1].set_xlabel('Latência API predict, 90 repetições (ms)')
    for task,arch,_ in MODELS:
        row=f[(f.task==task)&(f.model==arch)].iloc[0]
        accuracy=df[(df.task==task)&(df.model==arch)&(df.split=='test')&(df.metric_type==('mask' if task=='segmentation' else 'box'))].iloc[0].map50_95
        axes[2].scatter(row.median_ms,accuracy,label=task+' / '+arch,s=70)
    axes[2].set(xlabel='Latência mediana API predict (ms)',ylabel='mAP@0.5:0.95 da tarefa'); axes[2].legend(fontsize=10)
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle('Complexidade e compromisso entre qualidade e custo computacional'); fig.tight_layout(); fig.savefig(ROOT/'graficos/eficiencia.png',dpi=150); plt.close(fig)

def conclusions(df):
    paras=[]
    choices=[]
    bench=pd.read_csv(ROOT/'relatorios/benchmark_padronizado.csv').set_index(['task','model'])
    for task,kind,label in [('detection','box','detecção'),('segmentation','mask','segmentação de instâncias')]:
        f=df[(df.task==task)&(df.metric_type==kind)&(df.split=='test')].set_index('model')
        a,b=f.loc['YOLOv8'],f.loc['YOLO11']
        delta=b.map50_95-a.map50_95
        leader='YOLO11' if delta>0 else 'YOLOv8' if delta<0 else 'empate'
        faster=str(bench.loc[task].median_ms.idxmin())
        choices.append(f'para {label}, maior qualidade medida: {leader}; menor latência de API: {faster}')
        paras.append(f'Na {label}, o mAP@0.5:0.95 no teste foi {a.map50_95:.4f} para YOLOv8 e {b.map50_95:.4f} para YOLO11. A diferença YOLO11 − YOLOv8 foi {delta:+.4f}; a maior pontuação observada foi {leader}. Isso descreve esta execução e esta divisão, sem demonstrar superioridade estatística ou generalização para outras subestações. Precision/recall foram {a.precision:.4f}/{a.recall:.4f} e {b.precision:.4f}/{b.recall:.4f}, respectivamente.')
        ta,tb=bench.loc[(task,'YOLOv8')],bench.loc[(task,'YOLO11')]
        paras.append(f'No benchmark complementar da API predict em lote 1, a latência média foi {ta.mean_ms:.2f} ± {ta.std_ms:.2f} ms no YOLOv8 e {tb.mean_ms:.2f} ± {tb.std_ms:.2f} ms no YOLO11. As medianas foram {ta.median_ms:.2f} e {tb.median_ms:.2f} ms; os percentis 95 foram {ta.p95_ms:.2f} e {tb.p95_ms:.2f} ms. Foram dez aquecimentos e três rodadas de 30 repetições, sem remover valores atípicos. O ensaio inclui leitura, pré-processamento e pós-processamento da mesma imagem real de validação; difere do tempo interno do modelo medido sobre todo o teste. A quantidade de parâmetros foi {int(a.parameters):,} e {int(b.parameters):,}. Variações térmicas, carga do Windows e tamanho dos objetos afetam o desempenho observado; diferenças de mediana não demonstram separação estatística.')
    for task,arch,_ in MODELS:
        f=pd.read_csv(ROOT/'resultados'/task/arch/'results.csv').rename(columns=lambda s:s.strip())
        cols=[c for c in f if c.startswith('val/') and 'loss' in c]
        detail=[]
        relative_changes=[]
        train_detail=[f'{c}: {float(f[c].iloc[0]):.4f} → {float(f[c].iloc[-1]):.4f}' for c in f if c.startswith('train/') and 'loss' in c]
        for c in cols:
            initial=float(f[c].iloc[0]); final=float(f[c].iloc[-1]); early=float(f[c].iloc[-6:-3].mean()); late=float(f[c].iloc[-3:].mean())
            if abs(early)>1e-9:
                relative_changes.append(abs(late-early)/abs(early))
            detail.append(f'{c}: {initial:.4f} → {final:.4f}, variação das médias das últimas duas janelas de 3 épocas = {late-early:+.4f}')
        paras.append(f'{task}, {arch}. Perdas de treinamento: '+'; '.join(train_detail)+'. Perdas de validação: '+'; '.join(detail)+'. Vinte épocas constituem um orçamento fixo; redução de perda não prova convergência. A estabilidade das últimas janelas e as curvas de AP devem ser interpretadas juntas, especialmente após o encerramento do mosaico. Não foram realizados novos ajustes após consultar o teste.')
        ap_col='metrics/mAP50-95(M)' if task=='segmentation' else 'metrics/mAP50-95(B)'
        ap_delta=float(f[ap_col].iloc[-3:].mean()-f[ap_col].iloc[-6:-3].mean())
        stable=bool(relative_changes) and max(relative_changes)<.02 and abs(ap_delta)<.01
        verdict='Há sinal de estabilização no horizonte final, mas isso não demonstra convergência ao melhor desempenho possível.' if stable else 'As janelas finais ainda variam; as 20 épocas não demonstram estabilização suficiente por esse critério descritivo.'
        paras.append(f'{task}, {arch}: a variação da média de AP entre as duas últimas janelas de três épocas foi {ap_delta:+.4f}. {verdict} O critério descritivo adotado foi variação inferior a 2% em todos os componentes não nulos da perda de validação e inferior a 0,01 no AP; não é um teste estatístico nem foi usado para mudar o treinamento.')
    dist=pd.read_csv(ROOT/'relatorios/distribuicao_classes.csv').groupby('classe').objetos.sum().sort_values()
    paras.append(f'O desbalanceamento observado varia de {int(dist.iloc[0])} objetos em {dist.index[0]} a {int(dist.iloc[-1])} em {dist.index[-1]}, razão de {dist.iloc[-1]/max(dist.iloc[0],1):.1f}:1. O mAP macro atribui o mesmo peso às classes presentes; os CSVs por classe permitem identificar se classes raras tiveram baixo recall. Nenhuma classe foi removida por conveniência.')
    records=load_json(ROOT/'relatorios/anotacoes_convertidas.json')
    areas=[o['bbox'][2]*o['bbox'][3]*640**2 for r in records for o in r['objects']]
    frac=sum(a<32**2 for a in areas)/len(areas)
    paras.append(f'Como indicador geométrico, {frac:.1%} das caixas têm área inferior a 32² pixels após normalização para uma imagem 640×640. Isso é uma aproximação; o letterbox preserva a proporção e pode reduzir ainda mais a área útil. Objetos pequenos e oclusões dificultam localizar bordas e distinguir porcelanas próximas. Não foi calculado AP COCO por escala; esse indicador não demonstra sozinho uma queda de AP para objetos pequenos.')
    paras.append('As anotações foram publicadas para segmentação semântica. Aqui cada polígono sem group_id é uma instância operacional; polígonos com o mesmo group_id e classe permanecem agrupados. Fragmentos de um objeto físico sem identificador podem permanecer separados. Os registros de correções informam reparos, recortes, pontes entre partes e eventual preenchimento de furos, limitações que afetam a fidelidade das máscaras. As caixas são envoltórias dos mesmos objetos usados na segmentação.')
    paras.append('A divisão usa seed 42 e é idêntica nas duas tarefas, com agrupamento de duplicatas exatas por SHA256. Imagens próximas no tempo ou ponto de vista podem permanecer em conjuntos diferentes. O dataset contém uma única subestação, portanto a estimativa não representa validação externa. Houve apenas uma semente, sem intervalos de confiança e sem busca de hiperparâmetros; ganhos pequenos devem ser interpretados com cautela.')
    paras.append('As medições originais de latência mostraram dispersão, incluindo tempo interno de 42,27 ms na validação do detector YOLO11 e uma janela E2E do detector YOLOv8 com desvio próximo da própria média. Inicialização, aquecimento, compilação ou carga do sistema podem contribuir, mas a causa exata não foi isolada. Essas observações foram preservadas integralmente. Por isso, a análise de latência estável usa também o benchmark complementar em validação, com mediana e P95; nenhum peso, hiperparâmetro ou resultado de qualidade foi alterado após o teste.')
    paras.append('Para o cenário medido, as escolhas condicionais são: '+'; '.join(choices)+'. Se a prioridade for qualidade global, recomenda-se o maior mAP de teste da tarefa; se a prioridade for resposta rápida, a menor latência observada, verificando o recall das classes de interesse. A maior pontuação nesta execução é evidência limitada ao protocolo, insuficiente para demonstrar superioridade estatística ou uma arquitetura universalmente melhor. Inspeções críticas requerem validação por equipamento e em outras subestações antes de uso operacional.')
    return paras

def notebook(df,paras):
    cells=[]
    md=lambda text: cells.append(nbf.v4.new_markdown_cell(text))
    code=lambda text: cells.append(nbf.v4.new_code_cell(text))
    md(COVER)
    md('## 1. Objetivos\nComparar qualidade de localização, máscaras e custo computacional em condições equivalentes, usando resultados reais e preservando dados originais e checkpoints.')
    md('## 2. Introdução teórica\nYOLO é uma família de modelos de detecção de estágio único. A detecção estima classe e caixa; a segmentação de instâncias acrescenta uma máscara por objeto. Precision mede a fração de predições positivas corretas e recall a fração dos objetos recuperados. AP integra a curva precision–recall; mAP@0.5 usa IoU 0,5 e mAP@0.5:0.95 agrega dez limiares de 0,50 a 0,95. Caixas e máscaras têm IoU e métricas próprias. O ponto de precision/recall reportado pelo Ultralytics é escolhido na curva F1; o limiar visual 0,25 é usado apenas nas ilustrações.')
    md('## 3. Bibliotecas e ambiente computacional\nO ambiente existente foi preservado. As dependências completas estão fixadas em requirements.txt; CUDA não foi reinstalado.')
    code("from pathlib import Path\nimport json, sys\nimport pandas as pd\nfrom IPython.display import display, Image\npd.set_option('display.max_columns', None)\npd.set_option('display.max_colwidth', None)\nROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\nsys.path.insert(0, str(ROOT / 'src'))\nfrom common import load_json\ndisplay(pd.DataFrame([load_json(ROOT / 'relatorios/ambiente.json')]).T)")
    md('## 4. Descrição do dataset\nGomes et al. publicaram imagens de uma subestação brasileira com 15 classes de equipamentos. O ZIP e seu MD5 oficial foram verificados. As pastas de máscaras semânticas não são usadas para inventar instâncias. A tabela abaixo descreve o conteúdo efetivamente inspecionado.')
    code("prep = load_json(ROOT / 'relatorios/preparacao.json')\ndisplay(pd.DataFrame([(k, str(v)) for k,v in prep.items()], columns=['Propriedade','Valor']))")
    code("audit = load_json(ROOT/'relatorios/auditoria_fonte.json')\nprint(audit['note'])\nprint('Polígonos de equipamentos nos JSONs:',audit['raw_equipment_polygons'],'| instâncias após deduplicação:',audit['converted_instances'])\ndisplay(pd.read_csv(ROOT/'relatorios/auditoria_contagens.csv'))")
    code("masks = load_json(ROOT / 'relatorios/inspecao_mascaras.json')\nfor name,info in masks.items():\n    print(name, 'arquivos:',info['files'],'ausentes:',len(info['missing_for_images']),'órfãos:',len(info['orphan_masks']))\n    display(pd.DataFrame(info['samples']))")
    md('## 5. Exploração e distribuição das classes\nContagens após validação geométrica e agrupamento explícito de instâncias. O arquivo correcoes_descartes.json registra todas as alterações e exclusões.')
    code("dist = pd.read_csv(ROOT / 'relatorios/distribuicao_classes.csv')\ndisplay(dist.pivot(index='classe', columns='split', values='objetos'))\ndisplay(Image(filename=str(ROOT / 'graficos/distribuicao.png')))")
    md('## 6. Conversão das anotações\nOs JSONs reais são LabelMe: shapes contêm label, points, shape_type e group_id. O fundo não recebe ID YOLO. Para cada instância, a caixa é o mínimo retângulo envolvente; centro e dimensões são divididos por largura e altura. Os vértices dos polígonos são normalizados da mesma forma. Geometrias inválidas são reparadas, recortadas geometricamente na imagem e registradas. Partes da mesma instância são representadas com pontes de largura zero, pois o TXT YOLO não codifica MultiPolygon/furos. Veja o código integral no apêndice.')
    code("issues = load_json(ROOT / 'relatorios/correcoes_descartes.json')\ndisplay(pd.DataFrame(issues).head(20))\nprint('Total de registros de correção/descarte/limitação:', len(issues))")
    md('As dimensões consideradas são as orientadas pelo EXIF, iguais às usadas pelo decodificador OpenCV/Ultralytics e pelos JSONs. Os JPEGs não são recodificados. Duplicatas literais de classe e vértices são removidas nas duas tarefas, com registro. A primeira verificação foi arquivada em relatorios/preparacao_revisao_EXIF antes de qualquer treinamento; a versão final preserva todas as imagens dimensionalmente válidas.')
    md('## 7. Validação visual de três imagens\nCada figura apresenta original, caixas e polígonos anotados. Cores identificam classes; sobreposição transparente permite verificar contornos e oclusões. As figuras foram geradas das mesmas coordenadas gravadas nos TXT YOLO.')
    for i in range(1,4):
        for kind in ['original','boxes','polygons']:
            code(f"display(Image(filename=str(ROOT / 'graficos/validacao_{i}_{kind}.png')))")
    md('## 8. Divisão dos dados\n70% treino, 15% validação e 15% teste, seed 42, com arredondamento por imagem e agrupamento SHA256 de duplicatas exatas. Ambas as tarefas usam o mesmo manifesto anotacoes_convertidas.json. Nenhuma imagem de teste participa de treino, teste funcional ou escolha de hiperparâmetros. Correlação visual ou temporal entre imagens diferentes continua sendo uma limitação.')
    code("manifest = load_json(ROOT / 'relatorios/anotacoes_convertidas.json')\ndisplay(pd.Series([r['split'] for r in manifest]).value_counts().to_frame('Imagens'))\nsets = {s: {r['sha256'] for r in manifest if r['split']==s} for s in ['train','val','test']}\nassert all(not sets[a] & sets[b] for a,b in [('train','val'),('train','test'),('val','test')])\nprint('Verificação de duplicatas exatas entre conjuntos: aprovada')")
    md('## 9. Configuração dos experimentos\nTreinos sequenciais, checkpoints por época, SGD, resolução 640, seed 42, device 0, AMP quando aprovado no teste do Ultralytics, patience 0 e workers 0. Os demais parâmetros são os padrões da versão fixada e estão registrados em args.yaml. O batch efetivamente adotado e a eventual justificativa de OOM aparecem abaixo. O ensaio funcional usa uma amostra densa do treino por uma época e é armazenado separadamente.')
    code("from common import TRAIN, MODELS\ndisplay(pd.DataFrame([(k,str(v)) for k,v in TRAIN.items()],columns=['Parâmetro','Configuração inicial']))\nfor task,arch,_ in MODELS:\n    print(task, arch, load_json(ROOT / 'resultados' / task / arch / 'concluido.json'))\nfor p in (ROOT/'relatorios').glob('ajuste_batch_*.json'):\n    print(p.name, load_json(p))")
    code("import time\nstate = load_json(ROOT / 'relatorios/execucao.json')\nprint(state.get('time_scope',''))\nprint('Tempo corrido até esta análise (horas):', round((time.time()-state['started'])/3600, 3))\nfor task,arch,_ in MODELS:\n    timing = load_json(ROOT/'resultados'/task/arch/'tempos_epocas.json')\n    print(task,arch,'primeira época (s):',round(timing[0]['seconds'],2),'soma das épocas (s):',round(sum(t['seconds'] for t in timing),2))")
    md('## 10. Treinamento de detecção\nPesos iniciais: yolov8n.pt e yolo11n.pt. Os experimentos finais têm exatamente 20 linhas de métricas por época. best.pt é escolhido pela validação; last.pt permite retomada antes da finalização. Os treinos não são repetidos para gerar este documento.')
    code("for arch in ['YOLOv8','YOLO11']:\n    f = pd.read_csv(ROOT / 'resultados/detection' / arch / 'results.csv')\n    assert len(f)==20\n    print('Detecção',arch,'—',len(f),'épocas')\n    display(f.tail(3))")
    md('## 11. Treinamento de segmentação\nPesos iniciais: yolov8n-seg.pt e yolo11n-seg.pt. Máscaras e caixas têm métricas separadas. As perdas incluem os componentes box, seg, cls e dfl quando fornecidos pelo treinador. overlap_mask=False preserva máscaras-alvo independentes em regiões de sobreposição; mask_ratio=4 mantém a resolução padrão dos alvos nas duas arquiteturas.')
    md('A versão instalada também registra sem_loss. Nestes modelos, esse componente auxiliar permanece zero porque não há predição semântica auxiliar; ele não indica segmentação perfeita. A perda de instâncias relevante é seg_loss, em conjunto com as métricas de máscaras.')
    code("for arch in ['YOLOv8','YOLO11']:\n    f = pd.read_csv(ROOT / 'resultados/segmentation' / arch / 'results.csv')\n    assert len(f)==20\n    print('Segmentação',arch,'—',len(f),'épocas')\n    display(f.tail(3))")
    md('## 12. Avaliação na validação e no teste\nTodos os números abaixo provêm de best.pt. Validação e teste usam imgsz 640, CUDA, FP16, conf 0,001, IoU NMS 0,7 e max_det 300. O teste foi reservado para avaliação final. O tempo interno de inferência é média por imagem do validador. Um benchmark adicional usa lote 1, cinco aquecimentos e 30 repetições na mesma imagem de validação, com sincronização CUDA; a latência total inclui leitura e a API predict. A contagem de parâmetros é anterior à fusão Conv/BatchNorm.')
    code("metrics = pd.read_csv(ROOT / 'relatorios/metricas_comparativas.csv')\nfor s in ['val','test']:\n    print('Conjunto:', s)\n    display(metrics[metrics.split==s].round(5))")
    md('### Benchmark complementar de latência\nA dispersão das janelas originais motivou um ensaio adicional, sem refazer avaliações de qualidade ou treinos. Mesma imagem real de validação, lote 1, FP16 CUDA, dez aquecimentos e três rodadas de 30 repetições. Todas as 90 observações por modelo foram preservadas, sem excluir outliers. Média, desvio, mediana e P95 estão abaixo. As recomendações de velocidade usam a mediana deste ensaio; as medidas originais continuam nas tabelas anteriores.')
    code("display(pd.read_csv(ROOT/'relatorios/benchmark_padronizado.csv').round(4))\nprint(load_json(ROOT/'relatorios/benchmark_padronizado.json')['scope'])")
    md('## 13. Gráficos comparativos\nOs componentes das perdas aparecem em eixos separados para não somar escalas distintas. mAP de caixas e máscaras é separado. Parâmetros medem complexidade de armazenamento; não substituem latência ou FLOPs. A comparação de eficiência combina AP com a latência observada, sem atribuir uma única pontuação arbitrária.')
    for name in ['detection_train_loss','detection_val_loss','segmentation_train_loss','segmentation_val_loss','comparativo_map50','comparativo_map50_95','comparativo_inference_ms','eficiencia']:
        code(f"display(Image(filename=str(ROOT / 'graficos/{name}.png')))")
    md('## 14. Tabelas de resultados por classe\nAs tabelas agregadas da seção 12 são complementadas por AP e recall por classe, para revelar o efeito do desbalanceamento. Classes ausentes do conjunto avaliado não têm AP estimado.')
    code("for task,arch,_ in MODELS:\n    for kind in (['box','mask'] if task=='segmentation' else ['box']):\n        print(task,arch,kind,'teste')\n        display(pd.read_csv(ROOT/'resultados'/task/arch/'avaliacao_test'/f'classes_{kind}.csv').round(4))")
    md('## 15. Predições reais\nAs três primeiras imagens por nome no teste são mostradas sem seleção por qualidade; conf visual = 0,25. A segmentação mostra máscaras previstas, distintas das máscaras de verdade usadas na validação visual.')
    for task,arch,_ in MODELS:
        md(f'### {task} — {arch}')
        for i in range(1,4): code(f"display(Image(filename=str(ROOT/'resultados/{task}/{arch}/avaliacao_test/predicao_{i}.jpg'), width=1000))")
    md('## 16. Análise crítica\n\n'+'\n\n'.join(paras[:-1]))
    md('## 17. Conclusão\n\n'+paras[-1]+'\n\nOs quatro experimentos foram limitados a 20 épocas por exigência do protocolo. Nenhum resultado funcional foi incluído nas tabelas finais.')
    md('## 18. Referências\n- Gomes, A. et al. A Semantically Annotated 15-Class Ground Truth Dataset for Substation Equipment. Zenodo, 2023. https://doi.org/10.5281/zenodo.7884270\n- Ultralytics. YOLOv8. https://docs.ultralytics.com/models/yolov8/\n- Ultralytics. YOLO11. https://docs.ultralytics.com/models/yolo11/\n- Ultralytics. Train, Val e Segment: https://docs.ultralytics.com/modes/train/ ; https://docs.ultralytics.com/modes/val/ ; https://docs.ultralytics.com/tasks/segment/\n- PyTorch. Reproducibility. https://pytorch.org/docs/stable/notes/randomness.html\n\nVersões executadas estão no diagnóstico e no requirements.txt. Reprodutibilidade numérica exata entre hardware/versões não é garantida.')
    md('## Apêndice — reprodução e código-fonte\nExecute no PowerShell, na raiz do projeto:\n```powershell\n.\\.venv\\Scripts\\python.exe -u src/pipeline.py\n```\nPara reconstruir apenas a análise e o PDF, sem treinar:\n```powershell\n.\\.venv\\Scripts\\python.exe src/pipeline.py --report-only\n```\nO código a seguir é a implementação preservada no projeto, exibida para revisão.')
    for name in ['common.py','download.py','prepare.py','train.py','evaluate.py','benchmark.py','pipeline.py']:
        md('### '+name+'\n```python\n'+(ROOT/'src'/name).read_text(encoding='utf-8')+'\n```')
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python (.venv)','language':'python','name':'python3'}})
    NotebookClient(nb,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    path=ROOT/'notebooks/Prova_YOLO_Subestacoes.ipynb'
    nbf.write(nb,path)
    return nb,path

def pdf_fallback(nb,path):
    """Exporta todas as células e saídas, paginando código e tabelas largas."""
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Preformatted, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.utils import ImageReader
    font=Path('C:/Windows/Fonts/arial.ttf')
    mono=Path('C:/Windows/Fonts/consola.ttf')
    pdfmetrics.registerFont(TTFont('Body',str(font)))
    pdfmetrics.registerFont(TTFont('Code',str(mono)))
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle('BodyCustom',fontName='Body',fontSize=9,leading=13,spaceAfter=7))
    styles.add(ParagraphStyle('CodeCustom',fontName='Code',fontSize=7.5,leading=9.5,spaceAfter=5))
    story=[]
    def text(value,code=False):
        if code:
            lines=[]
            for line in value.expandtabs(4).splitlines():
                lines.extend(textwrap.wrap(line,width=94,replace_whitespace=False,drop_whitespace=False) or [''])
            # Pequenos blocos evitam células maiores que a altura da página.
            for start in range(0,len(lines),35):
                story.append(Preformatted('\n'.join(lines[start:start+35]),styles['CodeCustom']))
            return
        in_code=False; buffer=[]
        for line in value.splitlines():
            if line.startswith('```'):
                if in_code: text('\n'.join(buffer),True); buffer=[]
                in_code=not in_code; continue
            if in_code: buffer.append(line); continue
            if not line.strip(): story.append(Spacer(1,5)); continue
            heading=len(line)-len(line.lstrip('#')) if line.startswith('#') else 0
            clean=escape(line.lstrip('# ').replace('**','').replace('`',''))
            style=styles['Heading1' if heading<=1 else 'Heading2'] if heading else styles['BodyCustom']
            story.append(Paragraph(clean,style))
    for cell in nb.cells:
        if cell.cell_type=='markdown':
            text(cell.source)
        elif cell.cell_type=='code':
            text(cell.source,True)
            for out in cell.get('outputs',[]):
                if out.output_type=='stream': text(out.text,True)
                elif out.output_type=='error': raise RuntimeError('Notebook contém erro')
                else:
                    data=out.get('data',{})
                    if 'image/png' in data:
                        stream=io.BytesIO(base64.b64decode(data['image/png']))
                        w,h=ImageReader(stream).getSize(); ratio=min(480/w,620/h)
                        story.append(Image(stream,width=w*ratio,height=h*ratio)); story.append(Spacer(1,8))
                    elif 'image/jpeg' in data:
                        stream=io.BytesIO(base64.b64decode(data['image/jpeg']))
                        w,h=ImageReader(stream).getSize(); ratio=min(480/w,620/h)
                        story.append(Image(stream,width=w*ratio,height=h*ratio)); story.append(Spacer(1,8))
                    elif 'text/html' in data:
                        from bs4 import BeautifulSoup
                        soup=BeautifulSoup(data['text/html'],'html.parser')
                        table=soup.find('table')
                        if table:
                            rows=[[c.get_text(' ',strip=True) for c in r.find_all(['th','td'])] for r in table.find_all('tr')]
                            from reportlab.platypus import LongTable, TableStyle
                            # Divide colunas para garantir fonte legível em A4.
                            columns=max(map(len,rows))
                            for begin in range(1,columns,5):
                                segment=[[r[0]]+r[begin:begin+5] for r in rows]
                                formatted=[[Paragraph(escape(x),styles['CodeCustom']) for x in row] for row in segment]
                                t=LongTable(formatted,colWidths=[480/len(segment[0])]*len(segment[0]),repeatRows=1,hAlign='LEFT')
                                t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),.3,colors.lightgrey),('BACKGROUND',(0,0),(-1,0),colors.whitesmoke),('VALIGN',(0,0),(-1,-1),'TOP')]))
                                story.append(t); story.append(Spacer(1,7))
                        else: text(soup.get_text(' ',strip=True))
                    elif 'text/plain' in data: text(data['text/plain'],True)
    def page(canvas,doc):
        canvas.setFont('Body',8); canvas.drawString(45,25,'YOLO em subestações — comparação experimental / 20 épocas'); canvas.drawRightString(550,25,str(doc.page))
    doc=SimpleDocTemplate(str(path),pagesize=(595.28,841.89),rightMargin=50,leftMargin=50,topMargin=45,bottomMargin=45,title='Prova YOLO Subestações',author='Lucas Ferreira Neto')
    doc.build(story,onFirstPage=page,onLaterPages=page)

def report():
    pd.set_option('display.max_columns',None)
    df=pd.read_csv(ROOT/'relatorios/metricas_comparativas.csv')
    assert len(df)==12
    charts(df); paras=conclusions(df)
    (ROOT/'relatorios/analise_critica.md').write_text('\n\n'.join(paras),encoding='utf-8')
    nb,path=notebook(df,paras)
    from nbconvert import HTMLExporter
    html,_=HTMLExporter().from_notebook_node(nb)
    (ROOT/'entrega/Prova_YOLO_Subestacoes_Final.html').write_text(html,encoding='utf-8')
    pdf=ROOT/'entrega/Prova_YOLO_Subestacoes_Final.pdf'
    # ReportLab evita dependência de LaTeX/Chromium e preserva todas as saídas.
    pdf_fallback(nb,pdf)
    from pypdf import PdfReader
    reader=PdfReader(pdf)
    text='\n'.join(p.extract_text() or '' for p in reader.pages)
    assert len(reader.pages)>5 and len(text)>10000
    assert 'Conclusão' in text and 'Referências' in text
    save_json(ROOT/'relatorios/validacao_entrega.json',dict(notebook=str(path),pdf=str(pdf),pages=len(reader.pages),pdf_bytes=pdf.stat().st_size,
              code_cells=sum(c.cell_type=='code' for c in nb.cells),executed_cells=sum(c.cell_type=='code' and c.execution_count is not None for c in nb.cells),
              errors=sum(o.output_type=='error' for c in nb.cells if c.cell_type=='code' for o in c.outputs),pdf_text_characters=len(text)))
    print('Entregas geradas:',path,pdf,flush=True)

if __name__=='__main__':
    report()
