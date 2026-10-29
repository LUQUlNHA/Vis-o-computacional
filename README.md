# YOLO em subestações elétricas

Comparação acadêmica local entre **YOLOv8n e YOLO11n**, em detecção e segmentação de instâncias. Cada um dos quatro experimentos finais usa **20 épocas**, resolução 640, SGD, lr0=0,01, seed 42, CUDA 0 e workers 0. Batch inicial 8; apenas OOM comprovado justifica batch 4 nas duas arquiteturas da mesma tarefa.

**Aluno:** Lucas Ferreira Neto · **Matrícula:** 223026 · **Instituição:** Centro Universitário Facens (UniFacens).

## Entrega e resultados

- [PDF final](entrega/Prova_YOLO_Subestacoes_Final.pdf), com 55 páginas.
- [Notebook executado](notebooks/Prova_YOLO_Subestacoes.ipynb), com 43 células de código executadas.
- [Métricas de validação e teste](relatorios/metricas_comparativas.csv).
- [Benchmark complementar de latência](relatorios/benchmark_padronizado.csv).

O repositório inclui código, configurações, gráficos, métricas, exemplos de predição e as entregas finais. O dataset, o ambiente virtual, os pesos e os backups permanecem locais e são excluídos pelo `.gitignore`. O PDF e o notebook preservam as saídas da execução original; publicar o projeto não exige refazer os treinamentos.

As instruções abaixo usam o ambiente já existente no computador do experimento. Em outro computador, configure seu próprio ambiente compatível com `requirements.txt` e CUDA disponível. Os YAMLs publicados registram os caminhos absolutos da execução original; `src/prepare.py` gera os caminhos para a raiz local ao preparar os dados. Os checkpoints originais não estão no repositório; as métricas e saídas finais podem ser consultadas diretamente, enquanto reproduzir as avaliações exige os checkpoints locais ou uma nova execução completa.

## Executar ou retomar

Abra o PowerShell nesta pasta e use o ambiente existente:

```powershell
.\.venv\Scripts\python.exe -u src/pipeline.py 2>&1 | Tee-Object -FilePath relatorios/pipeline_manual.log -Append
```

O pipeline verifica ambiente, baixa e valida o dataset original, converte os polígonos, executa quatro testes funcionais separados, treina sequencialmente os quatro modelos, avalia best.pt, gera gráficos, executa o notebook de análise e exporta o PDF. Outro pipeline simultâneo é bloqueado. Não utiliza CPU para treinar. Mantenha o computador ligado e conectado à alimentação durante a execução.

Um treino interrompido retoma de `weights/last.pt`. O estado do otimizador é preservado enquanto o experimento está em andamento. Resultados com `concluido.json` são reaproveitados, sem novo treino. Uma pasta sem checkpoint é arquivada com sufixo `_interrompido_DATA`, sem apagar seu conteúdo. Não altere arquivos de dados ou parâmetros entre uma execução e sua retomada.

Se o teste funcional apontar **CUDA out of memory**, preserve o log e ajuste a tarefa afetada antes de iniciar os experimentos finais:

```powershell
.\.venv\Scripts\python.exe src/set_batch.py segmentation --reason "CUDA OOM no teste funcional batch 8; ver log preservado"
.\.venv\Scripts\python.exe -u src/pipeline.py
```

O ajuste é aplicado aos dois modelos da tarefa. O script recusa alterar batch quando há checkpoints finais, evitando misturar configurações. Erros diferentes de OOM precisam ser diagnosticados; não reduza épocas ou resolução.

## Reconstruir a entrega sem treinar

Para consultar o progresso durante os experimentos:

```powershell
.\.venv\Scripts\python.exe src/status.py
```

Após as avaliações:

```powershell
.\.venv\Scripts\python.exe src/pipeline.py --report-only
```

Esse comando lê artefatos reais, executa apenas as células de análise e recria:

- `notebooks/Prova_YOLO_Subestacoes.ipynb`
- `entrega/Prova_YOLO_Subestacoes_Final.pdf`
- `entrega/Prova_YOLO_Subestacoes_Final.html`

O exportador PDF usa ReportLab, fontes do Windows, paginação de código e divisão de tabelas largas. Não depende de LaTeX ou de instalar um navegador. O HTML é uma alternativa adicional que pode ser impresso em PDF pelo navegador. `relatorios/validacao_entrega.json` registra páginas, tamanho, células executadas e erros. A extração de texto verifica a presença das seções finais; a revisão visual permanece necessária para avaliar estética e legibilidade.

## Dados e metodologia

Fonte: [Zenodo 7884270](https://doi.org/10.5281/zenodo.7884270), arquivo `substation-semantic-dataset.zip`, MD5 oficial `ca897cb85a3b944af6a1355e83530c32`.

- O ZIP, metadados e extração ficam em `dataset/original/` e são preservados.
- Download por intervalos retomáveis; partes são mantidas para evitar baixar novamente após falhas.
- Os JSONs reais usam LabelMe. `classes.txt` define a ordem das 15 classes; Background é excluído do índice YOLO.
- Cada polígono sem group_id é uma instância operacional anotada. Mesmo group_id e classe são agrupados. Máscaras semânticas agregadas não são convertidas em supostas instâncias.
- Caixas são envoltórias dos mesmos polígonos da segmentação. Coordenadas são normalizadas por largura/altura.
- Dimensões usam orientação EXIF, como o OpenCV/Ultralytics e os JSONs reais. As imagens permanecem byte a byte iguais aos originais. Polígonos literalmente duplicados (mesma classe e vértices) são removidos de ambas as tarefas e registrados.
- Shapely verifica e repara geometrias; o relatório registra alterações, recortes, exclusões, partes e furos. O formato YOLO não codifica furos ou MultiPolygon nativamente; contornos externos e pontes de largura zero são limitações documentadas.
- Divisão 70/15/15 com seed 42 e arredondamento. Grupos SHA256 de duplicatas exatas permanecem no mesmo conjunto. Detecção e segmentação compartilham exatamente o mesmo manifesto.
- Uma única subestação e imagens correlacionadas limitam a generalização. A divisão aleatória não garante independência de datas ou pontos de vista.
- Testes funcionais usam apenas imagens do treino, incluindo exemplos densos. Resultados funcionais ficam em `resultados/funcional`, sem entrar na comparação final.
- Cada treino usa o dataset completo de sua divisão. As opções efetivas constam em args.yaml, incluindo padrões da versão instalada.
- Segmentação usa `overlap_mask=False` e `mask_ratio=4` nas duas arquiteturas, preservando alvos de máscaras independentes mesmo quando os polígonos de equipamentos se sobrepõem.
- O melhor checkpoint é escolhido pela validação; o teste é avaliado apenas ao final, sem ajustar hiperparâmetros.
- Precision, recall e AP de caixas e máscaras são separados. Perdas de treino e validação são lidas de results.csv. Métricas por classe são exportadas em CSV.
- Inferência: tempo interno médio do validador e benchmark adicional da API predict com lote 1, cinco aquecimentos e 30 repetições sincronizadas. Não são medidas equivalentes.
- As janelas iniciais mostraram valores atípicos de latência. `src/benchmark.py` acrescenta um ensaio separado: mesma imagem real de validação, dez aquecimentos, três rodadas de 30 repetições, média/desvio/mediana/P95 e todas as observações em CSV. As avaliações originais são preservadas. A análise de velocidade usa também esse ensaio e não altera pesos, métricas de qualidade ou hiperparâmetros após o teste.
- Parâmetros são contados antes da fusão Conv/BatchNorm. A figura de eficiência mostra o compromisso AP/latência; não inventa FLOPs.

## Estrutura e evidências

`src/` contém diagnóstico, download, preparação, treino, avaliação, relatório e orquestrador. `config/` contém YAMLs e batch por tarefa. `requirements.txt` é o snapshot exato do ambiente existente, acrescido somente das bibliotecas auxiliares necessárias. Não reinstale PyTorch ou CUDA para executar este projeto.

`relatorios/ambiente.json`, `preparacao.json`, `anotacoes_convertidas.json`, `correcoes_descartes.json`, `distribuicao_classes.csv` e `metricas_comparativas.csv` permitem auditar o experimento. Cada modelo tem `args.yaml`, `weights/best.pt`, `weights/last.pt`, `results.csv`, `results.png`, `tempos_epocas.json`, `concluido.json` e avaliações por conjunto. Logs de diagnóstico e execução ficam em `relatorios/`.

Nenhuma métrica é simulada. O relatório final exige que os quatro treinos tenham 20 épocas e que as avaliações estejam disponíveis. Identificação acadêmica: Lucas Ferreira Neto, matrícula 223026, Centro Universitário Facens (UniFacens). Para atualizar apenas a capa e reexportar as saídas já executadas, sem executar células ou treinamentos, use `.\.venv\Scripts\python.exe src/update_cover.py`.
