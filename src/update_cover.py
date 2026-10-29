"""Atualiza só a capa; exporta as saídas existentes sem executar nenhuma célula."""
from collections import Counter
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import hashlib
import shutil
import nbformat
import pymupdf
from nbconvert import HTMLExporter
from common import ROOT, load_json, save_json
from report import COVER, pdf_fallback

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def images(path):
    """Compara os bytes de pixels de todas as imagens incorporadas ao PDF."""
    with pymupdf.open(path) as doc:
        xrefs={entry[0] for page in doc for entry in page.get_images(full=True)}
        return Counter(hashlib.sha256(doc.xref_stream(xref)).hexdigest() for xref in xrefs)

def main():
    notebook=ROOT/'notebooks/Prova_YOLO_Subestacoes.ipynb'
    pdf=ROOT/'entrega/Prova_YOLO_Subestacoes_Final.pdf'
    html=ROOT/'entrega/Prova_YOLO_Subestacoes_Final.html'
    archive=ROOT/'entrega/versoes'/('antes_identificacao_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    archive.mkdir(parents=True)
    for path in [notebook,pdf,html]:
        shutil.copy2(path,archive/path.name)
    old=nbformat.read(notebook,as_version=4)
    original=deepcopy(old)
    protected=set()
    for folder in ['resultados','graficos']:
        protected.update(p for p in (ROOT/folder).rglob('*') if p.is_file())
    protected.update((ROOT/'relatorios').glob('*.csv'))
    protected.update((ROOT/'relatorios').glob('*.md'))
    protected.update(p for p in (ROOT/'relatorios').glob('*.json') if p.name not in {'validacao_entrega.json','validacao_identificacao.json'})
    protected.update((ROOT/'config').glob('*.yaml'))
    protected.update(ROOT.glob('*.pt'))
    before={str(p.relative_to(ROOT)):digest(p) for p in sorted(protected)}
    assert old.cells[0].cell_type=='markdown'
    old.cells[0].source=COVER
    assert old.cells[1:]==original.cells[1:]
    assert old.metadata==original.metadata
    nbformat.validate(old)
    nbtemp=notebook.with_name(notebook.stem+'_reexportacao.ipynb')
    pdftemp=pdf.with_name(pdf.stem+'_reexportacao.pdf')
    htmltemp=html.with_name(html.stem+'_reexportacao.html')
    nbformat.write(old,nbtemp)
    body,_=HTMLExporter().from_notebook_node(old)
    htmltemp.write_text(body,encoding='utf-8')
    pdf_fallback(old,pdftemp)
    reloaded=nbformat.read(nbtemp,as_version=4)
    assert reloaded==old
    code=[c for c in reloaded.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in code)
    assert not any(o.output_type=='error' for c in code for o in c.outputs)
    assert images(pdftemp)==images(archive/pdf.name), 'Imagens incorporadas diferem'
    with pymupdf.open(pdftemp) as doc:
        page_count=len(doc)
        first=doc[0].get_text()
        for value in ['Lucas Ferreira Neto','223026','Centro Universitário Facens (UniFacens)']:
            assert value in first, 'Identificação ausente na primeira página: '+value
        assert doc.metadata['author']=='Lucas Ferreira Neto'
        alltext='\n'.join(p.get_text() for p in doc)
        for title in ['12. Avaliação na validação e no teste','13. Gráficos comparativos','17. Conclusão','18. Referências']:
            assert title in alltext, 'Seção ausente: '+title
        assert not [i+1 for i,p in enumerate(doc) if len(p.get_text().strip())<70 and not p.get_images()]
        review={title:next(i for i,p in enumerate(doc) if title in p.get_text()) for title in ['12. Avaliação','13. Gráficos','17. Conclusão']}
        for label,index in [('capa',0)]+[(title.split('.')[0],index) for title,index in review.items()]:
            doc[index].get_pixmap(matrix=pymupdf.Matrix(1.3,1.3)).save(ROOT/'relatorios'/f'identificacao_revisao_{label}.png')
        pdf_text_chars=len(alltext)
        image_count=sum(len(p.get_images()) for p in doc)
    after={str(p.relative_to(ROOT)):digest(p) for p in sorted(protected)}
    assert before==after, 'Artefatos experimentais foram alterados'
    nbtemp.replace(notebook)
    pdftemp.replace(pdf)
    htmltemp.replace(html)
    save_json(ROOT/'relatorios/validacao_entrega.json',dict(notebook=str(notebook),pdf=str(pdf),pages=page_count,pdf_bytes=pdf.stat().st_size,
              code_cells=len(code),executed_cells=len(code),errors=0,pdf_text_characters=pdf_text_chars,
              identification=dict(aluno='Lucas Ferreira Neto',matricula='223026',instituicao='Centro Universitário Facens (UniFacens)')))
    save_json(ROOT/'relatorios/validacao_identificacao.json',dict(status='aprovado',backup=str(archive),
              only_notebook_change='source da primeira célula markdown; todas as demais células e metadados idênticos',
              cells_executed_during_update=0,training_runs_during_update=0,executed_cells_preserved=len(code),
              protected_files=len(before),protected_sha256=before,pdf_pages=page_count,pdf_image_placements=image_count,
              pdf_images_identical=True,sections_verified=list(review),blank_pages=[],pdf_sha256=digest(pdf),
              notebook_sha256=digest(notebook)))
    print('Capa atualizada; células e resultados preservados. PDF:',page_count,'páginas;',image_count,'imagens;',len(before),'arquivos protegidos.')

if __name__=='__main__':
    main()
