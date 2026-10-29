"""Reconcilia contagens dos JSONs reais com os metadados publicados."""
from collections import Counter
import pandas as pd
from common import ROOT,load_json,save_json

def audit():
    counts=Counter()
    for path in (ROOT/'dataset/original/extracted/labels_json').glob('*.json'):
        counts.update(s['label'].strip() for s in load_json(path)['shapes'])
    prep=load_json(ROOT/'relatorios/preparacao.json')
    actual=pd.read_csv(ROOT/'relatorios/distribuicao_classes.csv').groupby('classe').objetos.sum()
    # Contagens declaradas na tabela do registro oficial; não são métricas experimentais.
    expected=[310,5243,1599,966,980,355,3185,26499,1354,1976,2331,768,2136,654,2349]
    rows=[dict(classe=c,publicado_zenodo=e,poligonos_json=counts[c],instancias_convertidas=int(actual[c]),diferenca_json_publicado=counts[c]-e) for c,e in zip(prep['classes'],expected)]
    pd.DataFrame(rows).to_csv(ROOT/'relatorios/auditoria_contagens.csv',index=False)
    report=dict(raw_counts=dict(counts),raw_equipment_polygons=sum(counts[c] for c in prep['classes']),
                background_polygons=counts['Background'],converted_instances=prep['objects'],
                published_total=sum(expected),note='As contagens do arquivo original diferem da descrição do Zenodo. Foram usados os JSONs reais, sem criar objetos para forçar correspondência com os metadados.')
    save_json(ROOT/'relatorios/auditoria_fonte.json',report)
    print(report,flush=True)

if __name__=='__main__':
    audit()
