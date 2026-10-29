"""Download retomável por intervalos, checksum oficial e extração segura."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import time
import zipfile
import requests
from common import ROOT, load_json, log

def download():
    base = ROOT/'dataset/original'
    base.mkdir(parents=True, exist_ok=True)
    meta = base/'zenodo_record.json'
    if not meta.exists():
        r = requests.get('https://zenodo.org/api/records/7884270', timeout=120)
        r.raise_for_status()
        meta.write_text(r.text, encoding='utf-8')
    f = next(f for f in load_json(meta)['files'] if f['key']=='substation-semantic-dataset.zip')
    target = base/f['key']
    if not target.exists():
        part = target.with_suffix('.part')
        offset = part.stat().st_size if part.exists() else 0
        chunks = base/'download_parts'
        chunks.mkdir(exist_ok=True)
        size = 16*1024*1024
        spans = [(a,min(a+size,f['size'])-1) for a in range(offset,f['size'],size)]
        def fetch(span):
            a,b = span
            path = chunks/f'{a}-{b}'
            if path.exists() and path.stat().st_size == b-a+1:
                return path
            for attempt in range(6):
                try:
                    with requests.get(f['links']['self']+f'?range={a}', headers={'Range':f'bytes={a}-{b}'}, stream=True, timeout=120) as r:
                        r.raise_for_status()
                        if r.status_code != 206 or not r.headers.get('Content-Range','').startswith(f'bytes {a}-{b}/'):
                            raise ValueError('Resposta Range incompatível')
                        with path.with_suffix('.tmp').open('wb') as out:
                            for block in r.iter_content(1024*1024):
                                out.write(block)
                    if path.with_suffix('.tmp').stat().st_size != b-a+1:
                        raise ValueError('Intervalo incompleto')
                    path.with_suffix('.tmp').replace(path)
                    log(f'Download {a/1e6:.0f}–{b/1e6:.0f} MB concluído')
                    return path
                except Exception:
                    if attempt == 5:
                        raise
                    time.sleep(2**attempt)
        with ThreadPoolExecutor(max_workers=6) as pool:
            paths = list(pool.map(fetch,spans))
        with part.open('ab') as out:
            for p in paths:
                with p.open('rb') as inp:
                    while block := inp.read(8*1024*1024):
                        out.write(block)
        part.rename(target)
    expected = f['checksum'].split(':')[1]
    with target.open('rb') as stream:
        actual = hashlib.file_digest(stream,'md5').hexdigest()
    if actual != expected:
        raise ValueError(f'Checksum incorreto: {actual} != {expected}; arquivo preservado')
    log('MD5 original validado: '+actual)
    extracted = base/'extracted'
    if not (extracted/'.complete').exists():
        extracted.mkdir(exist_ok=True)
        with zipfile.ZipFile(target) as z:
            for info in z.infolist():
                dest = (extracted/info.filename).resolve()
                if not dest.is_relative_to(extracted.resolve()):
                    raise ValueError('Caminho inseguro no ZIP')
                z.extract(info,extracted)
        (extracted/'.complete').write_text(actual)
    return extracted

if __name__=='__main__':
    download()
