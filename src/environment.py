"""Diagnóstico sem instalar ou modificar dependências."""
import platform
import shutil
import subprocess
import sys
from common import ROOT, save_json
import psutil
import torch
import ultralytics

def check():
    info = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                torch=torch.__version__, ultralytics=ultralytics.__version__,
                cuda=torch.version.cuda, cuda_available=torch.cuda.is_available(),
                disk_free_bytes=shutil.disk_usage(ROOT).free)
    info.update(ram_total_bytes=psutil.virtual_memory().total,ram_available_bytes=psutil.virtual_memory().available)
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        info.update(gpu=torch.cuda.get_device_name(0), vram_free_bytes=free, vram_total_bytes=total)
        from ultralytics import YOLO
        m = YOLO(str(ROOT/'yolov8n.pt')).to('cuda:0')
        info['model_device'] = str(next(m.model.parameters()).device)
        del m
        torch.cuda.empty_cache()
    save_json(ROOT/'relatorios/ambiente.json', info)
    freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (ROOT/'requirements.txt').write_text(freeze, encoding='utf-8')
    print(info, flush=True)
    return info

if __name__ == '__main__':
    check()
