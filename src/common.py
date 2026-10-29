"""Configuração compartilhada; caminhos independentes do diretório do terminal."""
from pathlib import Path
import json
import os
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'config/matplotlib'))
for key,folder in [('IPYTHONDIR','ipython'),('JUPYTER_CONFIG_DIR','jupyter'),('JUPYTER_DATA_DIR','jupyter_data'),('JUPYTER_RUNTIME_DIR','jupyter_runtime')]:
    os.environ.setdefault(key,str(ROOT/'config'/folder))
    Path(os.environ[key]).mkdir(parents=True,exist_ok=True)
MODELS = [('detection', 'YOLOv8', 'yolov8n.pt'),
          ('detection', 'YOLO11', 'yolo11n.pt'),
          ('segmentation', 'YOLOv8', 'yolov8n-seg.pt'),
          ('segmentation', 'YOLO11', 'yolo11n-seg.pt')]
TRAIN = dict(epochs=20, imgsz=640, batch=8, optimizer='SGD', lr0=0.01,
             seed=42, device=0, amp=True, patience=0, workers=0,
             deterministic=True, cache=False, save=True, plots=True,
             overlap_mask=False, mask_ratio=4)

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,default=str)
    temp.replace(path)

def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)
