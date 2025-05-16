import os
from ultralytics import YOLO

# 현재 스크립트 기준으로 경로 조립
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML_PATH = os.path.join(BASE_DIR, "facial_data_yolo11", "data.yaml")

model = YOLO("yolo11n.pt")

results = model.train(
    data=DATA_YAML_PATH,
    epochs=10,
    imgsz=640,
    batch=16,
    name="face_yolo11n_exp"
)
