from ultralytics import YOLO
from PIL import Image

# 加载预训练模型
model = YOLO("yolov8n.pt")

# 使用模型进行训练
model.train(data="/root/autodl-tmp/cv_project/datasets/soccer/data.yaml", epochs=20)

model.val()
