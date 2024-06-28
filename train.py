from ultralytics import YOLO
from PIL import Image

# 加载预训练模型
model = YOLO("yolov8n.pt")

# 使用模型进行训练
model.train(data="/root/autodl-tmp/datasets/soccer/data.yaml", epochs=3)

model.val()

# 保存训练好的模型
model.save("yolov8n_trained.pt")