from ultralytics import YOLO
from PIL import Image

# 加载预训练模型
model = YOLO("no_team_best.pt")

# 使用模型进行训练
model.train(data="/root/autodl-tmp/cv_project/datasets/soccer_teams/data.yaml", epochs=200)

model.val()
