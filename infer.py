import os
from ultralytics import YOLO
from PIL import Image

def run_yolo(yolo, image_path, conf=0.25, iou=0.7):
    results = yolo(image_path, conf=conf, iou=iou)
    res = results[0].plot()[:, :, [2, 1, 0]]
    return Image.fromarray(res)

yolo = YOLO('/root/autodl-tmp/cv_project/runs/detect/train3/weights/best.pt')

image = "offside1.jpeg"
result_path = os.path.join("/root/autodl-tmp/cv_project/results", "predicted.jpg")
predicted_image = run_yolo(yolo, image)
predicted_image.save(result_path)