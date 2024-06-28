import os
from ultralytics import YOLO
from PIL import Image
import glob

def run_yolo(yolo, image_path, conf=0.25, iou=0.7):
    results = yolo(image_path, conf=conf, iou=iou)
    res = results[0].plot()[:, :, [2, 1, 0]]
    return Image.fromarray(res)

# 加载训练好的模型
yolo = YOLO('/root/autodl-tmp/runs/detect/train8/weights/best.pt')

# 验证图片路径
valid_image_path = "/root/autodl-tmp/datasets/soccer/valid/images"

# 结果保存路径
result_folder = "results"
os.makedirs(result_folder, exist_ok=True)

# 获取所有验证图片文件
image_files = glob.glob(os.path.join(valid_image_path, "*.jpg"))

# 遍历每张图片进行目标检测并保存结果
for image_file in image_files:
    predicted_image = run_yolo(yolo, image_file)
    
    # 构建结果文件路径
    result_file_path = os.path.join(result_folder, os.path.basename(image_file))
    
    # 保存预测结果
    predicted_image.save(result_file_path)

    print(f"Saved detected result to {result_file_path}")
