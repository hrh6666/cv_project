import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans

# 加载训练好的 YOLO 模型
model = YOLO('no_team_best.pt')

def detect_players(image_path):
    """
    使用 YOLO 模型检测图像中的球员、门将和裁判
    """
    results = model(image_path)
    detections = results[0]  # 获取第一个图像的检测结果
    return detections

def extract_colors(image, detections, class_names, target_classes=['soccer-player']):
    """
    提取每个检测框内的平均颜色
    """
    colors = []
    for detection in detections.boxes:
        cls = class_names[int(detection.cls)]
        if cls in target_classes:
            xyxy = detection.xyxy.cpu().numpy().astype(int).flatten()
            x1, y1, x2, y2 = xyxy  # 将坐标转换为整数
            player_img = image[y1:y2, x1:x2]
            player_img_rgb = cv2.cvtColor(player_img, cv2.COLOR_BGR2RGB)
            avg_color = player_img_rgb.mean(axis=0).mean(axis=0)  # 计算平均颜色
            colors.append(avg_color)
    return np.array(colors)

def cluster_colors(colors, n_clusters=2):
    """
    使用 K-means 聚类算法对提取的颜色进行聚类，返回聚类标签和聚类中心
    """
    if len(colors) < n_clusters:
        raise ValueError(f"n_samples={len(colors)} should be >= n_clusters={n_clusters}.")
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(colors)
    return kmeans.labels_, kmeans.cluster_centers_

def assign_teams(image, detections, labels, class_names, target_classes=['soccer-player']):
    """
    根据聚类结果为每个检测到的球员分配队伍，返回包含检测框坐标和队伍标签的列表
    """
    team_assignments = []
    label_index = 0
    for detection in detections.boxes:
        cls = class_names[int(detection.cls)]
        if cls in target_classes:
            xyxy = detection.xyxy.cpu().numpy().astype(int).flatten()
            x1, y1, x2, y2 = xyxy  # 将坐标转换为整数
            team_label = labels[label_index]
            label_index += 1
            team_assignments.append((x1, y1, x2, y2, team_label))
    return team_assignments

def save_results(image, team_assignments, output_path):
    """
    在图像上绘制检测结果和队伍分配结果，并将其保存到磁盘
    """
    for (x1, y1, x2, y2, team_label) in team_assignments:
        color = (0, 255, 0) if team_label == 0 else (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f'Team {team_label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.imwrite(output_path, image)

# 读取图像并运行检测和聚类
image_path = "/root/autodl-tmp/cv_project/offside1.jpeg"
output_path = "/root/autodl-tmp/cv_project/results/offside1_annotated.jpeg"
image = cv2.imread(image_path)
detections = detect_players(image_path)
colors = extract_colors(image, detections, model.names)

# 检查提取的颜色数量，并进行聚类
if len(colors) >= 2:
    labels, cluster_centers = cluster_colors(colors)
    team_assignments = assign_teams(image, detections, labels, model.names)
    save_results(image, team_assignments, output_path)
    print(f"Results saved to {output_path}")
else:
    print(f"Not enough samples to cluster: found {len(colors)} colors, but need at least 2.")