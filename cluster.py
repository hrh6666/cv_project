import cv2
import numpy as np
import os
from ultralytics import YOLO
from sklearn.cluster import DBSCAN, KMeans

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
            ratio = 0.25
            new_y1 = int((1 - ratio / 2) * y1 + (ratio / 2) * y2)
            new_y2 = int(ratio * y1 + (1 - ratio) * y2)
            new_x1 = int((1 - ratio) * x1 + ratio * x2)
            new_x2 = int(ratio * x1 + (1 - ratio) * x2)
            player_img = image[ new_y1:new_y2, new_x1:new_x2]
            player_img_rgb = cv2.cvtColor(player_img, cv2.COLOR_BGR2RGB)
            avg_color = player_img_rgb[:, :, [0, 1, 2]].mean(axis=0).mean(axis=0)  # 计算红色和蓝色通道的平均颜色
            colors.append(avg_color)
    return np.array(colors)

def cluster_colors_dbscan(colors, eps=40, min_samples=2):
    """
    使用 DBSCAN 聚类算法对提取的颜色进行聚类，返回聚类标签
    """
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(colors)
    return db.labels_

def remove_noise_and_cluster_kmeans(colors, labels, n_clusters=2):
    """
    去除噪声点后使用 K-means 聚类
    """
    # 去除噪声点
    filtered_colors = colors[labels != -1]
    if len(filtered_colors) < n_clusters:
        raise ValueError(f"Not enough samples to cluster after noise removal: {len(filtered_colors)} samples, but need at least {n_clusters}.")
    
    # 使用 K-means 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(filtered_colors)
    return kmeans.labels_, kmeans.cluster_centers_

def assign_teams(image, detections, labels, db_labels, class_names, target_classes=['soccer-player']):
    """
    根据聚类结果为每个检测到的球员分配队伍，返回包含检测框坐标和队伍标签的列表
    """
    team_assignments = []
    label_index = 0
    kmeans_label_index = 0  # K-means 标签索引
    for detection in detections.boxes:
        cls = class_names[int(detection.cls)]
        if cls in target_classes:
            xyxy = detection.xyxy.cpu().numpy().astype(int).flatten()
            x1, y1, x2, y2 = xyxy  # 将坐标转换为整数
            if db_labels[label_index] != -1:  # 排除噪声点
                team_label = labels[kmeans_label_index]  # 使用 K-means 标签
                team_assignments.append((x1, y1, x2, y2, team_label))
                kmeans_label_index += 1  # 仅在非噪声点时递增
            label_index += 1
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

# 遍历 offside_images 文件夹中的所有图片
input_folder = "/root/autodl-tmp/cv_project/offside_images"
output_folder = "/root/autodl-tmp/cv_project/results/cluster"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        image_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        image = cv2.imread(image_path)
        detections = detect_players(image_path)
        colors = extract_colors(image, detections, model.names)

        # 使用 DBSCAN 识别噪声点并去除噪声点
        if len(colors) >= 2:
            db_labels = cluster_colors_dbscan(colors)
            try:
                kmeans_labels, cluster_centers = remove_noise_and_cluster_kmeans(colors, db_labels)
                team_assignments = assign_teams(image, detections, kmeans_labels, db_labels, model.names)
                save_results(image, team_assignments, output_path)
                print(f"Results saved to {output_path}")
            except ValueError as e:
                print(f"Error processing {filename}: {e}")
        else:
            print(f"Not enough samples to cluster: found {len(colors)} colors in {filename}, but need at least 2.")