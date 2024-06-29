import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取彩色图像
image = cv2.imread('offside1.jpeg')

# 检查图像是否成功加载
if image is None:
    print("Error: Could not open or find the image.")
    exit()

# 提取绿色通道
image = cv2.GaussianBlur(image, (5, 5), 0.1)
green_channel = image[:, :, 0]

# 计算绿色通道的中位数
median_value = np.median(green_channel)

# 创建二值化图像
binary_image = np.zeros_like(green_channel)
binary_image[green_channel < median_value] = 255  # 小于中位数的点变成白色
binary_image[green_channel >= median_value] = 0   # 大于或等于中位数的点变成黑色

# 使用Canny边缘检测提取边缘
edges = cv2.Canny(binary_image, 50, 150)

# 使用霍夫变换检测直线
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=60, maxLineGap=10)

def calculate_slope(line):
    x1, y1, x2, y2 = line[0]
    if x2 - x1 == 0:
        return float('inf')  # 处理垂直线
    return abs((y2 - y1) / (x2 - x1))

# 选择最有可能的两条直线
line_image = np.copy(image)
if lines is not None:
    # 计算每条直线的长度
    lines_with_lengths = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        #length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        slope = calculate_slope(line)
        lines_with_lengths.append((line[0], slope))
    
    # 按长度排序并选择两条最长的直线
    lines_with_lengths.sort(key=lambda x: x[1], reverse=True)
    best_lines = [lines_with_lengths[0], lines_with_lengths[0], lines_with_lengths[0]]
    index = count = 1
    while count < 3:
        for i in range(index, len(lines_with_lengths)):
            if i == len(lines_with_lengths) - 1:
                count = 3
                break
            if abs(best_lines[count - 1][1] - lines_with_lengths[i][1]) < 1:
                continue
            best_lines[count] = lines_with_lengths[i]
            index = i + 1
            count += 1
            break

    # 在原始图像上绘制最有可能的两条直线
    line_image = np.copy(image)
    for line, _ in best_lines:
        x1, y1, x2, y2 = line
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 绘制红色的直线

# 使用Matplotlib显示原始图像、二值化图像、边缘图像和带有检测到直线的图像
plt.figure(figsize=(20, 10))

plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(2, 2, 2)
plt.title('Binary Image')
plt.imshow(binary_image, cmap='gray')

plt.subplot(2, 2, 3)
plt.title('Edges')
plt.imshow(edges, cmap='gray')

plt.subplot(2, 2, 4)
plt.title('Detected Lines')
plt.imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))

plt.show()
