import os
import sys

# 添加项目根目录和 deep_sort-master 到 Python 路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'deep_sort-master'))

# 打印 Python 的搜索路径
print("Python 搜索路径:")
for path in sys.path:
    print(path)

# 导入必要的模块
from ultralytics import YOLO
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from deep_sort import nn_matching
import numpy as np

def main():
    # 初始化 YOLO 模型
    model = YOLO('yolov8n.pt')  # 使用你需要的模型
    
    # 初始化 DeepSORT 跟踪器
    max_cosine_distance = 0.2  # 余弦距离阈值
    nn_budget = 100  # 特征库大小限制
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    
    # 你的代码逻辑...

if __name__ == "__main__":
    main() 