import numpy as np
from ultralytics import YOLO

# ------------------ TrackEval Output Settings ------------------
TRACKEVAL_OUTPUT_DIR = './result19-4-12'  # 对应 template2.yaml 的 trackers_folder

# ------------------ Video Paths ------------------
model = YOLO('/home/jingyichu/app/ultralytics-8.3.67/runs/detect/train18/weights/best.pt')
video_path = "/home/jingyichu/app/ultralytics-8.3.67/eval/19-4-12-1min.mp4"

results = model.track(source=video_path,persist=True, stream=False,show = False,tracker = "crab-botsort-nogmc.yaml")
with open("crab-yolo-botsort-nogmc.txt", "w") as f:
    for frame_id, r in enumerate(results, start=1):
        # 确保有检测目标且包含追踪 ID
        if r.boxes is not None and r.boxes.id is not None:
            # 筛选类别为 0 的索引
            mask = (r.boxes.cls == 0) & (r.boxes.conf > 0.8)
            
            # 使用 mask 过滤出类别 0 的坐标、ID 和置信度
            boxes_xyxy = r.boxes.xyxy[mask].cpu().numpy()
            track_ids = r.boxes.id[mask].int().cpu().tolist()
            for box, tid in zip(boxes_xyxy, track_ids):
                x1, y1, x2, y2 = box
                bw, bh = x2 - x1, y2 - y1
                # 写入格式: frame,id,left,top,w,h,1,1,1
                line = f"{frame_id},{tid},{x1:.2f},{y1:.2f},{bw:.2f},{bh:.2f},1,1,1\n"
                f.write(line)

