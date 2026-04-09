import cv2
from ultralytics import YOLO 
from collections import defaultdict
import numpy as np
import os

def save_tracking_results(frame_id, track_id, bbox, conf, output):
    """
    Save tracking results to file in TrackEval format
    Format: <frame id>,<object id>,<top-left-x>,<top-left-y>,<w>,<h>,<confidence score>,-1,...
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    line = f"{frame_id},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,1,1\n"
    output.write(line)

# Shrimp parameters
actual_shrimp_width_mm = 10
capture_number = 0
shrimp_number = 0
smoothed_window_size = 10

# Model initialization
model = YOLO('D:/khaki/ultralytics-8.3.27/runs/detect/train32/weights/best.pt')

# Video path
video_path = "E:/graduate/hydrothermal/eval/shrimp/17-7-23-short.mp4"

# Open video and output file
capture = cv2.VideoCapture(video_path)
track_txt_path = "shrimp-yolo-bytetrack.txt"   
track_file = open(track_txt_path, "w")
assert capture.isOpened(), 'error reading the video'

# Read video properties
w, h, fps = (int(capture.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Trajectory Storage
raw_tracks = defaultdict(lambda: [])
smoothed_tracks = defaultdict(lambda: [])

# Distance tracking
total_distances = defaultdict(float)
all_crab_distance = 0
max_trail_length = 80

# Pixel to mm conversion for each track
pixel_to_mm_ratios = {}

frame_id = 1

while capture.isOpened():
    success, img = capture.read()
    if not success:
        print('read complete')
        break
    
    shrimp_number = 0
    capture_number = 0
    
    results = model.track(
        img,
        persist=True,
        tracker="shrimp-bytetrack.yaml",  
        iou=0.5,
        conf=0.1,
        verbose=False
    )
    
    # Process results
    if results[0].boxes.id is not None:
        boxs = results[0].boxes.xyxy.cpu().tolist()  
        track_ids = results[0].boxes.id.int().cpu().tolist()
        clss = results[0].boxes.cls.int().cpu().tolist()
        confs = results[0].boxes.conf.cpu().tolist()
        
        shrimp_number += clss.count(1)
        capture_number += 1
        
        # Process each detection
        for box, track_id, cls, conf in zip(boxs, track_ids, clss, confs):
            if cls == 1:  # 0: mussel, 1: shrimp
    
                save_tracking_results(
                    frame_id=frame_id,
                    track_id=track_id,
                    bbox=box,
                    conf=conf,
                    output=track_file
                )
            x1, y1, x2, y2 = box
            bbox_width_px = x2 - x1
            
            # Calculate pixel to mm conversion (per track)
            if track_id not in pixel_to_mm_ratios:
                if bbox_width_px > 0:
                    pixel_to_mm_ratios[track_id] = actual_shrimp_width_mm / bbox_width_px
                else:
                    pixel_to_mm_ratios[track_id] = 0
            
            # Calculate center coordinates
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            raw_tracks[track_id].append((center_x, center_y))
            
            # Smooth processing - average over window
            if len(raw_tracks[track_id]) >= smoothed_window_size:
                smoothed = np.mean(raw_tracks[track_id][-smoothed_window_size:], axis=0)
            else:
                smoothed = (center_x, center_y)
            
            smoothed_tracks[track_id].append(smoothed)
            
            # Limit trajectory length
            if len(smoothed_tracks[track_id]) > max_trail_length:
                raw_tracks[track_id].pop(0)
            
            # Calculate distance between consecutive smoothed points
            if len(smoothed_tracks[track_id]) >= 2:
                curr = smoothed_tracks[track_id][-1]
                prev = smoothed_tracks[track_id][-2]
                
                distance_px = np.sqrt((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)
                distance_mm = distance_px * pixel_to_mm_ratios[track_id]
                
                # Filter unrealistic movements 
                if distance_mm > 1 and distance_mm <= 5:
                    total_distances[track_id] += distance_mm
                    all_crab_distance += distance_mm
            
            # Visualization
            # label = f'ID:{track_id}, Conf:{conf:.2f}, Dist:{total_distances[track_id]:.1f}mm'
            # cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (203, 227, 48), thickness=2)
            # cv2.putText(img, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (203, 227, 48), thickness=2)
            # cv2.circle(img, (int(center_x), int(center_y)), 5, (203, 227, 48), -1)
            # points = np.array(smoothed_tracks[track_id], dtype=np.int32).reshape((-1, 1, 2))
            # cv2.polylines(img, [points], False, (203, 227, 48), thickness=2)

    frame_id += 1

# Calculate average distance per shrimp
if shrimp_number:
    avg_shrimp_permin_distance = (all_crab_distance / shrimp_number)
else:
    avg_shrimp_permin_distance = 0

# Sort distances in reverse order
sorted_distance = sorted(total_distances.items(), key=lambda x: x[1], reverse=True)

# Release resources
capture.release()
track_file.close()
cv2.destroyAllWindows()

