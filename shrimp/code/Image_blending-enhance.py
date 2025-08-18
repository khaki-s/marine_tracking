import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO
from collections import defaultdict
import re
# ---------------- Configuration ----------------
# Path settings
model = YOLO('../runs/detect/train28/weights/best.pt')
video_directory = "E:/hydrothermal/2017-2018/shrimp/test"
out_directory = "E:/hydrothermal/2017-2018/shrimp/heatmap"
os.makedirs(out_directory, exist_ok=True)

# Real-world shrimp width (mm); used for pixel-to-mm conversion
actual_shrimp_width_mm = 10     
smoothed_window_size = 20         # Smoothing window size for trajectories

# Define grid for heatmap: 16 columns x 9 rows (16:9 aspect ratio)
grid_cols = 16  
grid_rows = 9
pixel_w, pixel_h = 1920, 1080
# ---------------- Main Processing ----------------
# Analyze the "year/month" label for each video
video_files_by_month = defaultdict(list)

pattern = re.compile(r"SMOOVE-(\d{2}-\d{2})")  # Match years and months like 20-10

for filename in os.listdir(video_directory):
    if not filename.lower().endswith(('.mp4', '.mkv')):
        continue
    match = pattern.search(filename)
    if match:
        month_key = match.group(1)  # e.g., "20-10"
        video_files_by_month[month_key].append(filename)
# For each month, compute the heatmap based on shrimp movement distance per grid cell.
for month_key, file_list in video_files_by_month.items():
    monthly_movement_grid = np.zeros((grid_rows, grid_cols), dtype=float)
    valid_video_count = 0

    for filename in file_list:
        video_path = os.path.join(video_directory, filename)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open {video_path}")
            continue
        # Get video dimensions (assume 1920x1080 if not, system will adapt)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # These grid dimensions are for heatmap – we use fixed grid of (grid_cols x grid_rows)
        # Each cell represents an area of size: cell_w = w/16, cell_h = h/9
        heatmap_img = np.zeros((1080, 1920), dtype=np.float32)
        #pixel size
        cell_w = 1920 / 16
        cell_h = 1080 / 9
        
        # Dictionaries to collect shrimp trajectories and pixel-to-mm ratios per track
        shrimp_tracks = defaultdict(list)
        pixel_to_mm_ratios = {}
        
        # Process each frame to collect shrimp center positions and compute conversion ratios
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model.track(frame, persist=True)
            if results[0].boxes.id is None:
                continue
            
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.int().cpu().tolist()
            confs = results[0].boxes.conf.cpu().tolist()
            
            for box, cls, track_id, conf in zip(boxes, clss, ids, confs):
                if cls == 1 and conf >= 0.5:  # process only shrimp detections
                    x1, y1, x2, y2 = box
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    shrimp_tracks[track_id].append((cx, cy))
                    
                    # Use bounding box area for pixel-to-mm conversion:
                    bbox_area_px = (x2 - x1) * (y2 - y1)
                    bbox_size_px = np.sqrt(bbox_area_px)
                    if track_id not in pixel_to_mm_ratios and bbox_size_px > 0:
                        pixel_to_mm_ratios[track_id] = actual_shrimp_width_mm / bbox_size_px
                        
        cap.release()
        
        # Initialize a grid for movement accumulation; each cell holds total movement distance (mm)
        movement_grid = np.zeros((grid_rows, grid_cols), dtype=float)
        
        # For each track, perform smoothing and compute movement distance segments.
        for track_id, points in shrimp_tracks.items():
            if len(points) < smoothed_window_size:
                continue
            # Apply sliding window smoothing:
            smoothed = [np.mean(points[max(0, i - smoothed_window_size + 1):i + 1], axis=0)
                        for i in range(len(points))]
            smoothed = np.array(smoothed)
            
            # For each segment, compute distance (in mm) and the midpoint
            for i in range(1, len(smoothed)):
                pt_prev = smoothed[i - 1]
                pt_curr = smoothed[i]
                seg_dist_px = np.linalg.norm(pt_curr - pt_prev)
                ratio = pixel_to_mm_ratios.get(track_id, 0)
                seg_dist_mm = seg_dist_px * ratio
                
                # Determine midpoint and map to grid cell
                mid_pt = (pt_prev + pt_curr) / 2.0
                # grid col index: mid_pt[0] divided by cell width, grid row index: mid_pt[1] divided by cell height
                col_idx = int(mid_pt[0] / cell_w)
                row_idx = int(mid_pt[1] / cell_h)
                if 0 <= col_idx < grid_cols and 0 <= row_idx < grid_rows:
                    movement_grid[row_idx, col_idx] += seg_dist_mm
        monthly_movement_grid += movement_grid
        valid_video_count += 1
    # ---------------- Plot Annotated Heatmap ----------------
    if valid_video_count == 0:
        continue
    #Calculate the average monthly movement distance
    avg_movement_grid = monthly_movement_grid / valid_video_count
    # For grid labels
    row_labels = [str(r) for r in range(grid_rows)]
    col_labels = [str(c) for c in range(grid_cols)]
    
    # ---------------- Generate 1920×1080 colormap PNG without coordinates ----------------
    fig = plt.figure(figsize=(pixel_w/100, pixel_h/100), dpi=100)

    plt.imshow(avg_movement_grid, 
            cmap="Reds", 
            interpolation="bilinear",
            origin='upper',
            extent=[0, pixel_w, pixel_h, 0])
    plt.axis('off')
    plt.tight_layout(pad=0)

    out_png = os.path.join(out_directory, f"{month_key}_heatmap_1920x1080.png")
    plt.savefig(out_png, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()