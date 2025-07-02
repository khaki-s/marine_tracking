import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import os
from ultralytics import YOLO
from collections import defaultdict

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

# ---------------- Heatmap Helper Functions ----------------
def heatmap(data, row_labels, col_labels, ax=None, cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.
    """
    if ax is None:
        ax = plt.gca()
    if cbar_kw is None:
        cbar_kw = {}
        
    im = ax.imshow(data, **kwargs)
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
        
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.1f}", textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    Annotate a heatmap.
    """
    if data is None:
        data = im.get_array()
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.
    
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)
    
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)
    
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)
    return texts

# ---------------- Main Processing ----------------
# For each video, compute the heatmap based on shrimp movement distance per grid cell.
for filename in os.listdir(video_directory):
    if not filename.lower().endswith(('.mp4', '.avi', '.mkv')):
        continue
        
    video_path = os.path.join(video_directory, filename)
    base_name = os.path.splitext(os.path.basename(filename))[0]
    output_pdf_path = os.path.join(out_directory, f"{base_name}_movement_heatmap.pdf")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        continue
    
    # Get video dimensions (assume 1920x1080 if not, system will adapt)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # These grid dimensions are for heatmap – we use fixed grid of (grid_cols x grid_rows)
    # Each cell represents an area of size: cell_w = w/16, cell_h = h/9
    cell_w = w / grid_cols
    cell_h = h / grid_rows
    
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
    
    # ---------------- Plot Annotated Heatmap ----------------
    # For grid labels,使用简单的列/行编号
    row_labels = [str(r) for r in range(grid_rows)]
    col_labels = [str(c) for c in range(grid_cols)]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im, cbar = heatmap(movement_grid, row_labels, col_labels, ax=ax, cmap="Blues", cbarlabel="Movement (mm)")
    annotate_heatmap(im, movement_grid, valfmt="{x:.1f}")
    plt.title(f"Shrimp Movement Density Heatmap: {base_name}\n(Grid: 16 Columns x 9 Rows)")
    plt.xlabel("Grid Column")
    plt.ylabel("Grid Row")
    plt.tight_layout()
    plt.savefig(output_pdf_path, bbox_inches='tight')
    plt.close()
    print(f" Heatmap saved to: {output_pdf_path}")
