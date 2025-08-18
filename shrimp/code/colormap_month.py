import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO
from collections import defaultdict
import re
from scipy import stats
import math
import pandas as pd
import seaborn as sns
"""
    This code implements traversing the video group, 
    finding the average output of a heatmap for videos of the same month, 
    and unifying the vertical coordinates of all months within the video group
"""
# ---------------- Configuration ----------------
# Path settings
model = YOLO('../../runs/detect/train28/weights/best.pt')
video_directory = "E:/hydrothermal/short_test"
out_directory = "E:/hydrothermal/short_test/heatmap"
os.makedirs(out_directory, exist_ok=True)

# Real-world shrimp width (mm); used for pixel-to-mm conversion
actual_shrimp_width_mm = 6.6    
smoothed_window_size = 20         # Smoothing window size for trajectories

# Define grid for heatmap: 16 columns x 9 rows (16:9 aspect ratio)
grid_cols = 16  
grid_rows = 9
#define the type of vent
vent_type = 'line'# point or 'line'

# Hydrothermal vent location (can be point or line)
# For point vent:
vent_point = (8, 4)  # (x, y) coordinates in grid units
# For line vent（x，y）X represents the x+1th from left to right, and y represents the y+1th from top to bottom:
vent_line_start = (7.5, 8.5)  # Start point of the line
vent_line_end =(9.5, 0.5)   # End point of the line

def calculate_point_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_line_distance(point, line_start, line_end):
    """Calculate perpendicular distance from point to line segment"""
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    # Calculate the perpendicular distance
    A = x - x1
    B = y - y1
    C = x2 - x1
    D = y2 - y1
    
    dot = A * C + B * D
    len_sq = C * C + D * D
    
    if len_sq == 0:
        return calculate_point_distance(point, line_start)
    
    param = dot / len_sq
    
    if param < 0:
        return calculate_point_distance(point, line_start)
    elif param > 1:
        return calculate_point_distance(point, line_end)
    else:
        xx = x1 + param * C
        yy = y1 + param * D
        return calculate_point_distance(point, (xx, yy))

def calculate_correlation(movement_grid, vent_type):
    """Calculate correlation between vent distance and shrimp movement"""
    distances = []
    movements = []
    
    for i in range(grid_rows):
        for j in range(grid_cols):
            if vent_type == 'point':
                dist = calculate_point_distance((j, i), vent_point)
            else:  # line vent
                dist = calculate_line_distance((j, i), vent_line_start, vent_line_end)
            
            distances.append(dist)
            movements.append(movement_grid[i, j])
    
    # Calculate Pearson correlation
    pearson_corr, pearson_p = stats.pearsonr(distances, movements)
    # Calculate Spearman correlation
    spearman_corr, spearman_p = stats.spearmanr(distances, movements)
    
    return {
        'pearson_correlation': pearson_corr,
        'pearson_p_value': pearson_p,
        'spearman_correlation': spearman_corr,
        'spearman_p_value': spearman_p
    }

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

# Store monthly grid data and maximum values
monthly_grids = {}
monthly_max_values = {}
# Store all distances and movements for correlation analysis
all_distances = []
all_movements = []
all_months = []

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
        monthly_movement_grid += movement_grid
        valid_video_count += 1

    if valid_video_count == 0:
        continue
    #Calculate the average monthly movement distance
    avg_movement_grid = monthly_movement_grid / valid_video_count
    
    # Record the maximum value and grid data for this month
    monthly_max_values[month_key] = np.max(avg_movement_grid)
    monthly_grids[month_key] = avg_movement_grid

    # Collect data for correlation analysis
    for i in range(grid_rows):
        for j in range(grid_cols):
            if vent_type == 'point':
                dist = calculate_point_distance((j, i), vent_point)
            else:  # line vent
                dist = calculate_line_distance((j, i), vent_line_start, vent_line_end)
            
            all_distances.append(dist)
            all_movements.append(avg_movement_grid[i, j])
            all_months.append(month_key)

# Find the global maximum value among all months
global_max = max(monthly_max_values.values())

# Create a DataFrame for correlation analysis
correlation_df = pd.DataFrame({
    'Month': all_months,
    'Distance': all_distances,
    'Movement': all_movements
})

# Create correlation visualization
plt.figure(figsize=(10, 8))
plt.rcParams['font.family'] = 'Arial'
plt.rcParams["font.size"] = 18

# Calculate both Pearson and Spearman correlations
pearson_corr, pearson_p = stats.pearsonr(correlation_df['Distance'], correlation_df['Movement'])
spearman_corr, spearman_p = stats.spearmanr(correlation_df['Distance'], correlation_df['Movement'])

with open("out_directory/results.txt","w+",encoding="utf-8") as f1:
    f1.write(f"pearson_corr:{pearson_corr},pearson_p:{pearson_p}\n")
    f1.write(f"spearman_corr:{spearman_corr},spearman_p:{spearman_p}\n")

# Create scatter plot with regression line
sns.regplot(data=correlation_df, x='Distance', y='Movement', 
            scatter_kws={'alpha':0.3, 'color':'#4c93c2ff'},
            line_kws={'color':'darkred'})

plt.title('Distance vs Movement')
plt.xlabel('Distance from Vent')
plt.ylabel('Movement (mm)')

# Add correlation information as text box
corr_text = f'Pearson r: {pearson_corr:.3f} (p={pearson_p:.3f})\n'
corr_text += f'Spearman r: {spearman_corr:.3f} (p={spearman_p:.3f})'
plt.text(0.02, 0.98, corr_text, transform=plt.gca().transAxes,
        verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(out_directory, 'correlation_analysis.pdf'), bbox_inches='tight')
plt.close()

# Continue with individual monthly heatmaps
for month_key, avg_movement_grid in monthly_grids.items():
    # Calculate correlation
    correlation_results = calculate_correlation(avg_movement_grid, vent_type)  
    
    # ---------------- Plot Contour-style Heatmap ----------------
    fig, ax = plt.subplots(figsize=(16, 9))

    # prepare X/Y coordinates for cell centers
    x = np.arange(0.5, grid_cols, 1.0)
    y = np.arange(0.5, grid_rows, 1.0)
    X, Y = np.meshgrid(x, y)

    # contour levels 
    levels = np.linspace(0, global_max, 15)#Make the maximum value of colorbar the same between each month, so that it can be compared between months

    # filled contour with single colormap and global max
    cf = ax.contourf(X, Y, avg_movement_grid, levels=levels, cmap="Reds", vmin=0, vmax=global_max)

    # contour lines in darker red
    cs = ax.contour(X, Y, avg_movement_grid, levels=levels, colors='#63111bff', linewidths=0.5)

    # Plot vent location
    if vent_type == 'point':
        ax.plot(vent_point[0], vent_point[1], 'bo', markersize=10, label='Vent Location')
    else:
        ax.plot([vent_line_start[0], vent_line_end[0]], 
                [vent_line_start[1], vent_line_end[1]], 
                'lightblue', linewidth=2, label='Vent Line')

    # colorbar
    cbar = fig.colorbar(cf, ax=ax, orientation="vertical", label="Movement (mm)")
    
    # Add correlation information to the plot
    corr_text = f"Pearson r: {correlation_results['pearson_correlation']:.3f} (p={correlation_results['pearson_p_value']:.3f})\n"
    corr_text += f"Spearman r: {correlation_results['spearman_correlation']:.3f} (p={correlation_results['spearman_p_value']:.3f})"
    ax.text(0.02, 0.98, corr_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # annotate values at cell centers (optional)
    for i in range(grid_rows):
        for j in range(grid_cols):
            ax.text(j+0.5, i+0.5, f"{avg_movement_grid[i,j]:.0f}",
                    ha='center', va='center', fontsize=6, color='black')

    ax.set_xlim(0, grid_cols)
    ax.set_ylim(0, grid_rows)
    ax.set_xticks(np.arange(grid_cols))
    ax.set_yticks(np.arange(grid_rows))
    ax.set_xticklabels([str(c) for c in range(grid_cols)])
    ax.set_yticklabels([str(r) for r in range(grid_rows)])
    ax.set_title(f"Shrimp Movement Contour: {month_key}")
    ax.set_xlabel("Grid Column")
    ax.set_ylabel("Grid Row")
    ax.legend()
    ax.invert_yaxis()  # optional: origin at top
    plt.tight_layout()
    out_path = os.path.join(out_directory, f"{month_key}_avg_movement_heatmap.pdf")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
