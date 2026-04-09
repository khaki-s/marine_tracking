import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

# ---------------- Configuration ----------------
model = YOLO('/home/jingyichu/app/marine-tracking/runs/detect/train32/weights/best.pt')
video_directory = "/home/jingyichu/data/2018-2019/2018_2019"
out_directory = "/home/jingyichu/app/marine-tracking/track/shirmp/trajectory/2018-2019"

min_dist_threshold = 2.0                         # Minimum movement (BL) to keep a track
smoothed_window_size = 20                        # Smoothing window size for trajectory
 
shrimp_color_map = plt.get_cmap('tab20')         # Color map for shrimp ID coloring
shrimp_color_default = "#a8d2e0ff"
mussel_color = "#c4db86ff"

# ---------------- Helper Functions ----------------
def get_color(idx):
    """Get a unique color for each shrimp ID."""
    return shrimp_color_map(idx % 20)

def total_distance(points, track_id, body_lengths_px):
    """Calculate total trajectory distance in Body Lengths (BL).

    Args:
        points: List of (cx, cy) center positions for the track.
        track_id: Track identifier used to look up body length.
        body_lengths_px: Dict mapping track_id → EMA body length (px diagonal).

    Returns:
        dist_bl (float): Total smoothed distance in BL.
        smoothed (np.ndarray): Smoothed trajectory points.
    """
    if len(points) < smoothed_window_size:
        return 0, np.array([])  # Return empty smoothed array too

    smoothed = [
        np.mean(points[max(0, i - smoothed_window_size + 1):i + 1], axis=0)
        for i in range(len(points))
    ]
    smoothed = np.array(smoothed)

    dist_px = sum(
        np.linalg.norm(np.array(smoothed[i]) - np.array(smoothed[i - 1]))
        for i in range(1, len(smoothed))
    )


    bl = body_lengths_px.get(track_id, 0)
    dist_bl = (dist_px / bl) if bl > 0 else 0

    return dist_bl, smoothed

# ---------------- Main Loop ----------------
for filename in os.listdir(video_directory):
    video_path = os.path.join(video_directory, filename)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_pdf_path = os.path.join(out_directory, f"{base_name}.png")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        continue

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    shrimp_tracks = defaultdict(list)
    body_lengths_px = {}
    last_frame = None

    # -------- Frame-by-frame processing --------
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        last_frame = frame.copy()  # Save last valid frame

        results = model.track(frame, persist=True,
                            show=False,
                            tracker="shrimp-botsort.yaml",
                            iou=0.5,
                            conf=0.1)
        if results[0].boxes.id is None:
            continue

        boxes = results[0].boxes.xyxy.cuda().tolist()
        ids = results[0].boxes.id.int().cuda().tolist()
        clss = results[0].boxes.cls.int().cuda().tolist()
        confs = results[0].boxes.conf.cuda().tolist()

        for box, cls, track_id, conf in zip(boxes, clss, ids, confs):
            if cls == 1 and conf >= 0.5:  # Shrimp
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                shrimp_tracks[track_id].append((cx, cy))

                body_length_px = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if track_id not in body_lengths_px:
                    body_lengths_px[track_id] = body_length_px
                else:
                    body_lengths_px[track_id] = (
                        0.8 * body_lengths_px[track_id] + 0.2 * body_length_px
                    )

    cap.release()

    # -------- Mussel detection on last frame --------
    mussel_boxes = []
    if last_frame is not None:
        results = model.predict(last_frame)
        if results[0].boxes:
            for box, cls, conf in zip(results[0].boxes.xyxy.cuda().tolist(),
                                    results[0].boxes.cls.int().cuda().tolist(),
                                    results[0].boxes.conf.cuda().tolist()):
                if cls == 0 and conf >= 0.5:  # Mussel
                    x1, y1, x2, y2 = box
                    mussel_boxes.append((x1, y1, x2 - x1, y2 - y1))
 
    # -------- Plotting to PDF --------
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)  # Invert Y-axis
    ax.set_facecolor('white')
    ax.axis('off')
 
    # Draw shrimp tracks
    for track_id, points in shrimp_tracks.items():
        dist_bl, smoothed = total_distance(points, track_id, body_lengths_px)
        if len(points) >= 2 and dist_bl > min_dist_threshold:
            ax.plot(smoothed[:, 0], smoothed[:, 1], color=get_color(track_id), linewidth=8)
 
    # Draw mussel boxes
    for (x, y, w_box, h_box) in mussel_boxes:
        rect = Rectangle((x, y), w_box, h_box, linewidth=10,
                        edgecolor=mussel_color, facecolor='none')
        ax.add_patch(rect)
 
    plt.tight_layout(pad=0)
    plt.savefig(output_pdf_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"pdf has saved in {out_directory}")