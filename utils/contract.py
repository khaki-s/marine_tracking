import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
import torch

def compute_iou(boxA, boxB):
    """
    Compute the Intersection over Union (IoU) between two bounding boxes.
    boxA, boxB: Bounding boxes in [x1, y1, x2, y2] format
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def evaluate_tracking(video_path, model_path, use_deepsort=True, output_dir="evaluation_results"):
    """
    Evaluate tracking performance, comparing YOLO's built-in tracker with an optimized DeepSORT
    
    Parameters:
        video_path: Path to the video file
        model_path: Path to the YOLO model
        use_deepsort: Whether to use DeepSORT (True) or YOLO's built-in tracker (False)
        output_dir: Directory to save results
    
    Returns:
        metrics: Dictionary containing tracking metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model
    model = YOLO(model_path)
    
    # DeepSORT optimization parameters
    MAX_AGE = 50                     # Maximum frames to keep a track without updates
    N_INIT = 5                       # Number of frames to confirm a track
    MIN_CONFIDENCE = 0.8             # Detection confidence threshold
    IOU_THRESHOLD = 0.6              # IoU matching threshold
    SMOOTH_WINDOW = 7                # Smoothing window size
    
    # Initialize tracker
    if use_deepsort:
        tracker = DeepSort(max_age=MAX_AGE, n_init=N_INIT, embedder="mobilenet", half=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Tracking metrics
    metrics = {
        "track_count": 0,           # Total number of tracks
        "lost_tracks": 0,           # Number of lost tracks
        "short_tracks": 0,          # Number of short-term tracks (<10 frames)
        "avg_track_length": 0,      # Average track length (frames)
        "id_switches": 0,           # Number of ID switches
        "fragmentation": 0,         # Track fragmentation
        "drift_corrections": 0,     # Drift correction count
        "valid_tracks_ratio": 0.0   # Ratio of valid tracks
    }
    
    # Track history
    track_history = defaultdict(lambda: {
        'frames': [],               # Frames where the track appeared
        'positions': deque(maxlen=SMOOTH_WINDOW),  # Position history for smoothing
        'speed_history': deque(maxlen=3),          # Speed history for drift detection
        'valid_updates': 0,         # Valid update count
        'total_updates': 0,         # Total update count
        'prev_box': None,           # Previous frame's bounding box
        'prev_smoothed': None       # Previous frame's smoothed center
    })
    prev_tracks = {}
    track_confidences = {}          # Confidence for each track ID
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process based on the selected tracker
        if use_deepsort:
            # Use DeepSORT with YOLO's predict function for detections
            results = model.predict(frame, verbose=False)
            
            detections = []
            detections_list = []
            
            # Ensure there are boxes in the results
            if hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                # Extract bounding boxes, confidences, and classes
                boxes = results[0].boxes.xyxy
                confs = results[0].boxes.conf
                clss = results[0].boxes.cls
                
                for box, conf, cls in zip(boxes, confs, clss):
                    if conf.item() >= MIN_CONFIDENCE and int(cls.item()) == 0:  # Assuming class 0 is the target
                        x1, y1, x2, y2 = map(int, box.tolist())
                        w, h = x2 - x1, y2 - y1
                        det = ([x1, y1, w, h], conf.item(), str(cls.item()))
                        detections.append(det)
                        detections_list.append(det)
            
            # Update the tracker
            tracks = tracker.update_tracks(detections, frame=frame)
            
            # Record current frame's tracks
            current_tracks = {}
            for track in tracks:
                if track.is_confirmed():
                    track_id = track.track_id
                    current_tracks[track_id] = True
                    
                    # Update track information
                    if track_id not in track_history:
                        track_history[track_id] = {
                            'frames': [],
                            'positions': deque(maxlen=SMOOTH_WINDOW),
                            'speed_history': deque(maxlen=3),
                            'valid_updates': 0,
                            'total_updates': 0,
                            'prev_box': None,
                            'prev_smoothed': None
                        }
                    
                    track_history[track_id]['frames'].append(frame_idx)
                    
                    # Get current Kalman state center
                    curr_mean = track.mean.copy()[:2]
                    track_history[track_id]['positions'].append(curr_mean)
                    
                    # Get current bounding box
                    x1, y1, x2, y2 = map(int, track.to_ltrb())
                    current_box = [x1, y1, x2, y2]
                    
                    # Update confidence
                    curr_conf = 0.0
                    for det in detections_list:
                        det_box, det_conf, _ = det
                        det_x1, det_y1, det_w, det_h = det_box
                        det_box_xyxy = [det_x1, det_y1, det_x1 + det_w, det_y1 + det_h]
                        iou = compute_iou(current_box, det_box_xyxy)
                        if iou > IOU_THRESHOLD and det_conf > curr_conf:
                            curr_conf = det_conf
                    track_confidences[track_id] = curr_conf
                    
                    # Calculate smoothed center when enough position history is available
                    if len(track_history[track_id]['positions']) >= SMOOTH_WINDOW:
                        smoothed_current = np.mean(list(track_history[track_id]['positions'])[-SMOOTH_WINDOW:], axis=0)
                        
                        # Check for bounding box changes
                        valid_update = True
                        if track_history[track_id]['prev_box'] is not None:
                            box_iou = compute_iou(current_box, track_history[track_id]['prev_box'])
                            if box_iou < 0.3:  # Large change in bounding box
                                valid_update = False
                                metrics["drift_corrections"] += 1
                        
                        track_history[track_id]['total_updates'] += 1
                        if valid_update:
                            track_history[track_id]['valid_updates'] += 1
                        
                        track_history[track_id]['prev_box'] = current_box
                        track_history[track_id]['prev_smoothed'] = smoothed_current
        else:
            # Use YOLO's built-in tracker - directly use the track function
            results = model.track(frame, persist=True, verbose=False)
            current_tracks = {}
            
            # Check for track IDs
            if hasattr(results[0], 'boxes') and hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
                # Extract tracking information directly from the results
                try:
                    boxes = results[0].boxes.xyxy
                    track_ids = results[0].boxes.id
                    confs = results[0].boxes.conf
                    clss = results[0].boxes.cls
                    
                    for i, (box, track_id, conf, cls) in enumerate(zip(boxes, track_ids, confs, clss)):
                        if conf.item() >= MIN_CONFIDENCE and int(cls.item()) == 0:  # Assuming class 0 is the target
                            track_id = int(track_id.item())
                            current_tracks[track_id] = True
                            
                            # Simple record of trajectory frames
                            if track_id not in track_history:
                                track_history[track_id] = {'frames': []}
                            
                            track_history[track_id]['frames'].append(frame_idx)
                except Exception as e:
                    print(f"Error processing YOLO tracking results: {e}")
        
        # Calculate ID switches and fragmentation
        if frame_idx > 0:
            for track_id in current_tracks:
                if track_id not in prev_tracks:
                    # New track, check for ID switch
                    for old_id in prev_tracks:
                        if old_id not in current_tracks and frame_idx - track_history[old_id]['frames'][-1] <= 5:
                            metrics["id_switches"] += 1
                            break
        
        prev_tracks = current_tracks
        frame_idx += 1
        
        # Display progress
        if frame_idx % 100 == 0:
            print(f"Processing frame: {frame_idx}/{total_frames}")
    
    cap.release()
    
    # Calculate metrics
    metrics["track_count"] = len(track_history)
    
    track_lengths = []
    valid_tracks_count = 0
    
    for track_id, track_info in track_history.items():
        frames = track_info['frames']
        track_len = len(frames)
        track_lengths.append(track_len)
        
        # Short-term tracks
        if track_len < 10:
            metrics["short_tracks"] += 1
        
        # Check for track fragmentation
        gaps = 0
        for i in range(1, len(frames)):
            if frames[i] - frames[i-1] > 1:
                gaps += 1
        
        if gaps > 0:
            metrics["fragmentation"] += 1
        
        # Calculate valid update ratio
        if track_info['total_updates'] > 0:
            valid_ratio = track_info['valid_updates'] / track_info['total_updates']
            if valid_ratio > 0.7:  # If valid update ratio is greater than 70%, consider it a valid track
                valid_tracks_count += 1
    
    # Average track length
    if track_lengths:
        metrics["avg_track_length"] = sum(track_lengths) / len(track_lengths)
    
    # Lost tracks
    metrics["lost_tracks"] = metrics["fragmentation"] + metrics["short_tracks"]
    
    # Valid track ratio
    if metrics["track_count"] > 0:
        metrics["valid_tracks_ratio"] = valid_tracks_count / metrics["track_count"]
    
    # Save results
    tracker_name = "DeepSORT" if use_deepsort else "YOLO-BoTSORT"
    result_file = os.path.join(output_dir, f"{os.path.basename(video_path)}_{tracker_name}_metrics.txt")
    
    with open(result_file, 'w') as f:
        f.write(f"Tracking Evaluation Results - {tracker_name}\n")
        f.write(f"Video: {video_path}\n")
        f.write(f"Total Frames: {total_frames}\n")
        f.write(f"Total Track IDs: {metrics['track_count']}\n")
        f.write(f"Average Track Length: {metrics['avg_track_length']:.2f} Frames\n")
        f.write(f"ID Switch Count: {metrics['id_switches']}\n")
        f.write(f"Track Fragmentation: {metrics['fragmentation']}\n")
        f.write(f"Short-Term Tracks: {metrics['short_tracks']}\n")
        f.write(f"Lost Tracks: {metrics['lost_tracks']}\n")
        f.write(f"Drift Corrections: {metrics['drift_corrections']}\n")
        f.write(f"Valid Track Ratio: {metrics['valid_tracks_ratio']:.2f}\n")
    
    print(f"Evaluation results saved to {result_file}")
    return metrics

# Example usage
if __name__ == "__main__":
    video_path = "E:/hydrothermal/2017-2018/2017_2018_mp4/17-7-23-short.mp4"  
    model_path = "D:/khaki/marine_tracking/runs/detect/train18/weights/best.pt"
    
    # Evaluate DeepSORT
    deepsort_metrics = evaluate_tracking(video_path, model_path, use_deepsort=True)
    
    # Evaluate YOLO's built-in tracker
    yolo_metrics = evaluate_tracking(video_path, model_path, use_deepsort=False)
    
    # Compare results
    print("\nComparison Results:")
    print(f"{'Metric':<20} {'Optimized DeepSORT':<15} {'YOLO-BoTSORT':<15} {'Difference':<10}")
    print("-" * 60)
    for metric in deepsort_metrics:
        diff = deepsort_metrics[metric] - yolo_metrics[metric]
        diff_str = f"{diff:+.2f}" if isinstance(deepsort_metrics[metric], float) else f"{diff:+d}"
        
        if metric in ["track_count", "avg_track_length", "valid_tracks_ratio"]:
            # Higher values are better, positive difference means DeepSORT is better
            better = "DeepSORT" if diff > 0 else "BoTSORT" if diff < 0 else "Same"
        else:
            # Lower values are better, negative difference means DeepSORT is better
            better = "DeepSORT" if diff < 0 else "BoTSORT" if diff > 0 else "Same"
        
        print(f"{metric:<20} {deepsort_metrics[metric]:<15.2f} {yolo_metrics[metric]:<15.2f} {diff_str:<10} ({better})")
    
    # Comprehensive evaluation
    deepsort_score = (
        deepsort_metrics["valid_tracks_ratio"] * 2 +
        deepsort_metrics["avg_track_length"] / 100 -
        deepsort_metrics["id_switches"] / 100 -
        deepsort_metrics["fragmentation"] / 100 -
        deepsort_metrics["lost_tracks"] / 100
    )
    
    botsort_score = (
        yolo_metrics["valid_tracks_ratio"] * 2 +
        yolo_metrics["avg_track_length"] / 100 -
        yolo_metrics["id_switches"] / 100 -
        yolo_metrics["fragmentation"] / 100 -
        yolo_metrics["lost_tracks"] / 100
    )
    
    print("\nComprehensive Score:")
    print(f"Optimized DeepSORT: {deepsort_score:.2f}")
    print(f"YOLO-BoTSORT: {botsort_score:.2f}")
    print(f"Suggested Tracker: {'Optimized DeepSORT' if deepsort_score > botsort_score else 'YOLO-BoTSORT'}")
