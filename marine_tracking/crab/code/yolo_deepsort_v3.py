import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict, deque
from easyocr_OCRtime import get_time 
from easyocr_OCRtime import get_first_and_last_time

# ------------------ Parameter Settings ------------------
actual_crab_width_mm = 42         # Actual crab width (mm) for pixel-to-mm conversion
MIN_CONFIDENCE = 0.75             # YOLO detection confidence threshold
MAX_AGE = 50                      # DeepSort parameter (max age for lost tracks)
N_INIT = 4                        # DeepSort parameter (initial frames to confirm track)
MAHALANOBIS_THRESHOLD = 8.0       # Maximum valid displacement (mm) for normal motion
IOU_THRESHOLD = 0.5               # IoU threshold for matching detections to tracks
SMOOTH_WINDOW = 6                # Number of frames used for smoothing
max_trail_length = 80             # Maximum trail points to display
ACCEL_THRESHOLD = 7.0             # Acceleration threshold (mm/s) to record acceleration events
OUTLIE_THRESHOLD = 8.5            # If computed displacement > this, assume drift (i.e. detection box offset)
CONF_SKIP_THRESHOLD = 0.1         # If confidence is below this, consider the detection invalid
DRAW_COLOR = (203, 227, 48)       # Color for drawing (B, G, R)

# ------------------ Video Paths ------------------
video_path = "E:/现象/17-9-30误判.mp4"
out_path = "E:/现象/test/1test.mp4"

# ------------------ Helper Functions ------------------
def compute_iou(boxA, boxB):
    """
    Compute Intersection over Union (IoU) between two boxes in [x1, y1, x2, y2] format.
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
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def get_detections(frame, model, min_confidence):
    """
    Detect objects using YOLO and format results for DeepSort.
    Returns:
        detections_list: Raw detections for confidence matching [bbox, conf, cls] where bbox=[x1, y1, w, h]
        detections: DeepSort-compatible detections
    """
    detections_list = []
    detections = []
    results = model(frame, verbose=False)[0]
    for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        if int(cls) == 0 and conf >= min_confidence:  # Class 0: crab
            x1, y1, x2, y2 = map(int, box.tolist())
            bbox_width = x2 - x1
            det = ([x1, y1, bbox_width, y2 - y1], conf, str(cls))
            detections.append(det)
            detections_list.append(det)
    return detections_list, detections

def update_tracker(frame, detections, tracker):
    """Update DeepSort tracker and return current tracks."""
    tracks = tracker.update_tracks(detections, frame=frame)
    return tracks

def update_track_info(tracks, detections_list, track_history, track_confidences, pixel_to_mm_ratios, fps):
    """
    Update each track's information:
    - Update current confidence using IoU matching.
    - Calculate the pixel-to-mm conversion ratio.
    - Compute a smoothed center using the last SMOOTH_WINDOW Kalman states.
    - If current confidence is below CONF_SKIP_THRESHOLD, mark the track as in predict mode,
    reset the smoothing baseline, and do not accumulate displacement.
    - Otherwise, compute the delta between the current smoothed center and the previous smoothed center.
    If the computed physical displacement exceeds OUTLIE_THRESHOLD, assume it is due to drift and do not accumulate.
    If the displacement is very small, set speed to 0.
    - Otherwise, update the total distance and average speed.
    - Also, update a 'smoothed_history' for drawing the full trail (up to max_trail_length).
    """
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(int, ltrb)
        
        # Confidence matching using IoU
        curr_conf = 0.0
        track_box = [x1, y1, x2, y2]
        for det in detections_list:
            det_box, det_conf, _ = det
            det_x1, det_y1, det_w, det_h = det_box
            det_box_xyxy = [det_x1, det_y1, det_x1 + det_w, det_y1 + det_h]
            iou = compute_iou(track_box, det_box_xyxy)
            if iou > IOU_THRESHOLD and det_conf > curr_conf:
                curr_conf = det_conf
        track_confidences[track_id] = curr_conf
        
        # Pixel-to-mm conversion
        bbox_width_px = x2 - x1
        if track_id not in pixel_to_mm_ratios:
            pixel_to_mm_ratios[track_id] = (actual_crab_width_mm / bbox_width_px) if bbox_width_px > 0 else 0.0
        
        # Get current Kalman state center (first two elements)
        curr_mean = track.mean.copy()[:2]
        
        # Initialize record for this track if not present
        if track_id not in track_history:
            track_history[track_id] = {
                'positions': deque(maxlen=SMOOTH_WINDOW),
                'smoothed_history': deque(maxlen=max_trail_length),
                'total_distance': 0.0,
                'average_speed': 0.0,
                'prev_smoothed': None,
                'predict_mode': False
            }
        track_history[track_id]['positions'].append(curr_mean)
        
        # When we have enough frames in the smoothing window, compute the smoothed center
        if len(track_history[track_id]['positions']) >= SMOOTH_WINDOW:
            smoothed_current = np.mean(list(track_history[track_id]['positions'])[-SMOOTH_WINDOW:], axis=0)
            if curr_conf < CONF_SKIP_THRESHOLD:
                # If confidence is very low, enter predict mode: reset baseline and skip accumulation.
                track_history[track_id]['predict_mode'] = True
                track_history[track_id]['prev_smoothed'] = smoothed_current
                track_history[track_id]['average_speed'] = 0.0
            else:
                if track_history[track_id].get('predict_mode', False):
                    # If recovering from predict mode, reset baseline to avoid accumulating regression displacement.
                    track_history[track_id]['prev_smoothed'] = smoothed_current
                    track_history[track_id]['average_speed'] = 0.0
                    track_history[track_id]['predict_mode'] = False
                else:
                    if track_history[track_id]['prev_smoothed'] is None:
                        track_history[track_id]['prev_smoothed'] = smoothed_current
                    else:
                        delta = smoothed_current - track_history[track_id]['prev_smoothed']
                        displacement = np.linalg.norm(delta)
                        physical_dist = displacement * pixel_to_mm_ratios[track_id]
                        dt = SMOOTH_WINDOW / fps  # Time interval (seconds)
                        # If displacement is very small, consider speed = 0
                        if displacement < 2.5:
                            speed = 0.0
                        else:
                            speed = physical_dist / dt
                        # If displacement is too large (drift), skip accumulation
                        if physical_dist > OUTLIE_THRESHOLD:
                            track_history[track_id]['average_speed'] = 0.0
                        else:
                            if physical_dist > 0.8 and physical_dist <= MAHALANOBIS_THRESHOLD:
                                track_history[track_id]['total_distance'] += physical_dist
                                track_history[track_id]['average_speed'] = speed
                                if speed >= ACCEL_THRESHOLD and speed < OUTLIE_THRESHOLD:
                                    ocr_time = get_time(track.frame) if hasattr(track, 'frame') else "UNKNOWN_TIME"
                                    with open("acceleration_events.txt", 'a+', encoding='utf-8') as f1:
                                        f1.write(f"Track {track_id} accelerated at {ocr_time}, speed: {speed:.1f} mm/s\n")
                        track_history[track_id]['prev_smoothed'] = smoothed_current
            # Update the smoothed_history for drawing the trail if not in drift (predict) mode
            if not track_history[track_id].get('predict_mode', False):
                track_history[track_id]['smoothed_history'].append(smoothed_current)
    return track_history, track_confidences, pixel_to_mm_ratios

def draw_tracks(frame, tracks, track_history, track_confidences):
    """
    Draw bounding boxes, track IDs, confidence, total distance, speed, and the full trail.
    """
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cv2.rectangle(frame, (x1, y1), (x2, y2), DRAW_COLOR, 2)
        label = f"ID:{track_id},Conf:{track_confidences.get(track_id, 0):.2f},Dist:{track_history[track_id]['total_distance']:.1f}mm,Speed:{track_history[track_id]['average_speed']:.1f} mm/s"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, DRAW_COLOR, 2)
        if track_id in track_history and len(track_history[track_id]['smoothed_history']) > 0:
            pts = np.array([(int(pt[0]), int(pt[1])) for pt in track_history[track_id]['smoothed_history']], dtype=np.int32)
            if pts.shape[0] > 1:
                cv2.polylines(frame, [pts.reshape(-1, 1, 2)], False, DRAW_COLOR, thickness=2)

def main():
    model = YOLO('./runs/detect/train18/weights/best.pt')
    tracker = DeepSort(max_age=MAX_AGE, n_init=N_INIT, embedder="mobilenet", half=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        exit(1)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_writer = cv2.VideoWriter(out_path, cv2.VideoWriter.fourcc(*'mp4v'), fps, (width, height))
    
    # Data structures for tracking
    track_history = defaultdict(dict)
    track_confidences = {}
    pixel_to_mm_ratios = {}
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        detections_list, detections = get_detections(frame, model, MIN_CONFIDENCE)
        tracks = update_tracker(frame, detections, tracker)
        # Pass the current frame to each track for OCR time reading if needed
        for track in tracks:
            track.frame = frame
        track_history, track_confidences, pixel_to_mm_ratios = update_track_info(
            tracks, detections_list, track_history, track_confidences, pixel_to_mm_ratios, fps)
        draw_tracks(frame, tracks, track_history, track_confidences)
        video_writer.write(frame)
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    
    # Save results
    crab_number = len(track_history)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_crab_distance = sum(data['total_distance'] for data in track_history.values())
    # Calculate average movement per crab per minute
    avg_crab_permin_distance = (all_crab_distance / crab_number)  if crab_number else 0
    first_time, last_time = get_first_and_last_time(video_path)
    with open('result2.txt', 'a+', encoding='utf-8') as f2:
        f2.write(f'{first_time} to {last_time}, the average movement distance per crab: {avg_crab_permin_distance:.2f}\n')

if __name__ == '__main__':
    main()
