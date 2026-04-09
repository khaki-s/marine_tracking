import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict, deque
from easyocr_OCRtime import get_time
from scipy.signal import savgol_filter
import os
import re
import csv
from datetime import datetime


MIN_CONFIDENCE = 0.8
MAX_AGE = 25
N_INIT = 8

SMOOTH_WINDOW_DIST = 1
SG_WINDOW = 7
SG_POLY   = 2
max_trail_length = 50

MAHALANOBIS_THRESHOLD    = 0.4
OUTLIE_THRESHOLD         = 1.4
ACCEL_THRESHOLD          = 0.85
MIN_DISPLACEMENT_THRESHOLD = 2
IOU_THRESHOLD            = 0.7
CONF_SKIP_THRESHOLD      = 0.1
MIN_HISTORY_COUNT        = 3
DRAW_COLOR = (203, 227, 48)

ACCEL_CONFIRM_FRAMES = 3
CSV_OUTPUT_PATH      = './2017_2018_results.csv'

SPEED_DEAD_ZONE_BL = 0.15
MIN_DIST_DISP_BL   = 0.02

AREA_HISTORY_LEN = 12
AREA_RATIO_MAX   = 1.4
AREA_RATIO_MIN   = 0.7

CONF_GATE_THRESHOLD = 0.35
CONF_GATE_FRAMES    = 10
CONF_STABLE_THRESHOLD = 0.6
STABLE_FRAMES         = 3
IOU_STRICT_THRESHOLD  = 0.7

MIN_TRACK_AGE    = 20
INIT_DISP_MAX_BL = 0.2

MAX_PREDICT_FRAMES_SPEED = 2
MAX_SPEED_INCREMENT_BL = 0.25

IOU_STABLE_THRESHOLD = 0.4


def parse_datetime_from_filename(video_path: str):
    basename = os.path.basename(video_path)
    m = re.search(r'(\d{2})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})', basename)
    if m:
        yy, mo, dd, hh, mi, ss = m.groups()
        year         = f"20{yy}"
        date_str     = f"{year}-{mo}-{dd}"
        datetime_str = f"{year}-{mo}-{dd} {hh}:{mi}:{ss}"
        return date_str, datetime_str
    return "UNKNOWN", "UNKNOWN"


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]);  yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]);  yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA);   interH = max(0, yB - yA)
    interArea = interW * interH
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(areaA + areaB - interArea + 1e-6)


def get_detections(frame, model, min_confidence):
    detections_list, detections = [], []
    results = model(frame, verbose=False)[0]
    for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        if int(cls.item()) == 0 and conf.item() >= min_confidence:
            x1, y1, x2, y2 = map(int, box.tolist())
            det = ([x1, y1, x2 - x1, y2 - y1], conf.item(), str(cls.item()))
            detections.append(det)
            detections_list.append(det)
    return detections_list, detections


def update_tracker(frame, detections, tracker):
    return tracker.update_tracks(detections, frame=frame)


def compute_body_length_px(x1, y1, x2, y2):
    w, h = x2 - x1, y2 - y1
    return np.sqrt(w * w + h * h)


def update_track_info(tracks, detections_list, track_history, track_confidences,
                    body_length_px_per_track, fps, date_str):

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())

        curr_conf = 0.0
        track_box = [x1, y1, x2, y2]
        for det in detections_list:
            det_box, det_conf, _ = det
            dx1, dy1, dw, dh = det_box
            det_xyxy = [dx1, dy1, dx1 + dw, dy1 + dh]
            iou = compute_iou(track_box, det_xyxy)
            if iou > IOU_THRESHOLD and det_conf > curr_conf:
                curr_conf = det_conf
        track_confidences[track_id] = curr_conf

        if track_id not in body_length_px_per_track:
            bl = compute_body_length_px(x1, y1, x2, y2)
            body_length_px_per_track[track_id] = bl if bl > 0 else 1.0
        bl_px = body_length_px_per_track[track_id]

        curr_mean = track.mean.copy()[:2]

        if track_id not in track_history:
            track_history[track_id] = {
                'positions_dist':     deque(maxlen=SMOOTH_WINDOW_DIST),
                'positions_speed':    deque(maxlen=SG_WINDOW),
                'smoothed_history':   deque(maxlen=max_trail_length),
                'prev_smoothed_dist': None,
                'prev_box':           None,
                'valid_track':        True,
                'total_distance_bl':  0.0,
                'average_speed_bl':   0.0,
                'accel_counter':      0,
                'accel_cooldown':     0,
                'predict_frames':     0,
                'prev_speed_bl':      0.0,  
                'age':                0,
                'area_history':       deque(maxlen=AREA_HISTORY_LEN),
                'low_conf_frames':    0,
                'conf_gated':         False,
                'id_switch_suspect':  False,
                'stable_counter': 0,
            }

        hist = track_history[track_id]
        hist['age'] += 1
        hist['positions_dist'].append(curr_mean)
        if track.time_since_update > 0:
            hist['predict_frames'] += 1
        else:
            hist['predict_frames'] = 0
            hist['positions_speed'].append(curr_mean)   # single append, real only


        if hist['predict_frames'] > MAX_PREDICT_FRAMES_SPEED:
            hist['stable_counter'] = 0
            hist['positions_speed'].clear()
            hist['prev_speed_bl'] = 0.0

        current_box = [x1, y1, x2, y2]
        if hist['prev_box'] is not None:
            iou_with_prev = compute_iou(hist['prev_box'], current_box)
            hist['id_switch_suspect'] = iou_with_prev < IOU_STABLE_THRESHOLD
        else:
            hist['id_switch_suspect'] = False
        if hist['prev_box'] is not None:
            iou_with_prev = compute_iou(hist['prev_box'], current_box)
        else:
            iou_with_prev = 0.0

        if (curr_conf >= CONF_STABLE_THRESHOLD and 
            iou_with_prev >= IOU_STRICT_THRESHOLD):
            
            hist['stable_counter'] += 1
        else:
            hist['stable_counter'] = 0
        if hist['id_switch_suspect']:
            hist['positions_speed'].clear()
            hist['positions_dist'].clear()
            hist['prev_smoothed_dist'] = None
            hist['prev_speed_bl']      = 0.0
            hist['accel_counter']      = 0
            hist['age']                = 0
            hist['stable_counter'] = 0
            hist['prev_box']           = current_box
            continue

        curr_area = max(1.0, float((x2 - x1) * (y2 - y1)))
        hist['area_history'].append(curr_area)
        area_stable = True
        if len(hist['area_history']) >= 3:
            median_area = float(np.median(list(hist['area_history'])))
            ratio = curr_area / (median_area + 1e-6)
            if ratio > AREA_RATIO_MAX or ratio < AREA_RATIO_MIN:
                area_stable = False

        if curr_conf < CONF_GATE_THRESHOLD:
            hist['low_conf_frames'] += 1
        else:
            hist['low_conf_frames'] = 0
            hist['conf_gated']      = False
        if hist['low_conf_frames'] >= CONF_GATE_FRAMES:
            hist['conf_gated'] = True

        if len(hist['positions_dist']) < SMOOTH_WINDOW_DIST:
            continue
        smoothed_dist = np.mean(list(hist['positions_dist']), axis=0)

        if len(hist['positions_speed']) >= SG_WINDOW and hist['predict_frames'] == 0:
            pts   = np.array(hist['positions_speed'])
            diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            if np.max(diffs) > bl_px * 0.5:
                hist['positions_speed'].clear()
                hist['prev_speed_bl'] = 0.0
                speed_bl = 0.0
            else:
                vx = savgol_filter(pts[:, 0], SG_WINDOW, SG_POLY, deriv=1)
                vy = savgol_filter(pts[:, 1], SG_WINDOW, SG_POLY, deriv=1)
                speed_px = np.linalg.norm([vx[-1], vy[-1]])
                speed_bl = (speed_px / bl_px) * fps

                # Rate cap: clip instantaneous spikes (drift/ID-switch) without
                # attenuating genuine sustained acceleration.
                max_allowed = hist['prev_speed_bl'] + MAX_SPEED_INCREMENT_BL
                speed_bl    = min(speed_bl, max_allowed)
        else:
            speed_bl = 0.0

        if speed_bl < SPEED_DEAD_ZONE_BL:
            speed_bl = 0.0

        # BUG FIX 3: prev_speed_bl was never updated, so rate cap always
        # compared against 0.0.  Update it here with the final capped value.
        hist['prev_speed_bl'] = speed_bl

        frame_valid = area_stable and not hist['conf_gated'] and hist['stable_counter'] >= STABLE_FRAMES

        if hist['prev_smoothed_dist'] is not None:
            delta   = smoothed_dist - hist['prev_smoothed_dist']
            disp_px = np.linalg.norm(delta)
            disp_bl = disp_px / bl_px

            if hist['age'] < 25 and disp_bl > INIT_DISP_MAX_BL:
                hist['prev_smoothed_dist'] = smoothed_dist
                hist['smoothed_history'].append(smoothed_dist)
                hist['prev_box'] = current_box
                continue

            if frame_valid:
                if MIN_DIST_DISP_BL <= disp_bl <= MAHALANOBIS_THRESHOLD:
                    hist['total_distance_bl'] += disp_bl
                hist['average_speed_bl'] = speed_bl
            else:
                hist['average_speed_bl'] = 0.0

        hist['prev_smoothed_dist'] = smoothed_dist
        hist['smoothed_history'].append(smoothed_dist)
        hist['prev_box'] = current_box

        if hist['accel_cooldown'] > 0:
            hist['accel_cooldown'] -= 1

        if hist['age'] >= MIN_TRACK_AGE and frame_valid and ACCEL_THRESHOLD <= speed_bl < OUTLIE_THRESHOLD:
            hist['accel_counter'] += 1
        else:
            hist['accel_counter'] = 0

        if (hist['age']          >= MIN_TRACK_AGE and
                hist['accel_counter']  >= ACCEL_CONFIRM_FRAMES and
                hist['accel_cooldown'] == 0):
            ocr_time = get_time(track.frame) if hasattr(track, 'frame') else "UNKNOWN_TIME"
            with open("acceleration_events.txt", 'a+', encoding='utf-8') as f1:
                f1.write(
                    f"[{date_str}] Track {track_id} accelerated at "
                    f"{ocr_time}, speed: {speed_bl:.3f} BL/s\n"
                )
            hist['accel_cooldown'] = 10
            hist['accel_counter']  = 0

    return track_history, track_confidences, body_length_px_per_track


def draw_tracks(frame, tracks, track_history, track_confidences):
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        hist = track_history.get(track_id, {})

        if not hist.get('valid_track', True) and hist.get('prev_box') is not None:
            x1, y1, x2, y2 = hist['prev_box']
        else:
            x1, y1, x2, y2 = map(int, track.to_ltrb())

        gated     = hist.get('conf_gated', False)
        #box_color = (0, 0, 220) if gated else DRAW_COLOR
        cv2.rectangle(frame, (x1, y1), (x2, y2), DRAW_COLOR, 2)

        dist_bl  = hist.get('total_distance_bl', 0.0)
        speed_bl = hist.get('average_speed_bl',  0.0)
        label = (
            f"ID:{track_id}  "
            f"Conf:{track_confidences.get(track_id, 0):.2f}  "
            f"Dist:{dist_bl:.2f} BL  "
            f"Speed:{speed_bl:.3f} BL/s"
            + ("  [GATED]" if gated else "")
        )
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, DRAW_COLOR, 2)

        if len(hist.get('smoothed_history', [])) > 1:
            pts = np.array(
                [(int(p[0]), int(p[1])) for p in hist['smoothed_history']],
                dtype=np.int32,
            )
            cv2.polylines(frame, [pts.reshape(-1, 1, 2)], False, DRAW_COLOR, 2)


def append_csv_result(csv_path: str, datetime_str: str, avg_dist_bl: float):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['time', 'distance/body_length/per_crab'])
        writer.writerow([datetime_str, f'{avg_dist_bl:.4f}'])


def main(model_path: str, video_path: str, out_path: str):
    date_str, datetime_str = parse_datetime_from_filename(video_path)
    model   = YOLO(model_path)
    tracker = DeepSort(max_age=MAX_AGE, n_init=N_INIT, embedder="mobilenet", half=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    #video_writer = cv2.VideoWriter(
    #    out_path, cv2.VideoWriter.fourcc(*'mp4v'), fps, (width, height)
    #)

    track_history            = defaultdict(dict)
    track_confidences        = {}
    body_length_px_per_track = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections_list, detections = get_detections(frame, model, MIN_CONFIDENCE)
        tracks = update_tracker(frame, detections, tracker)

        for track in tracks:
            track.frame = frame

        track_history, track_confidences, body_length_px_per_track = update_track_info(
            tracks, detections_list, track_history, track_confidences,
            body_length_px_per_track, fps, date_str
        )

        draw_tracks(frame, tracks, track_history, track_confidences)
        # video_writer.write(frame)
        #cv2.imshow("Tracking", frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

    cap.release()
    # video_writer.release()
    #cv2.destroyAllWindows()

    crab_number = len(track_history)
    all_dist_bl = sum(d.get('total_distance_bl', 0.0) for d in track_history.values())
    avg_dist_bl = (all_dist_bl / crab_number) if crab_number else 0.0

    append_csv_result(CSV_OUTPUT_PATH, datetime_str, avg_dist_bl)
    print(f"Done — crabs: {crab_number},  avg dist: {avg_dist_bl:.4f} BL")

if __name__ == '__main__':
    MODEL_PATH = '/home/jingyichu/app/ultralytics-8.3.67/runs/detect/train18/weights/best.pt'
    VIDEO_DIRECTORY = "/home/jingyichu/data/2017_2018/2017_2018"
    CSV_OUTPUT_PATH = './2017_2018_results.csv'
    video_out_path   = './SMOOVE-17-12-01_23-58-34-79_tracked.mp4'
    for filename in os.listdir(VIDEO_DIRECTORY):
        if filename.lower().endswith(('.mp4', '.mkv')):
            video_path = os.path.join(VIDEO_DIRECTORY, filename)
            if cv2.VideoCapture(video_path).isOpened():
                main(MODEL_PATH, video_path,video_out_path)
            else:
                print(f"Skipping unreadable file: {filename}")


