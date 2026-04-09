import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict, deque
# from easyocr_OCRtime import get_time 
# from easyocr_OCRtime import get_first_and_last_time
import os

# ------------------ Parameter Settings ------------------
actual_crab_width_mm = 42
MIN_CONFIDENCE = 0.75
MAX_AGE = 50
N_INIT = 5
MAHALANOBIS_THRESHOLD = 8.0
IOU_THRESHOLD = 0.6
SMOOTH_WINDOW = 7
max_trail_length = 50
ACCEL_THRESHOLD = 7.0
OUTLIE_THRESHOLD = 8.0
CONF_SKIP_THRESHOLD = 0.3
BOX_IOU_THRESHOLD = 0.3
DRAW_COLOR = (203, 227, 48)
MIN_DISPLACEMENT_THRESHOLD = 2
SPEED_STD_THRESHOLD = 0.3
MIN_HISTORY_COUNT = 3

# ------------------ TrackEval Output Settings ------------------
TRACKEVAL_OUTPUT_DIR = ',/'  # 对应 template2.yaml 的 trackers_folder
SEQ_NAME = 'crab-yolo-deepsort19-4-12'  # 序列名称，对应 template2.yaml 的 
# ------------------ Video Paths ------------------
video_path = "E:/graduate/hydrothermal/eval/crab/19-04-12-4,00_5,00.mp4"
#out_path = "F:/现象/test/1test.mp4"

# ------------------ Helper Functions ------------------
def compute_iou(boxA, boxB):
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
    detections_list = []
    detections = []
    results = model(frame, verbose=False)[0]
    for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        if int(cls) == 0 and conf >= min_confidence:
            x1, y1, x2, y2 = map(int, box.tolist())
            bbox_width = x2 - x1
            det = ([x1, y1, bbox_width, y2 - y1], conf, str(cls))
            detections.append(det)
            detections_list.append(det)
    return detections_list, detections

def update_tracker(frame, detections, tracker):
    tracks = tracker.update_tracks(detections, frame=frame)
    return tracks

def update_track_info(tracks, detections_list, track_history, track_confidences, pixel_to_mm_ratios, fps):
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        
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
        
        bbox_width_px = x2 - x1
        if track_id not in pixel_to_mm_ratios:
            pixel_to_mm_ratios[track_id] = (actual_crab_width_mm / bbox_width_px) if bbox_width_px > 0 else 0.0
        
        curr_mean = track.mean.copy()[:2]
        
        if track_id not in track_history:
            track_history[track_id] = {
                'positions': deque(maxlen=SMOOTH_WINDOW),
                'smoothed_history': deque(maxlen=max_trail_length),
                'speed_history': deque(maxlen=MIN_HISTORY_COUNT),
                'total_distance': 0.0,
                'average_speed': 0.0,
                'prev_smoothed': None,
                'prev_box': None,
                'predict_mode': False,
                'valid_track': True
            }
        track_history[track_id]['positions'].append(curr_mean)
        current_box = [x1, y1, x2, y2]
        
        if len(track_history[track_id]['positions']) >= SMOOTH_WINDOW:
            smoothed_current = np.mean(list(track_history[track_id]['positions'])[-SMOOTH_WINDOW:], axis=0)
            
            if curr_conf < CONF_SKIP_THRESHOLD:
                track_history[track_id]['predict_mode'] = True
                track_history[track_id]['prev_smoothed'] = smoothed_current
                track_history[track_id]['average_speed'] = 0.0
                track_history[track_id]['valid_track'] = False
            else:
                if track_history[track_id].get('predict_mode', False):
                    track_history[track_id]['prev_smoothed'] = smoothed_current
                    track_history[track_id]['average_speed'] = 0.0
                    track_history[track_id]['predict_mode'] = False
                    track_history[track_id]['valid_track'] = True
                else:
                    if track_history[track_id]['prev_smoothed'] is None:
                        track_history[track_id]['prev_smoothed'] = smoothed_current
                    else:
                        delta = smoothed_current - track_history[track_id]['prev_smoothed']
                        displacement = np.linalg.norm(delta)
                        physical_dist = displacement * pixel_to_mm_ratios[track_id]
                        dt = SMOOTH_WINDOW / fps
                        if displacement < MIN_DISPLACEMENT_THRESHOLD:
                            speed = 0.0
                        else:
                            speed = physical_dist / dt
                        
                        if track_history[track_id]['prev_box'] is not None:
                            box_iou = compute_iou(current_box, track_history[track_id]['prev_box'])
                        else:
                            box_iou = 1.0
                        
                        if box_iou < BOX_IOU_THRESHOLD or physical_dist > OUTLIE_THRESHOLD:
                            track_history[track_id]['average_speed'] = 0.0
                            track_history[track_id]['valid_track'] = False
                        else:
                            track_history[track_id]['speed_history'].append(speed)
                            if len(track_history[track_id]['speed_history']) >= MIN_HISTORY_COUNT:
                                speed_std = np.std(list(track_history[track_id]['speed_history']))
                                if speed_std < SPEED_STD_THRESHOLD:
                                    speed = 0.0
                                    track_history[track_id]['valid_track'] = False
                                else:
                                    track_history[track_id]['valid_track'] = True
                            
                            if physical_dist > 0.5 and physical_dist <= MAHALANOBIS_THRESHOLD:
                                track_history[track_id]['total_distance'] += physical_dist
                                track_history[track_id]['average_speed'] = speed
                                #if speed >= ACCEL_THRESHOLD and speed < OUTLIE_THRESHOLD:
                                    #ocr_time = get_time(track.frame) if hasattr(track, 'frame') else "UNKNOWN_TIME"
                                    #with open("acceleration_events.txt", 'a+', encoding='utf-8') as f1:
                                        #f1.write(f"Track {track_id} accelerated at {ocr_time}, speed: {speed:.1f} mm/s\n")
                        track_history[track_id]['prev_smoothed'] = smoothed_current
                track_history[track_id]['prev_box'] = current_box
            
            if track_history[track_id].get('valid_track', True):
                track_history[track_id]['smoothed_history'].append(smoothed_current)
    return track_history, track_confidences, pixel_to_mm_ratios

def draw_tracks(frame, tracks, track_history, track_confidences):
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        if track_id in track_history and not track_history[track_id].get('valid_track', True):
            if track_history[track_id]['prev_box'] is not None:
                x1, y1, x2, y2 = track_history[track_id]['prev_box']
            else:
                x1, y1, x2, y2 = map(int, track.to_ltrb())
        else:
            x1, y1, x2, y2 = map(int, track.to_ltrb())
        cv2.rectangle(frame, (x1, y1), (x2, y2), DRAW_COLOR, 2)
        label = f"ID:{track_id},Conf:{track_confidences.get(track_id,0):.2f},Dist:{track_history[track_id]['total_distance']:.1f}mm,Speed:{track_history[track_id]['average_speed']:.1f} mm/s"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, DRAW_COLOR, 2)
        if track_id in track_history and len(track_history[track_id]['smoothed_history']) > 0:
            pts = np.array([(int(pt[0]), int(pt[1])) for pt in track_history[track_id]['smoothed_history']], dtype=np.int32)
            if pts.shape[0] > 1:
                cv2.polylines(frame, [pts.reshape(-1, 1, 2)], False, DRAW_COLOR, thickness=2)

def save_tracking_results(tracks, track_confidences, frame_id, output_file):
    """
    保存跟踪结果到文件，格式符合 TrackEval 要求
    格式: <frame id>,<object id>,<top-left-x>,<top-left-y>,<w>,<h>,<confidence score>,-1,...
    """
    for track in tracks:
        if not track.is_confirmed():
            continue
        
        track_id = track.track_id
        x1, y1, x2, y2 = map(float, track.to_ltrb())
        
        # 计算bbox参数
        bb_left = x1
        bb_top = y1
        bb_width = x2 - x1
        bb_height = y2 - y1
        
        # 获取置信度
        score = track_confidences.get(track_id, 1.0)
        
        
        # 写入文件：frame,id,left,top,width,height,score,class,visibility
        line = f"{frame_id},{track_id},{bb_left:.2f},{bb_top:.2f},{bb_width:.2f},{bb_height:.2f},1,1,1\n"#{score:.3f}
        output_file.write(line)

def main():
    model = YOLO('D:/khaki/ultralytics-8.3.27/runs/detect/train18/weights/best.pt')
    tracker = DeepSort(max_age=MAX_AGE, n_init=N_INIT, embedder="mobilenet", half=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        exit(1)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    #video_writer = cv2.VideoWriter(out_path, cv2.VideoWriter.fourcc(*'mp4v'), fps, (width, height))
    
    # 创建 TrackEval 输出目录
    os.makedirs(TRACKEVAL_OUTPUT_DIR, exist_ok=True)
    trackeval_output_path = os.path.join(TRACKEVAL_OUTPUT_DIR, f'{SEQ_NAME}.txt')
    
    # 打开输出文件
    trackeval_file = open(trackeval_output_path, 'w')
    
    # Data structures for tracking
    track_history = defaultdict(dict)
    track_confidences = {}
    pixel_to_mm_ratios = {}
    
    frame_id = 1  # 帧号从1开始（对应 FRAME_START_IDX: 1）
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        detections_list, detections = get_detections(frame, model, MIN_CONFIDENCE)
        tracks = update_tracker(frame, detections, tracker)
        for track in tracks:
            track.frame = frame
        track_history, track_confidences, pixel_to_mm_ratios = update_track_info(
            tracks, detections_list, track_history, track_confidences, pixel_to_mm_ratios, fps)
        
        # 保存跟踪结果
        save_tracking_results(tracks, track_confidences, frame_id, trackeval_file)
        
        draw_tracks(frame, tracks, track_history, track_confidences)
        #video_writer.write(frame)
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_id += 1
    
    # 关闭文件
    trackeval_file.close()
    cap.release()
    #video_writer.release()
    cv2.destroyAllWindows()
    
    print(f"\n跟踪结果已保存到: {trackeval_output_path}")
    print(f"总帧数: {frame_id - 1}")
    print(f"跟踪目标数: {len(track_history)}")
    
    # Save results
    crab_number = len(track_history)
    all_crab_distance = sum(data['total_distance'] for data in track_history.values())
    avg_crab_permin_distance = (all_crab_distance / crab_number) if crab_number else 0
    #first_time, last_time = get_first_and_last_time(video_path)
    #with open('result2.txt', 'a+', encoding='utf-8') as f2:
        #f2.write(f'{first_time} to {last_time}, the average movement distance per crab: {avg_crab_permin_distance:.2f}\n')

if __name__ == '__main__':
    main()