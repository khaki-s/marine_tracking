import cv2
from ultralytics import YOLO 
from collections import defaultdict
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
#from easyocr_OCRtime import get_first_and_last_time
import os


def save_tracking_results(frame_id, track_id, bbox, conf, output):
    """
    TrackEval  format
    <frame id>,<object id>,<top-left-x>,<top-left-y>,<w>,<h>,<confidence score>,-1,...
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    line = f"{frame_id},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,1,1\n"
    output.write(line)


#The actual size of the crab (unit:mm）
actual_shrimp_width_mm = 10
capture_number = 0
shrimp_number = 0
MIN_TRACK_LEN = 20
smoothed_window_size = 10
model=YOLO('D:/khaki/ultralytics-8.3.27/runs/detect/train32/weights/best.pt')#Change the model path
tracker = DeepSort( embedder="mobilenet", half=True)
video_path="E:/graduate/hydrothermal/eval/shrimp/17-7-23-short.mp4"
#out_path='E:/hydrothermal/2017-2018/test/2017-7-23_test.mp4'

#Ergodic the directory
#for filename in os.listdir(video_directory):
#   if filename.endswith(('.mp4','.mkv')):
#      video_path=os.path.join(video_directory,filename)
capture=cv2.VideoCapture(video_path)
tracker = DeepSort(
    embedder="mobilenet",
    half=True
)
assert capture.isOpened(), 'error reading the video'
track_txt_path = "shrimp-yolo-deepsort.txt"   
track_file = open(track_txt_path, "w")
#Read the with,height,fps of the video
w,h,fps=(int(capture.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT,cv2.CAP_PROP_FPS))
#1920 1080 25

# # Video writer 
#video_writer=cv2.VideoWriter(out_path, cv2.VideoWriter.fourcc(*"mp4v"), fps, (w, h))

#Trajectory Storage Dictionary
raw_tracks = defaultdict(lambda: [])
smoothed_tracks = defaultdict(lambda:[])

#Accumulated distance
total_distances=defaultdict(float)
all_crab_distance=0
#The number of points displayed
max_trail_length=80

tracker = DeepSort(
    max_age=30,
    n_init=3,
    max_iou_distance=0.9,
    embedder="torchreid",  
    embedder_model_name="osnet_ain_x1_0", 
    half=True,
    bgr=True,
    embedder_gpu=True
)

# Proportional Dictionary: Save the pixel millimeter conversion coefficient for each crab
pixel_to_mm_ratios = {}
frame_id =1
while capture.isOpened():
    success,img=capture.read()
    if not success:
        print('read complete')
        break
    shrimp_number = 0
    capture_number = 0
    #Execute target tracking
    results=model(img, verbose=False)[0]#Confirm tracking for each frame
    detections = []

    for box, conf, cls in zip(
            results.boxes.xyxy.cpu().numpy(),
            results.boxes.conf.cpu().numpy(),
            results.boxes.cls.cpu().numpy()):
        if int(cls) == 1:  # 0:mussel,1:shrimp
            x1, y1, x2, y2 = box
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'shrimp'))
    tracks = tracker.update_tracks(detections, frame=img)
    
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(float, track.to_ltrb())
        conf = track.det_conf if track.det_conf is not None else 1.0
        save_tracking_results(
            frame_id=frame_id,
            track_id=track_id,
            bbox=[x1, y1, x2, y2],
            conf=conf,
            output = track_file,
        )  
        bbox_width_px = x2 - x1
        if track_id not in pixel_to_mm_ratios and bbox_width_px > 0:
            pixel_to_mm_ratios[track_id] = actual_shrimp_width_mm / bbox_width_px

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        raw_tracks[track_id].append((center_x, center_y))

        if len(raw_tracks[track_id]) >= smoothed_window_size:
            smoothed = np.mean(
                raw_tracks[track_id][-smoothed_window_size:], axis=0
            )
        else:
            smoothed = (center_x, center_y)

        smoothed_tracks[track_id].append(smoothed)

        if len(smoothed_tracks[track_id]) > max_trail_length:
            smoothed_tracks[track_id].pop(0)

        if len(smoothed_tracks[track_id]) >= 2:
            curr = smoothed_tracks[track_id][-1]
            prev = smoothed_tracks[track_id][-2]
            distance_px = np.linalg.norm(np.array(curr) - np.array(prev))
            distance_mm = distance_px * pixel_to_mm_ratios.get(track_id, 0)

            if 1 < distance_mm <= 5:
                total_distances[track_id] += distance_mm

    frame_id += 1

#On average, each crab moves mm per minute
if shrimp_number:
    avg_shrimp_permin_distance=(all_crab_distance/shrimp_number)
else:
    avg_shrimp_permin_distance = 0
#Call the function to obtain the time corresponding to the first and last frames
#first_time, last_time = get_first_and_last_time(video_path)

#sort the distance reverse order
sorted_distance=sorted(total_distances.items(),
                    key=lambda x:x[1],reverse=True)
#Write the result
# with open('track_result.txt','a+',encoding='utf-8') as f1:
#      for obj_id,total_distance in sorted_distance:
#        f1.write(f'{first_time}to{last_time},{obj_id},{total_distance:.2f}\n')

# with open('result2.txt','a+',encoding='utf-8')as f2:
#     f2.write(f'{first_time}to{last_time},{avg_shrimp_permin_distance:.2f}\n')

#Release resources
capture.release()
track_file.close()
#video_writer.release()
cv2.destroyAllWindows()

