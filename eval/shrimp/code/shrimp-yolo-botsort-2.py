import cv2
from ultralytics import YOLO 
from collections import defaultdict
import numpy as np
#from easyocr_OCRtime import get_first_and_last_time
import os

def save_tracking_results(frame_id, track_id, bbox, conf, output):
    """
    保存跟踪结果到文件，格式符合 TrackEval 要求
    格式: <frame id>,<object id>,<top-left-x>,<top-left-y>,<w>,<h>,<confidence score>,-1,...
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
smoothed_window_size = 10
model=YOLO('/home/jingyichu/app/ultralytics-8.3.67/runs/detect/train32/weights/best.pt')#Change the model path

video_path="/home/jingyichu/data/eval/17-7-23-short.mp4"
#out_path='/home/jingyichu/data/eval/results/shrimp-17-7-23-short2.mp4'

#Ergodic the directory
#for filename in os.listdir(video_directory):
#   if filename.endswith(('.mp4','.mkv')):
#      video_path=os.path.join(video_directory,filename)
capture=cv2.VideoCapture(video_path)
track_txt_path = "shrimp-yolo-botsort.txt"   
track_file = open(track_txt_path, "w")
assert capture.isOpened(), 'error reading the video'

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

# Proportional Dictionary: Save the pixel millimeter conversion coefficient for each crab
pixel_to_mm_ratios = {}
track_frame_counts = defaultdict(int)
min_track_len = int(1 * fps)
lost_frames = defaultdict(int)
max_lost = int(1 * fps)
conf=0.1
iou=0.6 
frame_id = 1
while capture.isOpened():
    
    success,img=capture.read()
    if not success:
        print('read complete')
        break
    active_ids = set()
    shrimp_number = 0
    capture_number = 0
    #Execute target tracking
    results=model.track(img,persist=True,
                        show =False,
                        tracker="shrimp-botsort.yaml",
                        iou=0.5,
                        conf=0.1)#Confirm tracking for each frame
    #Put out a part of results
    if results[0].boxes.id is not None:
        boxs = results[0].boxes.xyxy.cuda().tolist()
        track_ids = results[0].boxes.id.int().cuda().tolist()
        clss=results[0].boxes.cls.int().cuda().tolist()
        confs=results[0].boxes.conf.cuda().tolist()
        shrimp_number += clss.count(1)
        capture_number+=1
        
        #traverse
        for box,track_id,cls,conf in zip(boxs,track_ids,clss,confs):
            if cls != 1:
                continue#0:mussel,1:shrimp
            track_frame_counts[track_id] += 1
            active_ids.add(track_id)
            lost_frames[track_id] = 0
            
            x1, y1, x2, y2 = box
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
            bbox_width_px = x2 - x1
            if track_id not in pixel_to_mm_ratios and bbox_width_px > 0:
                pixel_to_mm_ratios[track_id] = actual_shrimp_width_mm / bbox_width_px
            if len(smoothed_tracks[track_id]) >= 2:
                curr = smoothed_tracks[track_id][-1]
                prev = smoothed_tracks[track_id][-2]
                distance_px = np.linalg.norm(np.array(curr) - np.array(prev))
                distance_mm = distance_px * pixel_to_mm_ratios.get(track_id, 0)

                if 1 < distance_mm <= 5:
                    total_distances[track_id] += distance_mm
                    all_crab_distance += distance_mm
            if track_frame_counts[track_id] >= min_track_len:

                save_tracking_results(frame_id, track_id, box, conf, track_file)

                label = f'ID:{track_id}, shrimp_Conf:{conf:.2f}, Dist:{total_distances[track_id]:.1f}mm'

                cv2.rectangle(
                    img,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (203, 227, 48),
                    2
                )

                cv2.putText(
                    img,
                    label,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (203, 227, 48),
                    2
                )

                points = np.array(smoothed_tracks[track_id], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [points], False, (203, 227, 48), 2)
        for track_id in list(smoothed_tracks.keys()):

            if track_id not in active_ids:

                lost_frames[track_id] += 1

                if lost_frames[track_id] <= max_lost and track_frame_counts[track_id] >= min_track_len:

                    last_point = smoothed_tracks[track_id][-1]
                    cv2.circle(
                        img,
                        (int(last_point[0]), int(last_point[1])),
                        4,
                        (150, 200, 50),
                        -1
                    )

                if lost_frames[track_id] > max_lost:

                    raw_tracks.pop(track_id, None)
                    smoothed_tracks.pop(track_id, None)
                    track_frame_counts.pop(track_id, None)
                    lost_frames.pop(track_id, None)
                    pixel_to_mm_ratios.pop(track_id, None)

        
    
        # # Write the processed frame to the output video
        # video_writer.write(img)
        frame_id += 1
        # Display the frame
        #cv2.imshow('track',img)

        # Exit on pressing 'q'
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            # break

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
# video_writer.release()
cv2.destroyAllWindows()

