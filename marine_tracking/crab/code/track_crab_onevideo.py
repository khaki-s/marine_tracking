import cv2
from ultralytics import YOLO 
from collections import defaultdict
import numpy as np
from easyocr_OCRtime import get_first_and_last_time


#The actual size of the crab (unit:mm）
actual_crab_width_mm = 42
capture_number = 0
crab_number = 0
smoothed_window_size = 10

model=YOLO('../../runs/detect/train28/weights/best.pt')#Change the model path

video_path="E:/hydrothermal/2017-2018/2017_2018_mp4/SMOOVE-17-12-01_23-58-34-79.mp4"
out_path='E:/hydrothermal/2017-2018/test/2017-12-01_test.mp4'

#Ergodic the directory
#for filename in os.listdir(video_directory):
#   if filename.endswith(('.mp4','.mkv')):
#      video_path=os.path.join(video_directory,filename)
capture=cv2.VideoCapture(video_path)
assert capture.isOpened(), 'error reading the video'

#Read the with,height,fps of the video
w,h,fps=(int(capture.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT,cv2.CAP_PROP_FPS))
#1920 1080 25

# # Video writer 
video_writer=cv2.VideoWriter(out_path, cv2.VideoWriter.fourcc(*"mp4v"), fps, (w, h))

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

while capture.isOpened():
    success,img=capture.read()
    if not success:
        print('read complete')
        break
    capture_number+=1

    #Execute target tracking
    results=model.track(img,persist=True)#Confirm tracking for each frame
    
    #Put out a part of results
    if results[0].boxes.id is not None:
        boxs = results[0].boxes.xyxy.cuda().tolist()
        track_ids = results[0].boxes.id.int().cuda().tolist()
        clss=results[0].boxes.cls.int().cuda().tolist()
        confs=results[0].boxes.conf.cuda().tolist()
        crab_number += clss.count(0)
        #traverse
        for box,track_id,cls,conf in zip(boxs,track_ids,clss,confs):
            if cls == 0 and conf>=0.5:#0:crab,1:crab_small,2:mussel
                x1,y1,x2,y2=box
                bbox_width_px = x2 - x1
                # Calculate the conversion coefficient from pixels to centimeters (calculated separately for each target)
                if track_id not in pixel_to_mm_ratios:
                    if bbox_width_px > 0:
                        pixel_to_mm_ratios[track_id] = actual_crab_width_mm / bbox_width_px
                    else:
                        pixel_to_mm_ratios[track_id] = 0

                #calculate the center coordinates
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                raw_tracks[track_id].append((center_x,center_y))
                
                #Smooth processing
                if len(raw_tracks[track_id]) >= smoothed_window_size:
                    smoothed=np.mean(raw_tracks[track_id][-smoothed_window_size:],axis=0)#Take the average of the last five values of the column
                else:
                    smoothed = (center_x,center_y)
                smoothed_tracks[track_id].append(smoothed)
                #limit trajectory length
                '''
                If it is greater than the specified maximum value, delete the first point
                '''
                if len(smoothed_tracks[track_id]) >max_trail_length:
                    raw_tracks[track_id].pop(0)
                
                #calculate the distance
                if len(smoothed_tracks[track_id]) >= 2:
                    curr = smoothed_tracks[track_id][-1]
                    prev = smoothed_tracks[track_id][-2]
                    
                    # Calculate distance based on the smoothed point
                    distance_px = np.sqrt((curr[0]-prev[0])**2 + (curr[1]-prev[1])**2)
                    distance_mm = distance_px * pixel_to_mm_ratios[track_id]
                    
                    if  distance_mm>1 and distance_mm <= 5 :  # Filter displacement less than 1mm and displacement greater than 5mm
                        total_distances[track_id] += distance_mm
                        all_crab_distance += distance_mm
                #Draw object detection box
                label = f'ID:{track_id}, crab_Conf:{conf:.2f}, Dist:{total_distances[track_id]:.1f}mm'
                cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(203,227,48),thickness=2)   
                cv2.putText(img,label,(int(x1),int(y1)-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(203,227,48),thickness=2)
                
                #paint the center point
                cv2.circle(img,(int(center_x),int(center_y)),5,(203,227,48),-1)#-1:draw a solid circle,5: radius
                
                #paint the trajectory
                points=np.array(smoothed_tracks[track_id],dtype=np.int32).reshape((-1,1,2))
                cv2.polylines(img,[points],False,(203,227,48),thickness=2)
        
    
    # # Write the processed frame to the output video
    video_writer.write(img)

    # Display the frame
    #cv2.imshow('track',img)

    # Exit on pressing 'q'
    #if cv2.waitKey(1) & 0xFF == ord('q'):
        # break

#On average, each crab moves mm per minute
if crab_number:
    avg_crab_permin_distance=(all_crab_distance/crab_number)*fps*60
else:
    avg_crab_permin_distance = 0
#Call the function to obtain the time corresponding to the first and last frames
first_time, last_time = get_first_and_last_time(video_path)

#sort the distance reverse order
sorted_distance=sorted(total_distances.items(),
                    key=lambda x:x[1],reverse=True)
#Write the result
# with open('track_result.txt','a+',encoding='utf-8') as f1:
#      for obj_id,total_distance in sorted_distance:
#        f1.write(f'{first_time}to{last_time},{obj_id},{total_distance:.2f}\n')

with open('result2.txt','a+',encoding='utf-8')as f2:
    f2.write(f'{first_time}to{last_time},{avg_crab_permin_distance:.2f}\n')

#Release resources
capture.release()
video_writer.release()
cv2.destroyAllWindows()

