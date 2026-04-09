import cv2
from ultralytics import YOLO 
from collections import defaultdict
import numpy as np
import os
import re
import csv


capture_number = 0
shrimp_number = 0
smoothed_window_size = 12
model=YOLO('/home/jingyichu/app/marine-tracking/runs/detect/train32/weights/best.pt')#Change the model path

video_directory="/home/jingyichu/data/2016_2017/VIDEOS"
#out_folder='E:/hydrothermal/2017-2018/shrimp/2017-7-23_test.mp4'


#Ergodic the directory
for filename in os.listdir(video_directory):
    if filename.endswith(('.mp4','.mkv')):
        video_path=os.path.join(video_directory,filename)
        total_frame_count = 0
        #out_path = os.path.join(out_folder,filename)
        capture=cv2.VideoCapture(video_path)
        assert capture.isOpened(), 'error reading the video'

        #Read the with,height,fps of the video
        w,h,fps=(int(capture.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT,cv2.CAP_PROP_FPS))
        #1920 1080 25

        # # Video writer 
        #video_writer=cv2.VideoWriter(out_path, cv2.VideoWriter.fourcc(*"mp4v"), fps, (w, h))
        # Reset per-video counters
        capture_number = 0
        shrimp_number = 0
        #Trajectory Storage Dictionary
        raw_tracks = defaultdict(lambda: [])
        smoothed_tracks = defaultdict(lambda:[])

        #Accumulated distance
        total_distances=defaultdict(float)
        all_shrimp_distance=0
        #The number of points displayed
        max_trail_length=80

        # Body length dictionary: store diagonal bbox length (px) per track
        body_lengths_px = {}

        while capture.isOpened():
            success,img=capture.read()
            if not success:
                print('read complete')
                break
            capture_number+=1
            
            #Execute target tracking
            results=model.track(img,persist=True,
                        show =False,
                        tracker="shrimp-botsort.yaml",
                        iou=0.5,
                        conf=0.1)
            #Put out a part of results
            if results[0].boxes.id is not None:
                boxs = results[0].boxes.xyxy.cuda().tolist()
                track_ids = results[0].boxes.id.int().cuda().tolist()
                clss=results[0].boxes.cls.int().cuda().tolist()
                confs=results[0].boxes.conf.cuda().tolist()
                # Count shrimp in current frame
                shrimp_number += clss.count(1)
                total_frame_count += clss.count(1)
                #traverse
                for box,track_id,cls,conf in zip(boxs,track_ids,clss,confs):
                    if cls == 1 and conf>=0.5:#0:mussel,1:shrimp
                        
                        x1,y1,x2,y2=box
                        body_lengths = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                        
                        if track_id not in body_lengths_px:
                            body_lengths_px[track_id] = body_lengths
                        else:
                            body_lengths_px[track_id] = 0.8 * body_lengths_px[track_id] + 0.2 * body_lengths

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
                            smoothed_tracks[track_id].pop(0)
                        
                        #calculate the distance
                        if len(smoothed_tracks[track_id]) >= 2:
                            curr = smoothed_tracks[track_id][-1]
                            prev = smoothed_tracks[track_id][-2]
                            distance_px = np.sqrt((curr[0]-prev[0])**2 + (curr[1]-prev[1])**2)
                            # Calculate distance based on the smoothed point
                            if body_lengths_px[track_id] > 0:
                                distance_bl = distance_px / body_lengths_px[track_id]
                            else:
                                distance_bl = 0

                            if 0.02 < distance_bl <= 0.5:
                                total_distances[track_id] += distance_bl
                                all_shrimp_distance += distance_bl
                        #Draw object detection box
                        label = f'ID:{track_id}, shrimp_Conf:{conf:.2f}, Dist:{total_distances[track_id]:.2f}BL'
                        cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(203,227,48),thickness=2)   
                        cv2.putText(img,label,(int(x1),int(y1)-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(203,227,48),thickness=2)
                        
                        #paint the center point
                        cv2.circle(img,(int(center_x),int(center_y)),5,(203,227,48),-1)#-1:draw a solid circle,5: radius
                        
                        #paint the trajectory
                        points=np.array(smoothed_tracks[track_id],dtype=np.int32).reshape((-1,1,2))
                        cv2.polylines(img,[points],False,(203,227,48),thickness=2)
                
            
            # # Write the processed frame to the output video
            #video_writer.write(img)

            # Display the frame
            #cv2.imshow('track',img)

            # Exit on pressing 'q'
            #if cv2.waitKey(1) & 0xFF == ord('q'):
                # break
    filename = os.path.basename(video_path)            
    match = re.search(r'SMOOVE-(\d{2}-\d{2}-\d{2})_', filename)
    if match:
        video_date = match.group(1)
    else:
        video_date = "unknown"
    #On average, each shrimp moves mm per minute
    shrimp_number = total_frame_count / capture_number if capture_number > 0 else 0
    if shrimp_number:
        avg_shrimp_permin_distance=(all_shrimp_distance/shrimp_number)
    else:
        avg_shrimp_permin_distance = 0

    #Write the result
    # with open('track_result.txt','a+',encoding='utf-8') as f1:
    #      for obj_id,total_distance in sorted_distance:
    #        f1.write(f'{first_time}to{last_time},{obj_id},{total_distance:.2f}\n')


    csv_path = '/home/jingyichu/app/marine-tracking/track/shirmp/results/2016-2017.csv'
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['time', 'distance_permin_pernumber', 'shrimp_number'])
        writer.writerow([video_date, f'{avg_shrimp_permin_distance:.4f}', shrimp_number])



    #Release resources
    capture.release()
    #video_writer.release()
    #cv2.destroyAllWindows()

