import cv2
from ultralytics import YOLO 
from collections import defaultdict
import numpy as np
from easyocr_OCRtime import get_first_and_last_time
import os

model=YOLO('../../runs/detect/train28/weights/best.pt')#Change the model path

video_directory='/home/jingyichu/data/2020_2021'
out_path='E:/hydrothermal/2020-2021_enhance/11-27-origin.mp4'

#Ergodic the directory
for filename in os.listdir(video_directory):
    if filename.endswith(('.mp4','.mkv')):
        video_path=os.path.join(video_directory,filename)
        capture=cv2.VideoCapture(video_path)
        assert capture.isOpened(), 'error reading the video'

        #Read the with,height,fps of the video
        w,h,fps=(int(capture.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT,cv2.CAP_PROP_FPS))
        #1920 1080 25

        # # Video writer 
        video_writer=cv2.VideoWriter(out_path, cv2.VideoWriter.fourcc(*"mp4v"), fps, (w, h))

        #Trajectory Storage Dictionary
        track_history=defaultdict(lambda: [])
        #Accumulated distance
        total_distances=defaultdict(float)
        all_crab_distance=0
        #The number of points displayed
        max_trail_length=80


        while capture.isOpened():
            success,img=capture.read()
            if not success:
                print('read complete')
                break
            
            #Execute target tracking
            results=model.track(img,persist=True)#Confirm tracking for each frame
            
            #Put out a part of results
            if results[0].boxes.id is not None:
                boxs = results[0].boxes.xyxy.cuda().tolist()
                track_ids = results[0].boxes.id.int().cuda().tolist()
                clss=results[0].boxes.cls.int().cuda().tolist()
                confs=results[0].boxes.conf.cuda().tolist()
                #traverse
                for box,track_id,cls,conf in zip(boxs,track_ids,clss,confs):
                    if cls == 1 :#0:crab,1:crab_small,2:mussel
                        x1,y1,x2,y2=box
                        #calculate the center coordinates
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        #aquire the history track
                        track=track_history[track_id]
                        track.append((center_x,center_y))

                        #limit trajectory length
                        '''
                        If it is greater than the specified maximum value, delete the first point
                        '''
                        if len(track) >max_trail_length:
                            track.pop(0)
                        
                        #Smooth processing
                        if len(track) >=5:
                            smoothed=np.mean(track[-5:],axis=0)#Take the average of the last five values of the column
                            center_x,center_y=smoothed[0],smoothed[1]
                        
                        #calculate the distance
                        if len(track)>1:
                            prev_x, prev_y = track[-2] if len(track) <5 else np.mean(track[-6:-1], axis=0)
                            #Calculate the Euclidean distance between adjacent points
                            distance=np.sqrt((center_x - prev_x)**2 + (center_y - prev_y)**2)
                            if distance > 4:
                                total_distances[track_id] += distance
                                all_crab_distance += distance
                        #Draw object detection box
                        label=f'ID:{track_id},crab:{conf:.2f},distance:{total_distances[track_id]:.1f}px'
                        cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(203,227,48),thickness=2)   
                        cv2.putText(img,label,(int(x1),int(y1)-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(203,227,48),thickness=2)
                        
                        #paint the center point
                        cv2.circle(img,(int(center_x),int(center_y)),5,(203,227,48),-1)#-1:draw a solid circle,5: radius
                        
                        #paint the trajectory
                        points=np.array(track,dtype=np.int32).reshape((-1,1,2))
                        cv2.polylines(img,[points],False,(203,227,48),thickness=2)

            # # Write the processed frame to the output video
            video_writer.write(img)

            # Display the frame
            cv2.imshow('track',img)

            # Exit on pressing 'q'
            #if cv2.waitKey(1) & 0xFF == ord('q'):
                #break

        #Call the function to obtain the time corresponding to the first and last frames
        first_time, last_time = get_first_and_last_time(video_path)

        #sort the distance reverse order
        sorted_distance=sorted(total_distances.items(),
                            key=lambda x:x[1],reverse=True)
        #Write the result
        # with open('track_result.txt','a+',encoding='utf-8') as f1:
        #     for obj_id,total_distance in sorted_distance:
        #         f1.write(f'{first_time}to{last_time},{obj_id},{total_distance:.2f}\n')

        with open('sum_of_all_crab_small_distances.txt','a+',encoding='utf-8')as f2:
            f2.write(f'{first_time}to{last_time},{all_crab_distance:.2f}\n')

        #Release resources
        capture.release()
        video_writer.release()
        cv2.destroyAllWindows()

