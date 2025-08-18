import cv2
from ultralytics import YOLO
from easyocr_OCRtime import get_first_and_last_time
import os


model=YOLO('../../runs/detect/train18/weights/best.pt')

video_directory = "E:/hydrothermal/test"
#out_path = 'E:/hydrothermal/test/1.mp4'

mussel_total_sum = 0
capture_number=0

for filename in os.listdir(video_directory):
    if filename.endswith(('.mp4','mkv')):
        video_path=os.path.join(video_directory,filename)
        capture=cv2.VideoCapture(video_path)
        assert capture.isOpened(),'error reading the video'

        #Read the with,height,fps of the video
        w,h,fps=(int(capture.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT,cv2.CAP_PROP_FPS))
        #1920 1080 25

        # Video writer 
        #video_writer=cv2.VideoWriter(out_path, cv2.VideoWriter.fourcc(*"mp4v"), fps, (w, h))

        while capture.isOpened():
            
            success,img=capture.read()
            if not success:
                print('read complete')
                break
            
            #Execute target tracking
            results=model.track(img,persist=True)#Confirm tracking for each frame
            count=0  # Reset count for each frame

            #Put out a part of results
            if results[0].boxes.id is not None:
                boxs = results[0].boxes.xyxy.cuda().tolist()
                track_ids = results[0].boxes.id.int().cuda().tolist()
                clss=results[0].boxes.cls.int().cuda().tolist()
                confs=results[0].boxes.conf.cuda().tolist()
                #traverse
                for box,track_id,cls,conf in zip(boxs,track_ids,clss,confs):
                    if cls == 5:
                        count +=1
                        mussel_total_sum += 1 
                        x1,y1,x2,y2=box
                        #calculate the center coordinates
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        #Draw object detection box
                        label=f'ID:{track_id},mussel:{conf:.2f}'
                        cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(203,227,48),thickness=2)   
                        cv2.putText(img,label,(int(x1),int(y1)-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(203,227,48),thickness=2)
            
            capture_number+=1
            #Output the counting number
            text = f'Number of objects: {count}'
            cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            #cv2.imshow('count',img)
            #video_writer.write(img)

        #Call the function to obtain the time corresponding to the first and last frames
        first_time, last_time = get_first_and_last_time(video_path)


        #Write the result
        with open('result2.txt','a+',encoding='utf-8') as f1:
            f1.write(f'{first_time}to{last_time},{mussel_total_sum/capture_number:.2f}\n')

        capture.release()
        #video_writer.release()
        cv2.destroyAllWindows()