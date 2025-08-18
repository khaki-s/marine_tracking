import cv2
from ultralytics import YOLO ,solutions
from easyocr_OCRtime import get_first_and_last_time



model=YOLO('./best_count.pt')#Change the model path

video_path = "E:/hydrothermal/2020-2021/2020-2021_enhance/20-11-26-lab.mp4"
out_path = 'test_enhance.mp4'
capture=cv2.VideoCapture(video_path)
mussel_total_sum=0
capture_number=0


assert capture.isOpened(), 'error reading the video'
#Read the with,height,fps of the video
w,h,fps=(int(capture.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT,cv2.CAP_PROP_FPS))
#1920 1080 25

# Video writer 
video_writer = cv2.VideoWriter(out_path, cv2.VideoWriter.fourcc(*"mp4v"), fps, (w, h))

#declare object counting function
line_points = [(0, 0), (0, 1080)]
counter = solutions.ObjectCounter(
    show=True,  
    region=line_points,  
    model="best_count.pt", #Change the model path
)

while capture.isOpened():
    success,img=capture.read()
    if not success:
        print('read complete')
        break
    result=model(img)
    capture_number+=1
    #Count the number of detection in the current frame
    count=len(result[0])
    mussel_total_sum=mussel_total_sum+count
    #Output the counting number
    text = f'Number of objects: {count}'
    cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    img = counter.count(img)#add frame recognize
    # video_writer.write(img)
    cv2.imshow('count',img)

# #Call the function to obtain the time corresponding to the first and last frames
# first_time, last_time = get_first_and_last_time(video_path)


# #Write the result
# with open('result1.txt','a+',encoding='utf-8') as f1:
#     f1.write(f'{first_time}to{last_time},{mussel_total_sum/capture_number:.2f}\n')

capture.release()
video_writer.release()
cv2.destroyAllWindows()