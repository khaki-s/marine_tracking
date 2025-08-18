import cv2
import os

video_path="E:/hydrothermal/2020-2021/2020-2021/SMOOVE-20-09-30_23-55-47-96.mp4"
output_path="E:/hydrothermal/2020-2021/crab"

cap=cv2.VideoCapture(video_path)
if not cap.isOpened():
    print('Error opening the video file')
    exit()

num=0#The Counter
save_step=200#Interval frame
count=0
while True:
    success,frame=cap.read()
    if not success:
        print('error reading the video')
        break
    num+=1
    if num % save_step ==0:#Take every thirty frames
        count +=1
        cv2.imwrite(os.path.join(output_path, f'{count}.jpg'), frame)

cap.release() # Release video resources
cv2.destroyAllWindows()


# #extract mussel
# video_directory="E:/hydrothermal/2017-2018/2017_2018"
# out_path  ="D:/khaki/ultralytics-8.3.27/mussel/out/2017-2018"
# count = 0
# #x=0
# for filename in os.listdir(video_directory):
#     if filename.endswith('.mkv') or filename.endswith('.mp4'):
#         video_path = os.path.join(video_directory,filename)
#         base_name = os.path.splitext(os.path.basename(video_path))[0]
#         capture = cv2.VideoCapture(video_path)
#         assert capture.isOpened(),"error reading the video"
#         #obtain sum of frames
#         total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
#         print(f"Video: {filename}, Total Frames: {total_frames}")
#         # set the last frame
#         capture.set(cv2.CAP_PROP_POS_FRAMES,1)
#         #x=x+200
#         success,frame = capture.read()
#         if success:
#             count += 1
#             cv2.imwrite(os.path.join(out_path,f"{base_name}.jpg"),frame)
#     capture.release()
# cv2.destroyAllWindows()


