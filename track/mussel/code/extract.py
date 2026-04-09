import cv2
import os

video_path = "E:/hydrothermal/2017-2018/2017_2018_mp4/SMOOVE-18-01-14_23-57-56-17.mp4"
out_path = "D:/khaki/ultralytics-8.3.27/mussel/mussel"
count = 0
capture = cv2.VideoCapture(video_path)
assert capture.isOpened(),"error reading the video"
total_frames = int (capture.get(cv2.CAP_PROP_FRAME_COUNT))
capture.set(cv2.CAP_PROP_POS_FRAMES,total_frames//2)
success,frame = capture.read()
if success:
    count = count+1
    cv2.imwrite(os.path.join(out_path,f"155.jpg"),frame)
capture.release()
cv2.destroyAllWindows()