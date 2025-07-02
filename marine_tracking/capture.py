import cv2
video_path = "E:/hydrothermal/2018-2019/2018_2019/SMOOVE-18-10-02_23-57-03-41.mkv"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error opening video file")
    exit(1)
frame_number =1
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
ret, frame = cap.read()
if ret:
    # 保存截图为图片文件
    screenshot_path = "screenshot.jpg"
    cv2.imwrite(screenshot_path, frame)
    print(f"截图已保存到 {screenshot_path}")
else:
    print("无法读取指定帧")

# 释放资源
cap.release()
cv2.destroyAllWindows()