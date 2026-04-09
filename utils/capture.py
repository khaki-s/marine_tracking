import cv2
video_path = "E:/hydrothermal/2018-2019/2018_2019/SMOOVE-18-10-02_23-57-03-41.mkv"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error opening video file")
    exit(1)
frame_number = 1
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
ret, frame = cap.read()
if ret:
    # Save the frame as an image file
    screenshot_path = "screenshot.jpg"
    cv2.imwrite(screenshot_path, frame)
    print(f"Screenshot saved to {screenshot_path}")
else:
    print("Failed to read the specified frame")

# Release resources
cap.release()
cv2.destroyAllWindows()
