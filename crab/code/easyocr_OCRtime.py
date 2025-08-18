import cv2
import easyocr

def get_first_and_last_time(vidio_path):

    capture = cv2.VideoCapture(vidio_path)
    reader = easyocr.Reader(['en'], gpu=True)

    assert capture.isOpened(), 'error reading the video'
    # Calculate the total frame rate of the video
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read the first frame
    success, first_frame = capture.read()
    if success:
        cropped_first_frame = first_frame[0:30, 0:400]  # Crop the upper left corner area
        first_time_result = reader.readtext(cropped_first_frame, detail=0, allowlist='0123456789-: ')
        first_time = None
        for line in first_time_result:
            if '20' in line:
                first_time = line.strip()
                break

    # Read the last frame
    capture.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)  # Set to the last frame
    success, last_frame = capture.read()
    if success:
        cropped_last_frame = last_frame[0:30, 0:400]
        last_time_result = reader.readtext(cropped_last_frame, detail=0, allowlist='0123456789-: ')
        last_time = None
        for line in last_time_result:
            if '20' in line:
                last_time = line.strip()
                break



    capture.release()
    return first_time,last_time



def get_time(frame):
    reader = easyocr.Reader(['en'], gpu=True)
    reader = easyocr.Reader(['en'], gpu=True)
    cropped_frame = frame[0:30, 0:400]  #  Crop the upper left corner area
    time_result = reader.readtext(cropped_frame, detail=0, allowlist='0123456789-: ')
    for line in time_result:
        if '20' in line:
            return line.strip()
    return "UNKNOWN_TIME"
