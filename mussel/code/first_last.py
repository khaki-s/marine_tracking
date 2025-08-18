from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
#---Parameter settings---
#path 
vedeo_path = "E:/hydrothermal/2017-2018/2017_2018_mp4/SMOOVE-17-08-19_00-00-01-83.mp4"
model = YOLO("D:/khaki/ultralytics-8.3.27/runs/detect/train21/weights/best.pt")

data_first = []
data_last = []
#Center point Euclidean distance threshold (unit: pixel)
CENTER_DIS_NO_MOVE = 50 #Lowver than this value no movement
CENTER_DIS_MISS = 180 #Above this value matching is unreasonable
MIN_CONFIDENCE = 0.5 #Minimum threshold
plt.rcParams['font.family'] = 'Arial'
#IOU
IOU_NO_MOVE = 0.4 #IOU above this value indicates no movement
IOU_MISSING = 0.05 #IOU lowver than this value matching is unreasonable
#Size ratio threshold
SIZE_RATIO_MIN = 0.8 #Minimum ratio of box areas
SIZE_RATIO_MAX = 1.5 #Maximum ratio of box areas

#Color settings(BGR)
COLOR_NO_MOVE = (134,219,196)
COLOR_MOVED = (224,210,168)
#COLOR_MISSING = (255,255,255)
COLOR_NEW = (212,187,252)
COLOR_CRAB = (120,212,245)
COLOR_DISAPPEAR = (209,144,164)


#-----Function settings-----
def compute_iou(boxA, boxB):
    """
    Compute Intersection over Union (IoU) between two boxes in [x1, y1, x2, y2] format.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    denominator = float(boxAArea + boxBArea - interArea + 1e-6)
    if denominator <= 0:
        return 0.0
    iou = interArea/denominator
    return iou if iou is not None else 0.0



def get_center(box):
    x_center = int((box[0]+box[2])/2)
    y_center = int((box[1]+box[3])/2)
    return (x_center,y_center)

def get_conf(box):
    conf =box[5]
    return conf

def get_box_area(box):
    """
    Calculate the area of a bounding box
    """
    width = box[2] - box[0]
    height = box[3] - box[1]
    return width * height

def process_frame_pair(detections_prev, detections_curr, image_prev, image_curr):
    """
    Match the detection results of the previous and subsequent frames and label them on the image:
        -For mussels: label according to the matching status (green: not moved; red: moved; gray: disappeared; blue: added), and display the center point distance with a line
        -For crabs: mark them directly in blue
    Return the annotated images of the previous and current frames
    """
    #mussel(cls==2),crab(cls==0)
    mussels_prev = [d for d in detections_prev if d[4] == 2]
    mussels_curr = [d for d in detections_curr if d[4] == 2]
    crab_curr = [d for d in detections_curr if d[4] == 0]

    all_dis = 0
    rate = 0
    move_amount = 0
    no_move_amount = 0 
    #Constructing a base matrix based on the Euclidean distance of the center point for mussels
    if mussels_prev and mussels_curr:
        m,n = len(mussels_prev),len(mussels_curr)
        base_matrix = np.zeros((m,n))
        for i,d_prev in enumerate(mussels_prev):
            center_pre = get_center(d_prev[0:4])
            area_prev = get_box_area(d_prev[0:4])
            for j,d_curr in enumerate(mussels_curr):
                center_curr = get_center(d_curr[0:4])
                area_curr = get_box_area(d_curr[0:4])
                dist = np.linalg.norm(np.array(center_pre)-np.array(center_curr)).item()
                # Calculate size ratio
                size_ratio = max(area_prev, area_curr) / min(area_prev, area_curr)
                # If size ratio is outside acceptable range, set distance to maximum
                if size_ratio < SIZE_RATIO_MIN or size_ratio > SIZE_RATIO_MAX:
                    dist = CENTER_DIS_MISS
                base_matrix[i,j] = dist
        #Customize virtual values
        fill_value = base_matrix.max()+CENTER_DIS_MISS 

        #Construct a square matrix
        size = max(m,n)
        cost_matrix = np.full((size, size), fill_value, dtype=np.float32)
        cost_matrix[:m,:n] = base_matrix
        #Using the Hungarian algorithm to obtain the optimal match
        row_ind,col_ind = linear_sum_assignment(cost_matrix)
        
        #Match indexes that are only retained within the original range
        matches = {}
        for i,j in zip(row_ind,col_ind):
            if i < m and j < n and cost_matrix[i, j] < CENTER_DIS_MISS:
                matches[i]=j
    else:
        matches={}

    #Copy image for annotation
    annotated_prev = image_prev.copy()
    annotated_curr = image_curr.copy()

    #Evaluate the matched mussels
    for i, d_prev in enumerate(mussels_prev):
        center_pre = get_center(d_prev[0:4])
        get_conf = d_prev[5]
        #Annotate the matched items
        if i in matches:
            j = matches[i]
            d_curr = mussels_curr[j]
            center_curr = get_center(d_curr[0:4])
            iou_val = compute_iou(d_prev[0:4],d_curr[0:4])
            if iou_val is None:
                iou_val = 0.0
            dist = np.linalg.norm(np.array(center_pre) - np.array(center_curr)).item()
            #Divide states based on distance and IOU
            if dist < CENTER_DIS_NO_MOVE and iou_val > IOU_NO_MOVE:
                color = COLOR_NO_MOVE
                label = f'still, conf{get_conf:.3f}'
                no_move_amount = no_move_amount +1
            # elif dist >CENTER_DIS_MISS or iou_val < IOU_MISSING:
            #     color = COLOR_MISSING
            #     label = f'missing, conf{get_conf:.3f}'
            else:
                color = COLOR_MOVED
                label = f"translocation in the view{dist:3f}, conf{get_conf:.3f}"
                if dist >= CENTER_DIS_NO_MOVE:
                    all_dis = all_dis+ dist
                move_amount = move_amount + 1
            #Annotate matching mussels on the current frame
            if move_amount+no_move_amount == 0:
                rate = 0
            else:
                rate = move_amount/(move_amount+no_move_amount)
            x1,y1,x2,y2 = map(int,d_curr[0:4])
            cv2.rectangle(annotated_curr,(x1,y1),(x2,y2),color,5)
            cv2.putText(annotated_curr,label,(x1,y1 - 10),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
            cv2.line(annotated_curr,center_pre,center_curr,color,5)
        else:
            #The mussels in the previous frame do not match in the current frame and are marked as'missing'
            x1,y1,x2,y2 = map(int,d_prev[0:4])
            cv2.rectangle(annotated_prev,(x1,y1),(x2,y2),COLOR_DISAPPEAR,5)
            cv2.putText(annotated_prev,"missing",(x1,y1 - 10),cv2.FONT_HERSHEY_SIMPLEX,0.6,COLOR_DISAPPEAR,2)
    #Mark the unmatched mussels in the current frame as' New comer '
    matched_curr_indices = list(matches.values()) 
    for j, d_curr in enumerate(mussels_curr): 
        if j not in matched_curr_indices: 
            x1, y1, x2, y2 = map(int,d_curr[0:4] )
            cv2.rectangle(annotated_curr, (x1, y1), (x2, y2), COLOR_NEW, 5) 
            cv2.putText(annotated_curr, "New comer", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_NEW, 2)
    #Tag Crab
    for d in crab_curr:
        x1,y1,x2,y2 = map(int,d[0:4])
        cv2.rectangle(annotated_curr, (x1, y1), (x2, y2), COLOR_CRAB, 5) 
        cv2.putText(annotated_curr, "crab", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_CRAB, 2)
    
    return annotated_prev,annotated_curr,all_dis,rate



#-----main-----
#1. capture the first and last frame of the video

cap = cv2.VideoCapture(vedeo_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret, frame_fisrt = cap.read()

cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
ret, frame_last = cap.read()
cap.release()


#2. YOLO detection to obtain the first and last frame results
# extract the first frame results
results_first = model(frame_fisrt)
boxs_first = results_first[0].boxes.xyxy.cuda().tolist()
clss_first = results_first[0].boxes.cls.int().cuda().tolist()
confs_first = results_first[0].boxes.conf.cuda().tolist()
for box,cls,conf in zip(boxs_first,clss_first,confs_first):
    x1,y1,x2,y2 = box
    data_first.append([x1,y1,x2,y2,cls,conf])

# extract the last frame results
results_last = model(frame_last)
boxs_last = results_last[0].boxes.xyxy.cuda().tolist()
clss_last = results_last[0].boxes.cls.int().cuda().tolist()
confs_last = results_last[0].boxes.conf.cuda().tolist()
for box,cls,conf in zip(boxs_last,clss_last,confs_last):
    x1,y1,x2,y2 = box
    data_last.append([x1,y1,x2,y2,cls,conf])


annotated_prev, annotated_curr,all_dis,rate = process_frame_pair(data_first,data_last,frame_fisrt,frame_last)
#For ease of observation, merge the previous frame and the current frame for display (left and right stitching)
# Create a white space between frames (20 pixels wide)
height = annotated_prev.shape[0]
white_space = np.ones((height, 20, 3), dtype=np.uint8) * 255
# Concatenate the three images horizontally
combined = np.hstack([annotated_prev, white_space, annotated_curr])
cv2.imwrite("out.png",combined)