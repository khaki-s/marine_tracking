import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2
import os 
from glob import glob
import pandas as pd
import re
import csv
import matplotlib.pyplot as plt

#---Parameter settings---
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
def extract_timestamp(filename):
    match = re.search(r'(\d{2}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', filename)
    return match.group(1) if match else None

def numerical_sort(value):
    #Extract numbers from file names
    numbers = re.findall(r'\d+', os.path.basename(value))
    return int(numbers[0]) if numbers else 0

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
    conf =box[6]
    return conf
def get_box_area(box):
    """
    Calculate the area of a bounding box
    """
    width = box[2] - box[0]
    height = box[3] - box[1]
    return width * height
def load_detections_from_file(csv_path, target_filename):
    """
    Load the detection results of the target file from the CSV file
    :param csv_path: YOLO result CSV file path
    :param target_filename: the image file name that needs to be searched
    :return: detect the target list, with each element being {'cls': cls, 'bbox': [x1, y1, x2, y2], 'conf': conf}
    """
    df = pd.read_csv(csv_path)
    df = df[df['filename']== target_filename]
    df = df[df['conf'] >= MIN_CONFIDENCE]
    detections = []
    for _, row in df.iterrows():
        detections.append({
            'cls': int(row['cls']),
            'bbox': [int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])],
            'conf': float(row['conf'])
        })
    return detections

def process_frame_pair(detections_prev, detections_curr, image_prev, image_curr):
    """
    Match the detection results of the previous and subsequent frames and label them on the image:
        -For mussels: label according to the matching status (green: not moved; red: moved; gray: disappeared; blue: added), and display the center point distance with a line
        -For crabs: mark them directly in blue
    Return the annotated images of the previous and current frames
    """
    #mussel(cls==2),crab(cls==0)
    mussels_prev = [d for d in detections_prev if d['cls'] == 2]
    mussels_curr = [d for d in detections_curr if d['cls'] == 2]
    crab_curr = [d for d in detections_curr if d['cls'] == 0]

    all_dis = 0
    rate = 0
    move_amount = 0
    no_move_amount = 0 
    #Constructing a base matrix based on the Euclidean distance of the center point for mussels
    if mussels_prev and mussels_curr:
        m,n = len(mussels_prev),len(mussels_curr)
        base_matrix = np.zeros((m,n))
        for i,d_prev in enumerate(mussels_prev):
            center_pre = get_center(d_prev['bbox'])
            area_prev = get_box_area(d_prev['bbox'])
            for j,d_curr in enumerate(mussels_curr):
                center_curr = get_center(d_curr['bbox'])
                area_curr = get_box_area(d_prev['bbox'])
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
        center_pre = get_center(d_prev['bbox'])
        get_conf = d_prev['conf']
        #Annotate the matched items
        if i in matches:
            j = matches[i]
            d_curr = mussels_curr[j]
            center_curr = get_center(d_curr['bbox'])
            iou_val = compute_iou(d_prev['bbox'],d_curr['bbox'])
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
            x1,y1,x2,y2 = d_curr['bbox']
            cv2.rectangle(annotated_curr,(x1,y1),(x2,y2),color,5)
            cv2.putText(annotated_curr,label,(x1,y1 - 10),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
            cv2.line(annotated_curr,center_pre,center_curr,color,5)
        else:
            #The mussels in the previous frame do not match in the current frame and are marked as'missing'
            x1,y1,x2,y2 = d_prev['bbox']
            cv2.rectangle(annotated_prev,(x1,y1),(x2,y2),COLOR_DISAPPEAR,5)
            cv2.putText(annotated_prev,"missing",(x1,y1 - 10),cv2.FONT_HERSHEY_SIMPLEX,0.6,COLOR_DISAPPEAR,2)
    #Mark the unmatched mussels in the current frame as' New comer '
    matched_curr_indices = list(matches.values()) 
    for j, d_curr in enumerate(mussels_curr): 
        if j not in matched_curr_indices: 
            x1, y1, x2, y2 = d_curr['bbox'] 
            cv2.rectangle(annotated_curr, (x1, y1), (x2, y2), COLOR_NEW, 5) 
            cv2.putText(annotated_curr, "New comer", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_NEW, 2)
    #Tag Crab
    for d in crab_curr:
        x1,y1,x2,y2 = d['bbox']
        cv2.rectangle(annotated_curr, (x1, y1), (x2, y2), COLOR_CRAB, 5) 
        cv2.putText(annotated_curr, "crab", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_CRAB, 2)
    
    return annotated_prev,annotated_curr,all_dis,rate


#-----main-----
frames_folder="D:/khaki/ultralytics-8.3.27/mussel/mussel"#Folder for storing each frame of images
csv_path = 'D:/khaki/ultralytics-8.3.27/mussel/results/2017-2018.csv'#YOLO results
output_folder = "D:/khaki/ultralytics-8.3.27/mussel/out/2017-2018"# Out put folder
csv_output = 'D:/khaki/ultralytics-8.3.27/mussel/results/move_dis_and_rate-2017-2018.csv'
#Read all images in the folder (sorted)
image_files = sorted(glob(os.path.join(frames_folder, '*.jpg')), key=numerical_sort)

with open(csv_output, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['starttime', 'endtime', 'rate', 'all_distance'])
prev_detections = None
prev_image = None
prev_base = None

for idx,image_path in enumerate(image_files):
    image = cv2.imread(image_path)
    base_name = os.path.basename(image_path)
    print("Processing image:", base_name)
    prev_detections = load_detections_from_file(csv_path, prev_base) if prev_base else None
    detections = load_detections_from_file(csv_path, base_name)
    
    if prev_detections is None and len(detections) > 0:
        prev_detections = detections
        prev_image = image
        prev_base = base_name
        continue
    else:
        annotated_prev, annotated_curr,all_dis,rate = process_frame_pair(prev_detections,detections,prev_image,image)
        start_time = extract_timestamp(prev_base)
        end_time = extract_timestamp(base_name)
        name = f"{start_time}to{end_time}.pdf" if start_time and end_time else f"{prev_base}_{base_name}.pdf"
        
        with open(csv_output,'a', newline='', encoding='utf-8') as f1:
            writer = csv.writer(f1)
            writer.writerow([start_time, end_time, f"{rate:.4f}", f"{all_dis:.2f}"])
        
        #For ease of observation, merge the previous frame and the current frame for display (left and right stitching)
        # Create a white space between frames (20 pixels wide)
        height = annotated_prev.shape[0]
        white_space = np.ones((height, 20, 3), dtype=np.uint8) * 255
        # Concatenate the three images horizontally
        combined = np.hstack([annotated_prev, white_space, annotated_curr])
        out_path = os.path.join(output_folder,name)
        if combined is not None:
            print(f"Saving output to: {out_path}, shape: {combined.shape}")
            # Convert BGR to RGB for matplotlib
            combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            # Create figure and axis with high DPI
            dpi = 300  # High resolution
            # Calculate figure size based on image dimensions and DPI
            height, width = combined_rgb.shape[:2]
            fig_width = width / dpi
            fig_height = height / dpi
            fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
            # Display image
            ax.imshow(combined_rgb)
            # Remove axes
            ax.axis('off')
            # Save as PDF with high quality
            plt.savefig(out_path, 
                    bbox_inches='tight', 
                    pad_inches=0,
                    dpi=dpi,
                    format='pdf',
                    metadata={'Creator': '', 'Producer': ''})
            plt.close()
        else:
            print(f"Error: Combined image is None for {prev_base} and {base_name}")
        #update data
        #prev_detections = detections
        prev_image = image
        prev_base = base_name
print('Processing completed, the annotated image is saved in',output_folder)


