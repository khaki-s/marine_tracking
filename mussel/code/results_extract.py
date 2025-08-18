from ultralytics import YOLO
import os
import csv
picture_directory = "D:/khaki/ultralytics-8.3.27/mussel/mussel"
csv_file = "D:/khaki/ultralytics-8.3.27/mussel/results/2017-2018.csv"
model = YOLO("D:/khaki/ultralytics-8.3.27/runs/detect/train21/weights/best.pt")

data = []
for filename in os.listdir(picture_directory):
    if filename.lower().endswith('.jpg'):
        picture_path = os.path.join(picture_directory,filename)
        #Operation prediction
        results = model(picture_path)
        #Process each image
        if len(results)>=0:
            boxs = results[0].boxes.xyxy.cuda().tolist()
            clss = results[0].boxes.cls.int().cuda().tolist()
            confs = results[0].boxes.conf.cuda().tolist()
            #traverse
            for box,cls,conf in zip(boxs,clss,confs):
                x1,y1,x2,y2 =box

                data.append([filename,cls,x1,y1,x2,y2,conf])
#Write to csv 
def extract_number(filename):
    return int(''.join(filter(str.isdigit,filename)))

data.sort(key=lambda x :extract_number(x[0]))

with open(csv_file, 'w', newline='', encoding='utf-8') as f1:
    writer = csv.writer(f1)
    writer.writerow(['filename', 'cls', 'x1', 'y1', 'x2', 'y2', 'conf'])
    for item in data:
        fields = [str(field) for field in item]
        writer.writerow(fields)