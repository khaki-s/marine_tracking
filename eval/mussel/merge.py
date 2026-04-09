import cv2
import os
from datetime import datetime
directory = "D:/khaki/ultralytics-8.3.27/mussel/mussel"
output_dir = "D:/khaki/ultralytics-8.3.27/eval/mussel/out"
os.makedirs(output_dir,exist_ok=True)
files = os.listdir(directory)
count =0
i=0
for i in range(len(files)-1):
    file1 = files[i]
    file2 = files[i+1]
    year1 = int(file1[7:9])
    year2 = int(file2[7:9])
    moth1 = int(file1[10:12])
    moth2 = int(file2[10:12])
    date1 = int(file1[13:15])
    date2 = int(file2[13:15])
    dt1 = datetime(year=2000 + year1, month=moth1, day=date1)
    dt2 = datetime(year=2000 + year1, month=moth2, day=date2)
    if (dt2 - dt1).days == 1:
        path1 = os.path.join(directory,file1)
        path2 = os.path.join(directory,file2)
        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)
        merge = cv2.hconcat([img1,img2])
        output_name = f"{year1}-{moth1}-{date1}__{year2}-{moth2}-{date2}.jpg"
        output_path = os.path.join(output_dir,output_name)
        cv2.imwrite(output_path,merge)
        count = count+1
print(f"一共拼接{count}张图片")