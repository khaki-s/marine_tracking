import cv2
import os
import tempfile
import numpy as np



def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def clahe_enhance_video(input_path, output_path, clip_limit=2, grid_size=(16, 16), gamma=1.2):

    # Read the video and frame it
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("error reading the video")
    
    # Retrieve video attributes
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    
    # Write the video
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Frame by frame processing
    for frame_idx in range(total_frames):
        success,img = cap.read()
        if not success:
            break
        
        # CLAHE enhance
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l,a,b=cv2.split(lab)
        
        l_clahe=clahe.apply(l)

        lab_clahe=cv2.merge([l_clahe,a,b])
        frame_clahe=cv2.cvtColor(lab_clahe,cv2.COLOR_Lab2BGR)
        
        # Gamma
        frame_corrected = adjust_gamma(frame_clahe, gamma)
        
        # Write the video
        out.write(frame_corrected)
        
        # Progress notification
        if frame_idx % 100 == 0:
            print(f"having processing {frame_idx}/{total_frames} frames...")
    
    # Release the resource
    cap.release()
    out.release()



# main
if __name__ == "__main__":
    video_directory='E:/hydrothermal/test'
    output_dir='E:/hydrothermal/2020-2021_enhance'
    for filename in os.listdir(video_directory):
        if filename.endswith(('.mp4','.mkv')):
            video_path=os.path.join(video_directory,filename)
            out_path=os.path.join(output_dir,filename)
            clahe_enhance_video(
                input_path=video_path,
                output_path=out_path,
                clip_limit=2.0, 
                grid_size=(16, 16),
                gamma=1.2)
