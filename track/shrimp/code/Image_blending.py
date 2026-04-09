import cv2
import numpy as np
import cv2
import numpy as np

def process_png_to_transparent(png_path, target_size):

    img = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"PNG not found: {png_path}")
    

    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    

    img = cv2.resize(img, target_size)

    white_threshold = 240
    b, g, r, a = cv2.split(img)
    mask = np.logical_and.reduce([b > white_threshold, 
                                g > white_threshold, 
                                r > white_threshold])
    a[mask] = 0
    return cv2.merge((b, g, r, a))

def blend_images(jpg_path, png_path, output_path):

    jpg_img = cv2.imread(jpg_path, cv2.IMREAD_COLOR).astype(np.float32)

    target_size = (jpg_img.shape[1], jpg_img.shape[0])  # (width, height)
    transparent_image = process_png_to_transparent(png_path, target_size)

    alpha = transparent_image[:, :, 3:4].astype(np.float32) / 255.0  
    
    blended = (
        alpha * transparent_image[:, :, :3]*0.8 + 
        (1 - alpha) * jpg_img*0.8
    )
    
    cv2.imwrite(output_path, blended.clip(0, 255).astype(np.uint8))

if __name__=="__main__":
    blend_images(
        jpg_path=r"E:\graduate\hydrothermal\2018-2019\crab\12.jpg",
        png_path=r"E:\graduate\hydrothermal\shrimp\trajectory\new\2018-2019\SMOOVE-19-02-23_00-01-21-64.png",
        output_path=r"E:\graduate\hydrothermal\shrimp\trajectory\new\2018-2019\19-02-23.png"
    )

