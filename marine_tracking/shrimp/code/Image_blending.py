import cv2
import numpy as np
import cv2
import numpy as np

def process_png_to_transparent(png_path, target_size):
    """将PNG图像处理为透明背景的RGBA格式"""
    # 读取PNG图像并保留Alpha通道
    img = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"PNG文件未找到：{png_path}")
    
    # 转换色彩空间并确保四通道
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    
    # 调整尺寸时保持通道数不变
    img = cv2.resize(img, target_size)
    
    # 将白色背景转换为透明 (BGR阈值判断)
    white_threshold = 240
    b, g, r, a = cv2.split(img)
    mask = np.logical_and.reduce([b > white_threshold, 
                                g > white_threshold, 
                                r > white_threshold])
    a[mask] = 0
    return cv2.merge((b, g, r, a))

def blend_images(jpg_path, png_path, output_path):
    """主合成函数"""
    # 读取并处理JPG图像
    jpg_img = cv2.imread(jpg_path, cv2.IMREAD_COLOR).astype(np.float32)

    
    # 处理PNG图像
    target_size = (jpg_img.shape[1], jpg_img.shape[0])  # (width, height)
    transparent_image = process_png_to_transparent(png_path, target_size)

    # 提取并准备Alpha通道
    alpha = transparent_image[:, :, 3:4].astype(np.float32) / 255.0  # 直接保持(H,W,1)
    
    #合成
    blended = (
        alpha * transparent_image[:, :, :3]*0.8 + 
        (1 - alpha) * jpg_img*0.8
    )
    
    # 后处理并保存
    cv2.imwrite(output_path, blended.clip(0, 255).astype(np.uint8))

if __name__=="__main__":
    blend_images(
        jpg_path="E:/hydrothermal/shrimp/trajectory/screenshot.jpg",
        png_path="E:/hydrothermal/shrimp/trajectory/SMOOVE-18-10-25_23-57-45-96.png",
        output_path="E:/hydrothermal/shrimp/trajectory/2018-10-25.png"
    )

