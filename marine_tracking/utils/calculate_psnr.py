import cv2
import numpy as np

def calculate_psnr(original, enhanced):
    mse = np.mean((original - enhanced) ** 2)
    if mse == 0:
        return float('inf')  # 两张图片完全一致
    max_pixel = 255.0  # 假设图像是8位的
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_uciqe(image, c1=0.3, c2=0.3, c3=0.4):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sigma_c = np.std(hsv_image[:, :, 1])  # 饱和度通道的标准差
    brightness = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    con_l = (brightness.max() - brightness.min()) / 255.0  # 归一化
    mu_s = np.mean(hsv_image[:, :, 1]) / 255.0  # 归一化
    uciqe = c1 * sigma_c + c2 * con_l + c3 * mu_s
    return np.clip(uciqe, 0, 1)

def calculate_uiqm(image):
    # UIQM 计算的简化示例
    # 请根据您的需求添加具体的实现
    brightness = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = (brightness.max() - brightness.min()) / 255.0
    uiqm = contrast  # 这只是一个简化示例，实际计算可能更复杂
    return np.clip(uiqm, 0, 1)

def evaluate_images(original_path, enhanced_path):
    # 加载图像
    original = cv2.imread(original_path)
    enhanced = cv2.imread(enhanced_path)

    if original is None or enhanced is None:
        raise ValueError("无法加载图像，请检查路径")

    # 计算PSNR
    psnr_value = calculate_psnr(original, enhanced)
    print(f"PSNR: {psnr_value:.2f} dB")

    # 计算UCIQE
    uciqe_value = calculate_uciqe(enhanced)
    print(f"UCIQE: {uciqe_value:.4f}")

    # 计算UIQM
    uiqm_value = calculate_uiqm(enhanced)
    print(f"UIQM: {uiqm_value:.4f}")

# 示例执行
original_image_path = 'E:/hydrothermal/2020-2021_enhance/origin.jpg'  # 替换为目标图像路径
enhanced_image_path ="E:/hydrothermal/2020-2021_enhance/13200.jpg"  # 替换为增强图像路径

evaluate_images(original_image_path, enhanced_image_path)




