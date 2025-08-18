import cv2
import numpy as np

def calculate_psnr(original, enhanced):
    # Calculate the Mean Squared Error (MSE) between the original and enhanced images
    mse = np.mean((original - enhanced) ** 2)
    
    # If MSE is zero, the images are identical, and PSNR is infinity
    if mse == 0:
        return float('inf')
    
    # Maximum pixel value for an 8-bit image
    max_pixel = 255.0
    
    # Calculate the Peak Signal-to-Noise Ratio (PSNR)
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_uciqe(image, c1=0.3, c2=0.3, c3=0.4):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Calculate the standard deviation of the saturation channel
    sigma_c = np.std(hsv_image[:, :, 1])  # Saturation channel
    
    # Convert the image to grayscale for brightness calculation
    brightness = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate contrast in the brightness channel
    con_l = (brightness.max() - brightness.min()) / 255.0  # Normalized
    
    # Calculate mean of the saturation channel
    mu_s = np.mean(hsv_image[:, :, 1]) / 255.0  # Normalized
    
    # Calculate the UCIQE score using the given coefficients
    uciqe = c1 * sigma_c + c2 * con_l + c3 * mu_s
    return np.clip(uciqe, 0, 1)

def calculate_uiqm(image):
    # Simplified example for UIQM calculation
    # Please add the full implementation based on your requirements
    brightness = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = (brightness.max() - brightness.min()) / 255.0  # Normalized
    uiqm = contrast  # This is a simplified example; actual calculation may be more complex
    return np.clip(uiqm, 0, 1)

def evaluate_images(original_path, enhanced_path):
    # Load the original and enhanced images
    original = cv2.imread(original_path)
    enhanced = cv2.imread(enhanced_path)

    if original is None or enhanced is None:
        raise ValueError("Failed to load images. Please check the file paths.")

    # Calculate PSNR
    psnr_value = calculate_psnr(original, enhanced)
    print(f"PSNR: {psnr_value:.2f} dB")

    # Calculate UCIQE
    uciqe_value = calculate_uciqe(enhanced)
    print(f"UCIQE: {uciqe_value:.4f}")

    # Calculate UIQM
    uiqm_value = calculate_uiqm(enhanced)
    print(f"UIQM: {uiqm_value:.4f}")

# Example execution
original_image_path = 'E:/hydrothermal/2020-2021_enhance/origin.jpg'  # Replace with the path to the original image
enhanced_image_path = "E:/hydrothermal/2020-2021_enhance/13200.jpg"  # Replace with the path to the enhanced image

evaluate_images(original_image_path, enhanced_image_path)
