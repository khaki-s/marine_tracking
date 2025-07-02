import cv2
import os
import tempfile

def clahe_enhance_video(input_path, output_path, clip_limit=2.5, grid_size=(16,16)):
    """
    完整流程：读取视频 → CLAHE逐帧增强 → 直接输出视频（无需保存中间帧）
    参数：
        input_path: 输入视频路径（如 "deepsea_dark.mp4"）
        output_path: 输出视频路径（如 "deepsea_clahe.mp4"）
        clip_limit: 对比度限制（推荐2.0-4.0）
        grid_size: 分块大小（如(8,8)）
    """
    # 创建临时文件夹存放分帧（自动清理）
    with tempfile.TemporaryDirectory() as temp_dir:
        # 1. 读取视频并分帧
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("无法打开输入视频")
        
        # 获取视频属性
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 2. 初始化CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        
        # 3. 创建输出视频写入器
        fourcc = cv2.VideoWriter.fourcc(*'XVID')#*'XVID',*'mp4v'
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 4. 逐帧处理
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            # CLAHE增强（仅处理Y通道）
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            y, u, v = cv2.split(yuv)
            y_clahe = clahe.apply(y)
            yuv_clahe = cv2.merge([y_clahe, u, v])
            frame_clahe = cv2.cvtColor(yuv_clahe, cv2.COLOR_YUV2BGR)
            
            # 直接写入输出视频（无需保存中间帧）
            out.write(frame_clahe)
            
            # 进度提示（可选）
            if frame_idx % 100 == 0:
                print(f"已处理 {frame_idx}/{total_frames} 帧...")
        
        # 5. 释放资源
        cap.release()
        out.release()
        print(f"处理完成！输出视频已保存至：{output_path}")

# 使用示例
if __name__ == "__main__":
    clahe_enhance_video(
        input_path="E:/hydrothermal/2020-2021_enhance/SMOOVE-20-12-23_23-58-15-28.mp4",   # 替换为你的输入视频路径
        output_path="E:/hydrothermal/2020-2021_enhance/12-23test_enhance2.mp4" ,# 输出视频路径
        clip_limit=2.0,                 # 对比度限制（根据效果调整）
        grid_size=(16, 16)                # 分块大小
    )