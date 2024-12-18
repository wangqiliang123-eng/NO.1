from paddleocr import PaddleOCR
import cv2
import os
import time

def extract_subtitles(video_path, output_dir='output'):
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 获取视频文件名（不含扩展名）
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_file = os.path.join(output_dir, f"{video_name}_subtitles.txt")
    
    # 初始化PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='ch')  # 中文模型
    
    print(f"正在处理视频: {video_path}")
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return False
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    
    # 用于存储已识别的文本，避免重复
    previous_text = set()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 每秒处理一帧
            if frame_count % int(fps) == 0:
                # 截取底部区域（通常是字幕区域）
                height = frame.shape[0]
                subtitle_region = frame[int(height*0.7):, :]
                
                # OCR识别
                result = ocr.ocr(subtitle_region, cls=True)
                
                # 提取文字
                if result:
                    for line in result:
                        text = line[1][0]  # 获取识别的文字
                        confidence = line[1][1]  # 获取置信度
                        if confidence > 0.9 and text not in previous_text:  # 只输出高置信度且非重复的结果
                            timestamp = frame_count/fps
                            f.write(f"[{timestamp:.1f}秒] {text}\n")
                            print(f"[{timestamp:.1f}秒] {text}")
                            previous_text.add(text)
            
            frame_count += 1
    
    cap.release()
    print(f"字幕已保存到: {output_file}\n")
    return True

def batch_process_videos(input_dir, output_dir='output'):
    """批量处理指定目录下的所有视频文件"""
    # 支持的视频格式
    video_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.flv')
    
    # 获取所有视频文件
    video_files = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(video_extensions)]
    
    if not video_files:
        print(f"在 {input_dir} 中没有找到视频文件")
        return
    
    print(f"找到 {len(video_files)} 个视频文件")
    
    # 处理每个视频
    for i, video_file in enumerate(video_files, 1):
        print(f"\n处理第 {i}/{len(video_files)} 个视频")
        video_path = os.path.join(input_dir, video_file)
        extract_subtitles(video_path, output_dir)
        
    print("\n所有视频处理完成！")

if __name__ == "__main__":
    print("请选择操作模式：")
    print("1. 处理单个视频")
    print("2. 批量处理文件夹中的视频")
    
    choice = input("请输入选择（1或2）: ")
    
    if choice == '1':
        video_path = input("请输入视频文件路径: ")
        extract_subtitles(video_path)
    elif choice == '2':
        input_dir = input("请输入视频文件夹路径: ")
        batch_process_videos(input_dir)
    else:
        print("无效的选择")
