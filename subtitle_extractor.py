import sys
import subprocess

# 在导入部分添加自动安装依赖的代码
try:
    import cv2
    import numpy as np
    from paddleocr import PaddleOCR
    import time
    import os
    from tqdm import tqdm
    from multiprocessing import Pool, cpu_count
    import multiprocessing
except Exception as e:
    print(f"导入库时出错: {str(e)}")
    exit(1)

print("所有库导入成功")

# 全局变量用于框选
drawing = False
ix, iy = -1, -1
selection = None
last_frame = None

def draw_rectangle(event, x, y, flags, param):
    """
    鼠标框选事件处理函数
    """
    global drawing, ix, iy, selection, last_frame
    window_name = param['window_name']
    
    # 创建静态显示缓冲
    if not hasattr(draw_rectangle, 'display_buffer'):
        draw_rectangle.display_buffer = param['frame'].copy()
        draw_rectangle.base_frame = param['frame'].copy()
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        selection = None
        # 保存原始帧
        draw_rectangle.base_frame = param['frame'].copy()
        # 重置显示缓冲
        draw_rectangle.display_buffer = draw_rectangle.base_frame.copy()
        
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        # 使用基础帧
        draw_rectangle.display_buffer = draw_rectangle.base_frame.copy()
        
        # 绘制矩形和文本
        cv2.rectangle(draw_rectangle.display_buffer, (ix, iy), (x, y), (0, 255, 0), 2)
        
        # 计算比例
        height = draw_rectangle.base_frame.shape[0]
        current_bottom = min(iy, y) / height
        current_top = max(iy, y) / height
        
        # 添加文本（带背景）
        text = f'区域: {current_bottom:.3f} - {current_top:.3f}'
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(draw_rectangle.display_buffer, (10, 5), 
                     (10 + text_w, 35), (0, 0, 0), -1)
        cv2.putText(draw_rectangle.display_buffer, text, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 使用 imshow 显示
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(window_name, draw_rectangle.display_buffer)
        
    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            drawing = False
            if abs(x - ix) > 10 and abs(y - iy) > 10:  # 确保选择区域足够大
                x1, y1 = min(ix, x), min(iy, y)
                x2, y2 = max(ix, x), max(iy, y)
                selection = (x1, y1, x2, y2)
                
                # 最终显示缓冲上绘制确认框
                final_frame = draw_rectangle.base_frame.copy()
                cv2.rectangle(final_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                height = final_frame.shape[0]
                bottom_ratio = y1 / height
                top_ratio = y2 / height
                
                # 添加确认文本（带背景）
                text = f'已选区域: {bottom_ratio:.3f} - {top_ratio:.3f}'
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(final_frame, (10, 5), 
                            (10 + text_w, 35), (0, 0, 0), -1)
                cv2.putText(final_frame, text, (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                cv2.imshow(window_name, final_frame)

def select_subtitle_area(video_path):
    """
    使用OpenCV实现的框选功能，可以播放视频并在暂停时框选
    """
    global selection  # 添加这行，声明使用全局变量
    selection = None  # 重置选择
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"错误：无法打开视频文件: {video_path}")
            return None
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 创建窗口
        window_name = '视频播放 (空格键暂停后框选，ESC重选，ENTER确认)'
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        paused = False
        frame = None
        
        print("\n播放控制：")
        print("空格键 - 暂停/继续")
        print("→ - 快进5秒")
        print("← - 快退5秒")
        print("暂停后可以框选字幕区域")
        print("ESC - 重新选择")
        print("ENTER - 确认选择")
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 循环播放
                    continue
                
                # 调整帧大小以便显示
                max_height = 720
                if frame.shape[0] > max_height:
                    scale = max_height / frame.shape[0]
                    new_width = int(frame.shape[1] * scale)
                    frame = cv2.resize(frame, (new_width, max_height))
                
                # 显示当前时间
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                current_time = current_frame / fps
                time_str = time.strftime('%H:%M:%S', time.gmtime(current_time))
                cv2.putText(frame, time_str, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow(window_name, frame)
            
            # 处理按键
            key = cv2.waitKey(25) & 0xFF
            
            if key == 27:  # ESC
                selection = None  # 重置选择
                if frame is not None:
                    cv2.imshow(window_name, frame.copy())
            elif key == 32:  # 空格键
                paused = not paused
                if paused and frame is not None:
                    # 设置鼠标回调
                    param = {'frame': frame.copy(), 'window_name': window_name}
                    cv2.setMouseCallback(window_name, draw_rectangle, param)
            elif key == 83 and not paused:  # →
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(total_frames - 1, current_frame + int(fps * 5)))
            elif key == 81 and not paused:  # ←
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_frame - int(fps * 5)))
            elif key == 13 and selection and paused:  # Enter
                x1, y1, x2, y2 = selection
                height = frame.shape[0]
                bottom_ratio = y1 / height
                top_ratio = y2 / height
                cap.release()
                cv2.destroyAllWindows()
                return (bottom_ratio, top_ratio)
            elif key == ord('q'):  # Q
                break
            
        cap.release()
        cv2.destroyAllWindows()
        return None
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        return None

def extract_subtitles(video_path, output_file='subtitles.srt', lang='ch', subtitle_area=(0.8, 0.9)):
    """
    从视频中提取字幕并保存到文本文件
    :param video_path: 视频文件路径
    :param output_file: 输出文本文件路径
    :param lang: 识别语言，支持 ch(中文)、en(英文)、japan(日语)
    :param subtitle_area: 字幕区域位置元组 (bottom_ratio, top_ratio)，范围0-1
    """
    bottom_ratio, top_ratio = subtitle_area
    
    # Validate subtitle area ratios
    if not (0 <= bottom_ratio <= 1 and 0 <= top_ratio <= 1):
        print("错误：字幕区域比例必须在0-1之间")
        return
    if bottom_ratio >= top_ratio:
        print("错误：底部比例必须小于顶部比例")
        return
    
    try:
        # 使用绝对路径
        det_model_dir = "C:/subtitle/models/det"  # 改为你的实际路径
        rec_model_dir = "C:/subtitle/models/rec"
        cls_model_dir = "C:/subtitle/models/cls"
        
        print(f"检测模型路径: {det_model_dir}")
        print(f"识别模型路径: {rec_model_dir}")
        print(f"分类模型路径: {cls_model_dir}")
        
        # Check if model directories exist
        model_paths = [det_model_dir, rec_model_dir, cls_model_dir]
        for path in model_paths:
            if not os.path.exists(path):
                print(f"错误：模型路径不存在: {path}")
                return
                
        # 初始化 OCR
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            det_model_dir=det_model_dir,
            rec_model_dir=rec_model_dir,
            cls_model_dir=cls_model_dir,
            show_log=True,  # 显示日志以便调试
            download_font=False,
            max_batch_size=7  # 添加这个参数
        )
    except Exception as e:
        print(f"始化 OCR 失败: {str(e)}")
        return
    
    if not os.path.exists(video_path):
        print(f"错误：视频文件不存在: {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("错误：无法打开视频文件")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 修改字幕持续时的阈
    MIN_DURATION = 0.5  # 最小持续0.5秒
    MAX_DURATION = 3.0  # 最大持续3秒
    EMPTY_FRAMES_THRESHOLD = 6  # 修改为6帧，连续6帧无字幕才认为字幕消失
    
    subtitles = []
    last_text = ""
    frame_count = 0
    start_time = None
    empty_frames = 0
    subtitle_index = 1
    
    print(f"开始处理视频 - FPS: {fps:.2f}")
    
    with tqdm(total=total_frames, desc="处理进度") as pbar:
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                pbar.update(1)
                
                # 修改为每秒处理10帧
                if frame_count % int(fps/10) != 0:  # 将5改为10
                    continue
                    
                # 获取视频底部区域
                height = frame.shape[0]
                width = frame.shape[1]

                # 修改字幕区域的截取方式
                # 只截取底部 20% 的区域，个比例可以根实际视频调整
                bottom_margin = int(height * bottom_ratio)  # 从底部 80% 处开始
                top_margin = int(height * top_ratio)     # 到底部 90% 结束

                # 获取更精确的字幕区域
                subtitle_region = frame[bottom_margin:top_margin, :]

                # 可以选择性地只取中间部分的宽度，避免边缘干扰
                # left_margin = int(width * 0.1)    # 左边留出 10%
                # right_margin = int(width * 0.9)   # 右边留出 10%
                # subtitle_region = frame[bottom_margin:top_margin, left_margin:right_margin]
                
                try:
                    # OCR识别
                    result = ocr.ocr(subtitle_region, cls=True)
                    
                    if result:
                        # 提高置信度要求，并添加文本长度限制
                        text_items = []
                        for line in result:
                            for item in line:
                                if item[1][1] > 0.9:  # 提高置信度阈值
                                    text = item[1][0].strip()
                                    if len(text) >= 2:  # 只保留长度大于等于2的文本
                                        text_items.append(text)
                        
                        text = " ".join(text_items)
                        
                        # 简单的文本验证
                        if text.strip() and len(text) <= 50:  # 限制最大长度，避免误识别
                            empty_frames = 0
                            current_time = frame_count/fps
                            
                            if text != last_text:
                                if start_time is not None:
                                    duration = current_time - start_time
                                    
                                    # 确保字幕持续时间在合理范围内
                                    if duration >= MIN_DURATION:
                                        if duration > MAX_DURATION:
                                            end_time = start_time + MAX_DURATION
                                        else:
                                            end_time = current_time
                                            
                                        start_str = time.strftime('%H:%M:%S,', time.gmtime(start_time)) + f'{int((start_time % 1) * 1000):03d}'
                                        end_str = time.strftime('%H:%M:%S,', time.gmtime(end_time)) + f'{int((end_time % 1) * 1000):03d}'
                                        
                                        # 直接使用原文本，不添加拼音
                                        subtitle_entry = f"{subtitle_index}\n{start_str} --> {end_str}\n{last_text}\n"
                                        subtitles.append(subtitle_entry)
                                        subtitle_index += 1
                                
                                start_time = current_time
                                last_text = text
                        else:
                            empty_frames += 1
                            # 果连续多帧没有检测到字幕，结束当前字幕
                            if empty_frames >= EMPTY_FRAMES_THRESHOLD and start_time is not None:
                                current_time = frame_count/fps
                                duration = current_time - start_time
                                
                                if duration >= MIN_DURATION:
                                    start_str = time.strftime('%H:%M:%S,', time.gmtime(start_time)) + f'{int((start_time % 1) * 1000):03d}'
                                    end_str = time.strftime('%H:%M:%S,', time.gmtime(current_time)) + f'{int((current_time % 1) * 1000):03d}'
                                    subtitle_entry = f"{subtitle_index}\n{start_str} --> {end_str}\n{last_text}\n"
                                    subtitles.append(subtitle_entry)
                                    subtitle_index += 1
                                    
                                start_time = None
                                last_text = ""
                
                except Exception as e:
                    continue
        
            # 处理最后一帧字幕
            if start_time is not None and last_text:
                end_time = frame_count/fps
                start_str = time.strftime('%H:%M:%S,', time.gmtime(start_time)) + f'{int((start_time % 1) * 1000):03d}'
                end_str = time.strftime('%H:%M:%S,', time.gmtime(end_time)) + f'{int((end_time % 1) * 1000):03d}'
                subtitle_entry = f"{subtitle_index}\n{start_str} --> {end_str}\n{last_text}\n"
                subtitles.append(subtitle_entry)
    
        except Exception as e:
            print(f"\n处理视频时出错: {str(e)}")
        finally:
            cap.release()
    
    if subtitles:
        try:
            # 从第一条字幕中提取时间信息
            first_subtitle = subtitles[0]
            time_line = first_subtitle.split('\n')[1]
            start_time = time_line.split('-->')[0].strip()
            seconds = start_time.split(':')[2]
            time_str = seconds.replace(',', '')
            
            # 获取视频所在的目录
            video_dir = os.path.dirname(video_path)
            if not video_dir:
                video_dir = os.getcwd()
                
            # 在视频所在目录创建output文件夹
            output_dir = os.path.join(video_dir, 'output')
            os.makedirs(output_dir, exist_ok=True)
            
            # 构建基本文件名
            file_name, file_ext = os.path.splitext(os.path.basename(video_path))
            base_name = f"{file_name}_{time_str}"
            
            # 检查文件是否存在并生成唯一文件名
            counter = 1
            output_file = f"{base_name}.srt"
            save_path = os.path.join(output_dir, output_file)
            
            while os.path.exists(save_path):
                output_file = f"{base_name}_{counter}.srt"
                save_path = os.path.join(output_dir, output_file)
                counter += 1
            
            # 保存文件
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(subtitles))
            print(f"\n字幕提取完成，共提取 {len(subtitles)} 条字幕")
            print(f"第一条字幕时间点: {start_time}")
            print(f"已保存到: {save_path}")
            
        except Exception as e:
            print(f"\n保存字幕文件时出错: {str(e)}")
            # 如果保存失败，尝试保存到当前目录
            try:
                # 在当前目录也要避免覆盖
                base_name = f"{os.path.splitext(output_file)[0]}_{time_str}"
                counter = 1
                save_file = f"{base_name}.srt"
                save_path = os.path.join(os.getcwd(), save_file)
                
                while os.path.exists(save_path):
                    save_file = f"{base_name}_{counter}.srt"
                    save_path = os.path.join(os.getcwd(), save_file)
                    counter += 1
                
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(subtitles))
                print(f"已保存到当前目录: {save_path}")
            except Exception as e2:
                print(f"保存到当前目录也失败: {str(e2)}")
    else:
        print("\n未能提取到任何字幕")

def preview_subtitle_area(video_path, bottom_ratio, top_ratio):
    """
    预览字幕区域的选择效果
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("错误：无法打开视频文件")
        return False

    # 读取视频中间的一帧
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("错误：无法读取视频帧")
        return False

    height = frame.shape[0]
    
    # 在图像上画出字幕区域
    preview_frame = frame.copy()
    bottom_margin = int(height * bottom_ratio)
    top_margin = int(height * top_ratio)
    
    # 在选定区域下方画红线
    cv2.line(preview_frame, (0, bottom_margin), (frame.shape[1], bottom_margin), (0, 0, 255), 2)
    cv2.line(preview_frame, (0, top_margin), (frame.shape[1], top_margin), (0, 0, 255), 2)
    
    # 显示预览图像
    cv2.imshow('字幕区域预览 (按ESC退出，按ENTER确认)', preview_frame)
    
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC键
            cv2.destroyAllWindows()
            return False
        elif key == 13:  # ENTER键
            cv2.destroyAllWindows()
            return True

def process_single_video(args):
    """
    处理单个视频的函数
    """
    try:
        video_path, lang, subtitle_area = args
        output_file = os.path.splitext(os.path.basename(video_path))[0] + ".srt"
        extract_subtitles(video_path, output_file, lang, subtitle_area)
        return True
    except Exception as e:
        print(f"处理视频时出错: {str(e)}")
        return False

def process_videos_in_groups(video_files, subtitle_areas, lang, group_size=5):
    """
    每组5个视频并行处理，一组完成后自动处理下一组
    """
    try:
        # 准备所有需要处理的视频参数
        process_args = []
        for video_path in video_files:
            if video_path in subtitle_areas:
                process_args.append((video_path, lang, subtitle_areas[video_path]))
        
        total_videos = len(process_args)
        if total_videos == 0:
            print("没有可处理的视频")
            return
        
        # 将视频分组
        video_groups = [process_args[i:i + group_size] 
                       for i in range(0, len(process_args), group_size)]
        
        print(f"\n总共 {total_videos} 个视频，分成 {len(video_groups)} 组处理")
        print(f"每组同时处理 {group_size} 个视频（最后一组可能少于{group_size}个）")
        
        # 处理每一组视频
        total_processed = 0
        for group_idx, group in enumerate(video_groups, 1):
            print(f"\n开始处理第 {group_idx}/{len(video_groups)} 组:")
            for args in group:
                print(f"- {os.path.basename(args[0])}")
            
            try:
                # 使用进程池处理当前组的视频
                with Pool(processes=min(len(group), cpu_count())) as pool:
                    # 使用 map 同步处理视频
                    results = pool.map(process_single_video, group)
                    
                    # 更新处理进度
                    successful = sum(1 for r in results if r)
                    total_processed += len(group)
                    
                    print(f"\n第 {group_idx} 组处理完成！")
                    print(f"成功: {successful}/{len(group)}")
                    print(f"总进度: {total_processed}/{total_videos}")
                
                # 自动继续处理下一组
                if group_idx < len(video_groups):
                    print(f"\n自动开始处理第 {group_idx + 1} 组...")
                    
            except Exception as e:
                print(f"处理第 {group_idx} 组时出错: {str(e)}")
                continue
        
        print("\n所有视频处理完成！")
        
    except Exception as e:
        print(f"处理视频时出错: {str(e)}")

if __name__ == "__main__":
    try:
        print("程序开始运行...")
        print("请输入视频文件路径或文件夹路径")
        print("支持以下格式：")
        print("1. 单个视频文件路径，如：E:\\video\\test.mp4")
        print("2. 包含多个视频的文件夹路径，如：E:\\video")
        path = input("请输入路径: ").strip('"')  # 去除可能的引号
        
        # 设置默认值
        lang = "ch"
        video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv']
        
        # 获取要处理的视频文件列表
        video_files = []
        if os.path.isfile(path):
            if os.path.splitext(path)[1].lower() in video_extensions:
                video_files.append(path)
        elif os.path.isdir(path):
            for file in os.listdir(path):
                if os.path.splitext(file)[1].lower() in video_extensions:
                    video_files.append(os.path.join(path, file))
        
        if not video_files:
            print("未找到任何视频文件！")
            exit(1)
            
        print(f"\n找到 {len(video_files)} 个视频文件:")
        for i, video in enumerate(video_files, 1):
            print(f"{i}. {os.path.basename(video)}")
        
        # 存储每个视频的字幕区域
        subtitle_areas = {}
        
        # 先进行所有视频的字幕区域框选
        print("\n开始框选每个视频的字幕区域...")
        for i, video_path in enumerate(video_files, 1):
            print(f"\n请框选第 {i}/{len(video_files)} 个视频的字幕区域: {os.path.basename(video_path)}")
            area = select_subtitle_area(video_path)
            if area:
                subtitle_areas[video_path] = area
                bottom_ratio, top_ratio = area
                print(f"已记录字幕区域：底部 {bottom_ratio:.3f}，顶部 {top_ratio:.3f}")
            else:
                print(f"跳过视频 {os.path.basename(video_path)}")
                continue
        
        if not subtitle_areas:
            print("没有成功框选任何视频的字幕区域，程序退出")
            exit(1)
            
        # 确认开始处理
        print(f"\n已完成 {len(subtitle_areas)} 个视频的字幕区域框选")
        input("按回车键开始处理视频...")
        
        # 使用分组处理函数
        process_videos_in_groups(video_files, subtitle_areas, lang, group_size=5)
        
    except Exception as e:
        print(f"程序出错: {str(e)}") 