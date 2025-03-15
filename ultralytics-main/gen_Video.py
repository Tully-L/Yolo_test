import os
import cv2
import random
import numpy as np
from tqdm import tqdm

def create_natural_scrolling_video(dataset_path, output_video_path, max_images=100, resolution=(1280, 720), speed_factor=1.0):
    """
    创建自然的从左往右滚动显示图片的视频，限制最多使用100张图片
    
    参数:
        dataset_path: PIDay数据集的路径
        output_video_path: 输出视频的路径
        max_images: 最多使用的图片数量
        resolution: 视频的分辨率
        speed_factor: 速度因子，1.0表示正常速度
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_video_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取所有图像文件
    image_files = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
    
    # 排序图像文件以确保连续性
    image_files.sort()
    
    total_images = len(image_files)
    print(f"找到 {total_images} 张照片")
    
    # 如果图片超过限制数量，随机选择指定数量的图片
    if total_images > max_images:
        print(f"随机选择 {max_images} 张照片进行处理")
        image_files = random.sample(image_files, max_images)
        # 重新排序选择的图片
        image_files.sort()
    
    # 设置帧率
    fps = 30  # 降低帧率以减少处理负担
    
    # 计算一分钟视频所需的总帧数
    total_frames_needed = fps * 60  # 一分钟视频的总帧数
    
    # 设置滚动速度（像素/帧）
    scroll_speed = 2  # 每帧移动的像素数，可以调整以改变滚动速度
    
    # 预加载所有图像并调整大小
    print("预加载图像...")
    processed_images = []
    for image_path in tqdm(image_files):
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"无法读取图像: {image_path}")
                continue
            
            # 调整图像大小，保持原始宽高比
            h, w = img.shape[:2]
            scale = resolution[1] / h
            new_w = int(w * scale)
            img_resized = cv2.resize(img, (new_w, resolution[1]))
            
            processed_images.append(img_resized)
        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {e}")
    
    # 如果图像不足，复制已有图像
    if len(processed_images) < 10:  # 确保至少有10张图片用于滚动
        processed_images = processed_images * (10 // len(processed_images) + 1)
    
    print(f"实际处理了 {len(processed_images)} 张照片")
    
    # 计算所有图像的总宽度
    total_width = sum(img.shape[1] for img in processed_images)
    
    # 创建一个足够长的画布来容纳所有图像
    long_canvas_width = total_width + resolution[0]
    long_canvas = np.zeros((resolution[1], long_canvas_width, 3), dtype=np.uint8)
    
    # 将所有图像放置在长画布上
    current_x = 0
    for img in processed_images:
        h, w = img.shape[:2]
        # 确保不会超出画布边界
        if current_x + w <= long_canvas_width:
            long_canvas[:, current_x:current_x+w] = img
            current_x += w
    
    # 计算需要多少帧来完成整个滚动
    total_scroll_frames = (long_canvas_width - resolution[0]) // scroll_speed
    
    # 调整滚动速度以适应一分钟的视频长度
    if total_scroll_frames < total_frames_needed:
        # 减慢滚动速度以填满一分钟
        new_scroll_speed = max(1, (long_canvas_width - resolution[0]) // total_frames_needed)
        if new_scroll_speed < scroll_speed:
            scroll_speed = new_scroll_speed
            total_scroll_frames = (long_canvas_width - resolution[0]) // scroll_speed
            print(f"减慢滚动速度至 {scroll_speed} 像素/帧以填满一分钟")
        else:
            # 如果减慢速度仍然不足，则需要重复滚动
            repeat_times = total_frames_needed // total_scroll_frames + 1
            print(f"滚动将重复 {repeat_times} 次以达到一分钟")
    else:
        repeat_times = 1
        # 如果帧数过多，增加滚动速度
        if total_scroll_frames > total_frames_needed:
            scroll_speed = (long_canvas_width - resolution[0]) // total_frames_needed + 1
            print(f"增加滚动速度至 {scroll_speed} 像素/帧以适应一分钟")
            total_scroll_frames = (long_canvas_width - resolution[0]) // scroll_speed
    
    # 创建视频写入器 - 移到这里以确保所有参数都已设置好
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, resolution)
        
        if not video_writer.isOpened():
            raise Exception("无法创建视频写入器")
    except Exception as e:
        print(f"创建视频写入器时出错: {e}")
        return
    
    # 创建滚动效果
    print("正在创建自然滚动效果视频...")
    frame_count = 0
    
    try:
        for _ in range(repeat_times):
            current_pos = 0
            while current_pos + resolution[0] <= long_canvas_width and frame_count < total_frames_needed:
                # 从长画布中截取当前视图
                frame = long_canvas[:, current_pos:current_pos + resolution[0]].copy()
                
                # 确保帧的尺寸正确
                if frame.shape[1] != resolution[0] or frame.shape[0] != resolution[1]:
                    # 调整帧大小以匹配分辨率
                    frame = cv2.resize(frame, resolution)
                
                # 确保帧是BGR格式，而不是BGRA
                if frame.shape[2] > 3:
                    frame = frame[:, :, :3]
                
                # 写入视频帧
                video_writer.write(frame)
                frame_count += 1
                
                # 移动位置
                current_pos += scroll_speed
                
                # 显示进度
                if frame_count % 100 == 0:
                    print(f"已生成 {frame_count}/{total_frames_needed} 帧")
        
        # 如果还不足一分钟，添加静态帧
        if frame_count < total_frames_needed:
            remaining_frames = total_frames_needed - frame_count
            print(f"需要额外添加 {remaining_frames} 帧以达到一分钟")
            
            # 使用最后一帧作为静态帧
            last_pos = min(long_canvas_width - resolution[0], current_pos)
            last_frame = long_canvas[:, last_pos:last_pos + resolution[0]].copy()
            
            # 确保帧的尺寸正确
            if last_frame.shape[1] != resolution[0] or last_frame.shape[0] != resolution[1]:
                last_frame = cv2.resize(last_frame, resolution)
            
            # 确保帧是BGR格式
            if last_frame.shape[2] > 3:
                last_frame = last_frame[:, :, :3]
            
            for _ in range(remaining_frames):
                video_writer.write(last_frame)
    
    except Exception as e:
        print(f"生成视频帧时出错: {e}")
    finally:
        # 确保视频写入器被正确释放
        video_writer.release()
        print(f"视频已保存到: {output_video_path}")

if __name__ == "__main__":
    # 设置参数
    dataset_path = r"E:\Yolo\Pidray\test"  # PIDay数据集路径
    output_dir = r"E:\Yolo\Pidray\test_Video"  # 输出目录
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 设置输出视频路径
    output_video_path = os.path.join(output_dir, "test_Video_natural_scroll_100imgs.mp4")
    
    # 设置视频分辨率
    resolution = (1280, 720)  # 720p分辨率
    
    # 生成自然滚动效果的视频，限制使用100张图片
    create_natural_scrolling_video(dataset_path, output_video_path, max_images=100, resolution=resolution, speed_factor=1.0)
    
    print("视频生成完成！")
    print(f"视频保存在: {output_video_path}")
    print("您可以使用UI.py中的'��视频检测'功能加载此视频进行检测")
