import cv2

def get_mp4_frame_count_opencv(video_path):
    """
    使用 OpenCV 获取 MP4 视频的总帧数
    :param video_path: MP4 文件路径（绝对路径或相对路径）
    :return: 总帧数（int），失败返回 -1
    """
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        return -1
    
    # 获取总帧数（CV_CAP_PROP_FRAME_COUNT = 7）
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 释放资源（必须关闭，避免占用内存）
    cap.release()
    
    return total_frames

# ------------------- 测试 -------------------
if __name__ == "__main__":
    video_path = "/home/unitree/桌面/label_task/episode_videos/wrist_left/part1_r1_lite_ep0.mp4"  # 替换为你的 MP4 文件路径
    frame_count = get_mp4_frame_count_opencv(video_path)
    if frame_count != -1:
        print(f"MP4 总帧数：{frame_count}")
