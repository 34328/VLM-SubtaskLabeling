
import os
import argparse
import subprocess

import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow_datasets as tfds
# Configure Tensorflow with *no GPU devices* (to prevent clobber with PyTorch)
tf.config.set_visible_devices([], "GPU")

# 添加命令行参数解析
parser = argparse.ArgumentParser(description='Convert TFRecord to videos')
parser.add_argument('--shard_path', type=str, 
                    help='Path to the TFRecord shard file')
parser.add_argument('--output_dir', type=str, default="./episode_videos",
                    help='Output directory for videos (default: ./episode_videos)')
args = parser.parse_args()
# 1) shard 文件路径（通过命令行参数获取）
shard_path = args.shard_path
if not shard_path:
    raise ValueError("必须提供 --shard_path 参数指定 TFRecord 文件路径")

# 例如
# shard_path = "/home/unitree/桌面/label_task/galaxea_part1/rlds/part1_r1_lite/1.0.0/merged_dataset_large_r1_lite-train.tfrecord-00000-of-02048"

# 输出目录
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# 三个相机对应的子目录
cam_dirs = {
    "head": os.path.join(output_dir, "head"),
    "wrist_left": os.path.join(output_dir, "wrist_left"),
    "wrist_right": os.path.join(output_dir, "wrist_right"),
}
for d in cam_dirs.values():
    os.makedirs(d, exist_ok=True)

# 2) 解析 episode（RLDS 格式，返回生成器以节省内存）
def parse_episode(episode):
    """
    解析 RLDS episode，返回迭代器以节省内存
    Args:
        episode: RLDS episode 字典，包含 'steps' dataset
    Returns:
        instruction, step_count, steps_iterator
    """
    # 获取 steps dataset
    steps = episode['steps']
    
    # 获取第一个 step 来提取指令和计数
    first_step = None
    step_count = 0
    for step in steps:
        if first_step is None:
            first_step = step
        step_count += 1
    
    if first_step is None or step_count == 0:
        return None, 0, None
    
    # 提取语言指令
    instr = first_step['language_instruction'].numpy().decode('utf-8')
    
    # 重新创建迭代器（因为上面已经消耗了）
    steps_iter = episode['steps']
    
    return instr, step_count, steps_iter


# 3) 遍历 shard 中的 episodes
builder = tfds.builder_from_directory(os.path.join(args.shard_path, "1.0.0"))
raw_ds = builder.as_dataset(split="train", shuffle_files=False)

print("开始解析并导出视频...")

for ep_idx, episode in tqdm(enumerate(raw_ds), desc="Episodes"):
    instr, T, steps_iter = parse_episode(episode)

    if steps_iter is None or T == 0:
        tqdm.write(f"[WARN] Episode {ep_idx}: 没有有效帧，跳过")
        continue

    # 打印每个 episode 的帧数
    tqdm.write(f"Episode {ep_idx}: {T} 帧, 指令: {instr}")

    # 基础文件名（3 个相机共用同一个前缀）
    base_name = f"part1_r1_lite_ep{ep_idx}"

    # 使用 FFmpeg 通过管道写入视频（H.264 编码）
    ffmpeg_procs = {}
    video_paths = {}
    fps = 30
    
    # 遍历所有帧
    first_step = True
    for step in steps_iter:
        # 提取三个相机的图像
        img_head = step['observation']['image_camera_head'].numpy()
        img_wrist_left = step['observation']['image_camera_wrist_left'].numpy()
        img_wrist_right = step['observation']['image_camera_wrist_right'].numpy()
        
        # 如果是第一帧，初始化 FFmpeg 进程
        if first_step:
            for cam_key, img in [('head', img_head), 
                                  ('wrist_left', img_wrist_left), 
                                  ('wrist_right', img_wrist_right)]:
                h, w, _ = img.shape
                video_path = os.path.join(cam_dirs[cam_key], base_name + ".mp4")
                video_paths[cam_key] = video_path
                
                # 使用 FFmpeg 管道写入，H.264 编码
                ffmpeg_cmd = [
                    'ffmpeg',
                    '-y',  # 覆盖输出文件
                    '-f', 'rawvideo',
                    '-vcodec', 'rawvideo',
                    '-s', f'{w}x{h}',
                    '-pix_fmt', 'rgb24',
                    '-r', str(fps),
                    '-i', '-',  # 从管道读取
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-crf', '23',
                    '-pix_fmt', 'yuv420p',
                    video_path
                ]
                
                proc = subprocess.Popen(
                    ffmpeg_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                ffmpeg_procs[cam_key] = proc
            
            first_step = False
        
        # 写入当前帧到各个视频
        for cam_key, img in [('head', img_head), 
                              ('wrist_left', img_wrist_left), 
                              ('wrist_right', img_wrist_right)]:
            if cam_key in ffmpeg_procs:
                # 直接写入 RGB 数据
                ffmpeg_procs[cam_key].stdin.write(img.astype(np.uint8).tobytes())
        
        # 释放当前帧的图像数据
        del img_head, img_wrist_left, img_wrist_right, step
    
    # 关闭所有 FFmpeg 进程
    for cam_key, proc in ffmpeg_procs.items():
        proc.stdin.close()
        proc.wait()
    
    # 清理
    del ffmpeg_procs, video_paths, steps_iter

print("全部视频生成完毕！输出目录：", output_dir)
