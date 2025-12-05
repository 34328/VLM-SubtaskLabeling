
import os
import cv2
import argparse

import numpy as np
from tqdm import tqdm
import tensorflow as tf


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

# 2) 定义 TFRecord feature 结构（解析三路 RGB + 指令）
feature_description = {
    "steps/observation/image_camera_head": tf.io.VarLenFeature(tf.string),
    "steps/observation/image_camera_wrist_left": tf.io.VarLenFeature(tf.string),
    "steps/observation/image_camera_wrist_right": tf.io.VarLenFeature(tf.string),
    "steps/language_instruction": tf.io.VarLenFeature(tf.string),
}


# 3) 解析 episode（取三路 RGB 头/左右腕）
def parse_episode(raw):
    ex = tf.io.parse_single_example(raw, feature_description)

    # 语言指令（每帧一样）
    lang = tf.sparse.to_dense(ex["steps/language_instruction"]).numpy()
    T = len(lang)

    if T == 0:
        return None, None, 0

    instr = lang[0].decode("utf-8")

    # 小工具：解码一整条序列的 JPEG 图像
    def decode_rgb_sequence(varname):
        bytes_arr = tf.sparse.to_dense(ex[varname]).numpy()
        frames = []
        for b in bytes_arr:
            img = tf.io.decode_jpeg(b)  # (H,W,3) uint8
            frames.append(img.numpy())
        return frames

    frames_head = decode_rgb_sequence("steps/observation/image_camera_head")
    frames_wl = decode_rgb_sequence("steps/observation/image_camera_wrist_left")
    frames_wr = decode_rgb_sequence("steps/observation/image_camera_wrist_right")

    # 安全起见：取三路里最小的长度作为本 episode 的有效帧数
    T_effective = min(len(frames_head), len(frames_wl), len(frames_wr), T)

    frames = {
        "head": frames_head[:T_effective],
        "wrist_left": frames_wl[:T_effective],
        "wrist_right": frames_wr[:T_effective],
    }

    return frames, instr, T_effective


# 4) 遍历 shard 中的 episodes
raw_ds = tf.data.TFRecordDataset(shard_path)

print("开始解析并导出视频...")

for ep_idx, raw in tqdm(enumerate(raw_ds), desc="Episodes"):
    frames, instr, T = parse_episode(raw)

    if frames is None or T == 0:
        tqdm.write(f"[WARN] Episode {ep_idx}: 没有有效帧，跳过")
        continue

    # 打印每个 episode 的帧数
    tqdm.write(f"Episode {ep_idx}: {T} 帧")

    # 基础文件名（3 个相机共用同一个前缀）
    base_name = f"part1_r1_lite_ep{ep_idx}"

    # 对每路相机都生成一个 mp4
    for cam_key, cam_frames in frames.items():
        if len(cam_frames) == 0:
            tqdm.write(f"[WARN] Episode {ep_idx} 相机 {cam_key} 无帧，跳过")
            continue

        # 视频存放路径
        video_path = os.path.join(cam_dirs[cam_key], base_name + ".mp4")

        # OpenCV 视频写入器
        h, w, _ = cam_frames[0].shape
        fps = 30
        fourcc = cv2.VideoWriter_fourcc(*'H264')

        writer = cv2.VideoWriter(
            video_path,
            fourcc,
            fps,
            (w, h),
        )

        for frame in cam_frames:
            # cv2 使用 BGR
            frame_bgr = frame[:, :, ::-1]
            writer.write(frame_bgr)

        writer.release()

print("全部视频生成完毕！输出目录：", output_dir)
