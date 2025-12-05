import tensorflow as tf

# 1) 你的 shard 路径
shard_path = "/home/unitree/桌面/label_task/galaxea_part1/rlds/part1_r1_lite/1.0.0/merged_dataset_large_r1_lite-train.tfrecord-00000-of-02048"

# 2) 定义 feature 结构：把 features.json 里的所有字段都列出来
feature_description = {
    # episode-level
    "episode_metadata/file_path": tf.io.FixedLenFeature([], tf.string),

    # step-level flags / meta
    "steps/is_first": tf.io.VarLenFeature(tf.int64),
    "steps/is_last": tf.io.VarLenFeature(tf.int64),
    "steps/segment_idx": tf.io.VarLenFeature(tf.int64),
    "steps/variant_idx": tf.io.VarLenFeature(tf.int64),
    "steps/language_instruction": tf.io.VarLenFeature(tf.string),

    # observation: low-dim
    "steps/observation/base_velocity": tf.io.VarLenFeature(tf.float32),
    "steps/observation/joint_position_arm_left": tf.io.VarLenFeature(tf.float32),
    "steps/observation/joint_position_arm_right": tf.io.VarLenFeature(tf.float32),
    "steps/observation/joint_velocity_arm_left": tf.io.VarLenFeature(tf.float32),
    "steps/observation/joint_velocity_arm_right": tf.io.VarLenFeature(tf.float32),
    "steps/observation/joint_position_torso": tf.io.VarLenFeature(tf.float32),
    "steps/observation/gripper_state_left": tf.io.VarLenFeature(tf.float32),
    "steps/observation/gripper_state_right": tf.io.VarLenFeature(tf.float32),
    "steps/observation/last_action": tf.io.VarLenFeature(tf.float32),

    # observation: images (encoded bytes)
    "steps/observation/image_camera_head": tf.io.VarLenFeature(tf.string),
    "steps/observation/image_camera_wrist_left": tf.io.VarLenFeature(tf.string),
    "steps/observation/image_camera_wrist_right": tf.io.VarLenFeature(tf.string),
    "steps/observation/depth_camera_head": tf.io.VarLenFeature(tf.string),
    "steps/observation/depth_camera_wrist_left": tf.io.VarLenFeature(tf.string),
    "steps/observation/depth_camera_wrist_right": tf.io.VarLenFeature(tf.string),

    # action
    "steps/action": tf.io.VarLenFeature(tf.float32),
}


def parse_episode(raw):
    ex = tf.io.parse_single_example(raw, feature_description)
    episode = {}

    # -------- episode-level --------
    episode["file_path"] = ex["episode_metadata/file_path"].numpy().decode("utf-8")

    # -------- step-level meta --------
    lang = tf.sparse.to_dense(ex["steps/language_instruction"]).numpy()
    T = len(lang)  # 时间长度

    episode["T"] = T
    episode["language_instruction"] = [s.decode("utf-8") for s in lang]

    episode["is_first"] = tf.sparse.to_dense(ex["steps/is_first"]).numpy().astype(bool)
    episode["is_last"] = tf.sparse.to_dense(ex["steps/is_last"]).numpy().astype(bool)
    episode["segment_idx"] = tf.sparse.to_dense(ex["steps/segment_idx"]).numpy().astype(int)
    episode["variant_idx"] = tf.sparse.to_dense(ex["steps/variant_idx"]).numpy().astype(int)

    # -------- observation: low-dim --------
    def _reshape(name, dim):
        arr = tf.sparse.to_dense(ex[name]).numpy()
        return arr.reshape(T, dim)

    obs = {}
    obs["base_velocity"] = _reshape("steps/observation/base_velocity", 3)
    obs["joint_position_arm_left"] = _reshape("steps/observation/joint_position_arm_left", 6)
    obs["joint_position_arm_right"] = _reshape("steps/observation/joint_position_arm_right", 6)
    obs["joint_velocity_arm_left"] = _reshape("steps/observation/joint_velocity_arm_left", 6)
    obs["joint_velocity_arm_right"] = _reshape("steps/observation/joint_velocity_arm_right", 6)
    obs["joint_position_torso"] = _reshape("steps/observation/joint_position_torso", 4)
    obs["gripper_state_left"] = _reshape("steps/observation/gripper_state_left", 1)
    obs["gripper_state_right"] = _reshape("steps/observation/gripper_state_right", 1)
    obs["last_action"] = _reshape("steps/observation/last_action", 26)

    # -------- observation: images --------
    def _decode_rgb(varname):
        bytes_arr = tf.sparse.to_dense(ex[varname]).numpy()
        imgs = []
        for b in bytes_arr:
            img = tf.io.decode_jpeg(b)  # (224, 224, 3) uint8
            imgs.append(img.numpy())
        return imgs

    def _decode_depth(varname):
        bytes_arr = tf.sparse.to_dense(ex[varname]).numpy()
        imgs = []
        for b in bytes_arr:
            # 深度一般是 PNG，dtype=uint16
            img = tf.io.decode_png(b, dtype=tf.uint16)
            imgs.append(img.numpy())
        return imgs

    obs["image_camera_head"] = _decode_rgb("steps/observation/image_camera_head")
    obs["image_camera_wrist_left"] = _decode_rgb("steps/observation/image_camera_wrist_left")
    obs["image_camera_wrist_right"] = _decode_rgb("steps/observation/image_camera_wrist_right")
    obs["depth_camera_head"] = _decode_depth("steps/observation/depth_camera_head")
    obs["depth_camera_wrist_left"] = _decode_depth("steps/observation/depth_camera_wrist_left")
    obs["depth_camera_wrist_right"] = _decode_depth("steps/observation/depth_camera_wrist_right")

    episode["observation"] = obs

    # -------- action --------
    actions_flat = tf.sparse.to_dense(ex["steps/action"]).numpy()
    episode["actions"] = actions_flat.reshape(T, 26)

    return episode


# 4) 读取 shard 的第一个 episode
raw_ds = tf.data.TFRecordDataset(shard_path)
raw_first_episode = next(iter(raw_ds))
episode = parse_episode(raw_first_episode)

print("====== 解析成功 ======")
print("文件来源:", episode["file_path"])
print("Episode 长度(steps):", episode["T"])
print("第一帧指令:", episode["language_instruction"][0])
print("第一帧动作 (26维):", episode["actions"][0])
print("第一帧头部图像 shape:", episode["observation"]["image_camera_head"][0].shape)

# 统计每种相机的帧数
print("\n=== 每种相机的帧数（这个 episode）===")
print("RGB  head           :", len(episode["observation"]["image_camera_head"]))
print("RGB  wrist_left     :", len(episode["observation"]["image_camera_wrist_left"]))
print("RGB  wrist_right    :", len(episode["observation"]["image_camera_wrist_right"]))
print("Depth head          :", len(episode["observation"]["depth_camera_head"]))
print("Depth wrist_left    :", len(episode["observation"]["depth_camera_wrist_left"]))
print("Depth wrist_right   :", len(episode["observation"]["depth_camera_wrist_right"]))

print("\n=== 图像尺寸一致性检查（这个 episode）===")

rgb_keys = [
    "image_camera_head",
    "image_camera_wrist_left",
    "image_camera_wrist_right",
]

for key in rgb_keys:
    frames = episode["observation"][key]

    # 统计所有帧的 shape
    shapes = [f.shape for f in frames]

    # 用 set 去重
    unique_shapes = set(shapes)

    print(f"{key}:")
    print("  帧数:", len(frames))
    print("  不同的 shape 数量:", len(unique_shapes))
    print("  shape 列表:", unique_shapes)