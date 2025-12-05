# VLM SubTask Labeling Tool

我们创建了一个基于 Streamlit 的视频标注工具（v0.2版本），用于处理视频子任务标注，为后续VLM的训练做准备。
标注工具提供以下功能：
- 🎞️ 实时视频预览，支持进度条拖拽同步
- 📝 编辑子任务的 step 信息（描述、start_frame、end_frame）
- 💾 自动保存标注结果到 JSON 文件
- ✅ 已标注/未标注视频分类管理


持续跟新中.....

## 安装

```bash
conda create -n Labeling python=3.10 tensorflow tqdm numpy opencv ffmpeg  -c conda-forge
conda activate Labeling
conda install streamlit
```

## 文件说明

- [dld.py](./dld.py): 从 Hugging Face 下载 Galaxea 开放世界数据集的脚本，使用了阿里镜像加速下载过程。

- [trans2video.py](./trans2video.py): 将 TFRecord 格式的数据集转换为 MP4 视频文件的脚本，从 RLDS 格式的 shard 文件中提取三路摄像头（头部、左手腕、右手腕）的图像序列并生成对应的视频文件。

- [rlds_data_parsing.py](./rlds_data_parsing.py): 解析 RLDS 格式 TFRecord 数据的示例脚本，演示如何定义特征结构并读取 TFRecord 文件中的机器人操作数据。

- [preprocess_results.py](./preprocess_results.py): 对标注结果进行预处理的脚本，负责清理字符串化的字段（如 result）、根据 img_id_list 转换帧索引为实际图像ID等操作。

- [src/app.py](./src/app.py): 基于 Streamlit 的视频标注工具主程序，提供图形界面供用户观看视频并进行子任务标注。



## 预处理
### 1. 解析生成视频

从 TFRecord 数据集生成 MP4 视频文件。该脚本会从 RLDS 格式的 shard 文件中提取三路摄像头的图像序列，并转换为 MP4 视频。

**使用方式：**
```bash
# 编辑 trans2video.py 中的以下参数：
# - shard_path: TFRecord shard 文件路径
# - output_dir: 输出目录（默认为 ./episode_videos）

python trans2video.py  --shard_path /path/to/tfrecord/file 
```

**输出结构：**
```
episode_videos/
├── head/                    # 头部摄像头视频
│   ├── part1_r1_lite_ep0.mp4
│   ├── part1_r1_lite_ep1.mp4
│   └── ...
├── wrist_left/              # 左腕摄像头视频
│   ├── part1_r1_lite_ep0.mp4
│   └── ...
└── wrist_right/             # 右腕摄像头视频
    ├── part1_r1_lite_ep0.mp4
    └── ...
```

**参数说明：**
- `shard_path`: RLDS 数据集的 TFRecord 文件路径（`merged_dataset_large_r1_lite-train.tfrecord-*`）
- `output_dir`: 输出目录，默认为 `./episode_videos`，会自动创建三个子目录：`head`、`wrist_left`、`wrist_right`

### 2. 处理元数据

清理和预处理子任务标注的 JSON 元数据文件。该脚本会：
- 解析字符串化的 JSON 字段
- 转换帧索引为实际帧号
- 移除临时字段，生成干净的数据文件

**使用方式：**
```bash
python preprocess_results.py \
  --input galaxea_subtask_label/part1_r1_lite/results.json \
  --output galaxea_subtask_label/part1_r1_lite/results_cleaned.json
```

**参数说明：**
- `--input`: 输入的原始 JSON 文件路径（默认：`galaxea_subtask_label/part1_r1_lite/results.json`）
- `--output`: 输出的清理后 JSON 文件路径（默认：`galaxea_subtask_label/part1_r1_lite/results_cleaned.json`）

## 运行标注工具

```bash
cd ./src && streamlit run app.py
```

## 注意事项
1. 项目基于**streamlit** 没有高级的文件锁逻辑，请勿同时打开多个文件标注。
2. 标注时候请遵循 两个互斥逻辑（违反会有提示）：
    - 每个子任务的 start_frame < end_frame ，且第一个子任务 end_frame < 第二个子任务的 start_frame
    - 第一个step start_frame>=0，最后一个step end_frame<=视频总帧数。
3. 未标注和已经标注下拉框 里面必须有一个为 "--选择--"，不要同时处理。

