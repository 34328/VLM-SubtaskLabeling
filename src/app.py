import os
import json
import glob
import ast
import copy
import re
import time
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import base64 
import streamlit as st
import streamlit.components.v1 as components 

try:
    import cv2
except ImportError:
    cv2 = None
    st.warning("⚠ 未安装 OpenCV（cv2），无法自动读取视频帧数和 FPS，请先 `pip install opencv-python`。")


# ==========================
# 配置加载函数
# ==========================
def load_config_from_file(config_path: str) -> Optional[Dict[str, Any]]:
    """从配置文件加载配置"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        st.error(f"❌ 无法加载配置文件 {config_path}: {e}")
        return None


def get_config_from_args_or_env():
    """
    从命令行参数或环境变量获取配置
    
    优先级:
    1. 命令行参数 --config <path>
    2. 环境变量 ANNOTATOR_ID
    3. 环境变量 ANNOTATOR_CONFIG
    4. 默认配置
    """
    config = {}
    
    # 1. 尝试从命令行参数读取 --config
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv):
            if arg == '--config' and i + 1 < len(sys.argv):
                config_path = sys.argv[i + 1]
                loaded_config = load_config_from_file(config_path)
                if loaded_config:
                    return loaded_config
    
    # 2. 尝试从环境变量 ANNOTATOR_CONFIG 读取
    annotator_config = os.environ.get('ANNOTATOR_CONFIG')
    if annotator_config and os.path.exists(annotator_config):
        loaded_config = load_config_from_file(annotator_config)
        if loaded_config:
            return loaded_config
    
    # 3. 尝试从环境变量 ANNOTATOR_ID 推导配置
    annotator_id = os.environ.get('ANNOTATOR_ID')
    if annotator_id:
        # 假设工作目录在 multi_annotator_workspace/annotator_N/
        workspace_root = os.environ.get('WORKSPACE_ROOT', './multi_annotator_workspace')
        config_path = os.path.join(workspace_root, f'annotator_{annotator_id}', 'config.json')
        if os.path.exists(config_path):
            loaded_config = load_config_from_file(config_path)
            if loaded_config:
                return loaded_config
    
    # 4. 返回空配置，使用默认值
    return config


# 加载配置
LOADED_CONFIG = get_config_from_args_or_env()

# ==========================
# 默认配置（可被 config 文件或 sidebar 覆盖）
# ==========================
if LOADED_CONFIG:
    # 从配置文件加载
    # video_dir 可选，如果没有则从 annotations_file 中读取视频路径
    VIDEO_DIR = LOADED_CONFIG.get("video_dir", None)
    ORIG_META_PATH = LOADED_CONFIG.get("annotations_file", "./annotations/tasks.jsonl")
    OUTPUT_DIR = LOADED_CONFIG.get("output_dir", "./output")
    ANNOTATOR_ID = LOADED_CONFIG.get("annotator_id", "unknown")
    WORKSPACE_ROOT = LOADED_CONFIG.get("workspace_root", "/home/jensen/world_model")
    st.sidebar.success(f"✅ 已加载标注者 {ANNOTATOR_ID} 的配置")
else:
    # 使用默认配置
    VIDEO_DIR = "/home/jensen/remote_jensen2/Galaxea-Open-World-Dataset-Video/part1_r1_lite/head"
    ORIG_META_PATH = "/home/jensen/remote_jensen2/Galaxea-Open-World-Dataset-Video/galaxea_subtask_label/part1_r1_lite/results_cleaned.jsonl"
    OUTPUT_DIR = "/home/jensen/remote_jensen2/Galaxea-Open-World-Dataset-Video/galaxea_subtask_label/part1_r1_lite/opt"
    ANNOTATOR_ID = None
    WORKSPACE_ROOT = None


# ==========================
# 工具函数
# ==========================
def get_file_signature(path: str) -> Tuple[float, int]:
    """Return (mtime, size) for cache invalidation."""
    try:
        stat_res = os.stat(path)
        return stat_res.st_mtime, stat_res.st_size
    except FileNotFoundError:
        return 0.0, 0


@st.cache_data(show_spinner=False)
def build_jsonl_index(meta_path: str, signature: Tuple[float, int]) -> Dict[str, int]:
    """Build an index that maps candidate keys to byte offsets inside the JSONL file."""
    if signature == (0.0, 0):
        return {}

    index: Dict[str, int] = {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                key = record.get("key")
                data = record.get("data")

                if isinstance(key, str):
                    index.setdefault(key, offset)

                if isinstance(data, dict):
                    episode_id = data.get("episode_id")
                    if isinstance(episode_id, str):
                        index.setdefault(episode_id, offset)
    except FileNotFoundError:
        return {}

    return index


def read_jsonl_entry(meta_path: str, index: Dict[str, int], key: str) -> Optional[Dict[str, Any]]:
    """Fetch a single entry from JSONL via the offset index."""
    if not key:
        return None

    offset = index.get(key)
    if offset is None:
        return None

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            f.seek(offset)
            line = f.readline()
        record = json.loads(line)
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return None

    data = record.get("data") if isinstance(record, dict) else None
    return data if isinstance(data, dict) else None


def natural_sort_key(s: str) -> List:
    """
    自然排序的 key 函数，让 ep1, ep2, ..., ep10, ep100 按数字顺序排列
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def list_videos_from_meta(meta_path: str, workspace_root: str = None) -> Dict[str, str]:
    """
    从 meta 文件中读取视频路径
    适用于视频路径已经在 meta 文件中的情况
    """
    if not os.path.exists(meta_path):
        return {}
    
    mapping = {}
    signature = get_file_signature(meta_path)
    index = build_jsonl_index(meta_path, signature)
    
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    key = record.get("key")
                    data = record.get("data", {})
                    video_path = data.get("video_path", "")
                    
                    if key and video_path:
                        # 如果是相对路径，拼接 workspace_root
                        if workspace_root and not os.path.isabs(video_path):
                            full_path = os.path.join(workspace_root, video_path)
                        else:
                            full_path = video_path
                        
                        # 检查文件是否存在
                        if os.path.exists(full_path):
                            mapping[key] = full_path
                except json.JSONDecodeError:
                    continue
    except Exception:
        return {}
    
    return mapping


def list_videos(video_dir: str, meta_path: str = None, workspace_root: str = None) -> Dict[str, str]:
    """
    列出视频文件
    优先从 meta 文件读取视频路径，如果失败则从目录扫描
    
    支持两种模式:
    1. 从 meta 文件读取（多标注者模式）
    2. 从目录扫描（传统模式）
    """
    # 模式1: 如果提供了 meta_path，尝试从中读取视频路径
    if meta_path:
        mapping = list_videos_from_meta(meta_path, workspace_root)
        if mapping:
            return mapping
    
    # 模式2: 从目录扫描
    if not video_dir or not os.path.isdir(video_dir):
        return {}
    
    mapping = {}
    
    # 首先尝试扁平结构（直接在 video_dir 下）
    direct_files = glob.glob(os.path.join(video_dir, "*.mp4"))
    if direct_files:
        for f in direct_files:
            eid = os.path.splitext(os.path.basename(f))[0]
            mapping[eid] = f
    
    # 然后尝试分类结构（video_dir/task_type/*.mp4）
    for subdir in os.listdir(video_dir):
        subdir_path = os.path.join(video_dir, subdir)
        if os.path.isdir(subdir_path):
            subdir_files = glob.glob(os.path.join(subdir_path, "*.mp4"))
            for f in subdir_files:
                # 使用 task_type_episode_id 作为 key，或者只用 episode_id
                eid = os.path.splitext(os.path.basename(f))[0]
                # 避免重复的 key
                if eid in mapping:
                    eid = f"{subdir}_{eid}"
                mapping[eid] = f
    
    # 按自然数顺序排序（使用 episode ID）
    sorted_keys = sorted(mapping.keys(), key=natural_sort_key)
    return {k: mapping[k] for k in sorted_keys}


def safe_literal_eval_list(s):
    if isinstance(s, list):
        return s
    if isinstance(s, str):
        try:
            v = ast.literal_eval(s)
            if isinstance(v, list):
                return v
        except Exception:
            pass
    return []


@st.cache_data
def load_video_info(path: str) -> Tuple[int, float, float]:
    if cv2 is None or not os.path.exists(path):
        return 0, 0.0, 0.0
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return 0, 0.0, 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()
        duration = frame_count / fps if fps > 0 else 0.0
        return frame_count, fps, duration
    except Exception:
        return 0, 0.0, 0.0


@st.cache_data(hash_funcs={type(None): lambda _: None})
def get_frame_image(path: str, frame_idx: int):
    """读取指定帧，返回 RGB 图像（给 st.image 用）。"""
    if cv2 is None or not os.path.exists(path):
        return None
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        
        if not ok or frame is None:
            return None
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    except Exception:
        return None



@st.cache_data
def get_video_base64(path: str) -> str:
    """把本地视频文件转成 base64 字符串，方便在 HTML 里用 data URL 播放。"""
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode("utf-8")
    except Exception:
        return ""






def save_episode_meta(meta: Dict[str, Any], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    episode_id = meta.get("episode_id", "unknown")
    output_path = os.path.join(output_dir, f"{episode_id}.json")

    # 创建一个副本用于保存，以免修改原始 meta 对象
    meta_to_save = copy.deepcopy(meta)

    # 移除 img_id_list（确保不保存）
    meta_to_save.pop("img_id_list", None)
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(meta_to_save, f, ensure_ascii=False, indent=2)
        st.success(f"✅ 标注已保存到: {output_path}")
    except Exception as e:
        st.error(f"保存失败: {e}")


def normalize_steps(steps: List[Dict[str, Any]], frame_count: int) -> List[Dict[str, Any]]:
    """
    标准化 steps，确保每个 step 都有 start_frame 和 end_frame
    - 如果缺少 start_frame，使用 0
    - 如果缺少 end_frame，使用视频最后一帧 (frame_count - 1)
    - 不限制已有的帧号范围（因为 frame_count 可能不准确）
    """
    normalized = []
    max_frame = max(frame_count - 1, 0)
    
    for step in steps:
        if not isinstance(step, dict):
            continue
        
        # 获取或设置默认值
        start_frame = step.get("start_frame")
        end_frame = step.get("end_frame")
        
        # 处理 start_frame：缺失时使用 0
        if start_frame is None or start_frame == "":
            start_frame = 0
        else:
            try:
                start_frame = int(start_frame)
            except (ValueError, TypeError):
                start_frame = 0
        
        # 处理 end_frame：缺失时使用 max_frame
        if end_frame is None or end_frame == "":
            end_frame = max_frame
        else:
            try:
                end_frame = int(end_frame)
            except (ValueError, TypeError):
                end_frame = max_frame
        
        # 确保 start_frame >= 0
        start_frame = max(0, start_frame)
        
        # 确保 end_frame >= start_frame（但不限制上限）
        if end_frame < start_frame:
            # 如果 end_frame 小于 start_frame，使用默认值
            end_frame = max_frame
        
        normalized.append({
            "step_description": step.get("step_description", ""),
            "start_frame": start_frame,
            "end_frame": end_frame,
        })
    
    return normalized


def load_episode_from_original_meta(episode_id: str, meta_path: str):
    if not os.path.exists(meta_path):
        return {}

    signature = get_file_signature(meta_path)
    index = build_jsonl_index(meta_path, signature)
    if not index:
        return {}

    candidates: List[str] = []

    def add_candidate(value: Any):
        if isinstance(value, str) and value and value not in candidates:
            candidates.append(value)

    add_candidate(episode_id)

    try:
        num_part = re.findall(r"\d+", episode_id)
        if num_part:
            episode_num = int(num_part[-1])
            prefix = os.path.basename(os.path.dirname(meta_path))
            add_candidate(f"{prefix}_ep{episode_num}")
            add_candidate(f"episode_{episode_num:06d}")
            add_candidate(f"episode_{episode_num}")
    except (ValueError, TypeError, IndexError):
        pass

    entry = None
    for candidate in candidates:
        entry = read_jsonl_entry(meta_path, index, candidate)
        if entry:
            break

    if entry is None:
        return {}

    task = entry.get("task", "")
    frame_count = entry.get("frame_count", 0)
    video_path = entry.get("video_path", "")
    result_raw = entry.get("result", {})

    if isinstance(result_raw, str):
        try:
            result = json.loads(result_raw)
        except Exception:
            result = {}
    else:
        result = result_raw

    # 不在这里标准化 steps，因为 frame_count 可能不准确
    # 标准化将在主程序中、获取真实视频帧数后进行
    raw_steps = result.get("steps", [])

    return {
        "episode_id": episode_id,
        "task": task,
        "frame_count": frame_count,
        "video_path": video_path,
        "result": {
            "task_summary": result.get("task_summary", ""),
            "steps": raw_steps,
        },
    }


def load_episode_meta(episode_id: str, meta_path: str, output_dir: str):
    """
    加载 episode 的元数据和标注。
    返回: (meta_dict, annotation_status)
    annotation_status: 'new'(未标注), 'annotated'(已标注), 'reannotate'(需要重新标注)
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{episode_id}.json")

    # 优先加载手动标注
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data.setdefault("episode_id", episode_id)
            data.setdefault("result", {})
            data["result"].setdefault("steps", [])
            data["result"].setdefault("task_summary", data.get("task", ""))
            # 标记为已标注
            status = data.get("annotation_status", "annotated")
            return data, status
        except Exception as e:
            st.error(f"读取已保存标注失败: {e}")
            return None, None

    # 否则从原始 meta 加载
    data = load_episode_from_original_meta(episode_id, meta_path)
    if data:
        # 标记为未标注
        data["annotation_status"] = "new"
        return data, "new"

    # 默认空结构
    return {
        "episode_id": episode_id,
        "task": "",
        "frame_count": 0,
        "video_path": "",
        "result": {
            "task_summary": "",
            "steps": [],
        },
        "annotation_status": "new",
    }, "new"


def get_current_step(steps, frame):
    """
    根据当前帧号查找对应的 step
    增强鲁棒性：处理缺失 start_frame 或 end_frame 的情况
    """
    for idx, s in enumerate(steps):
        try:
            start_frame = s.get("start_frame", 0)
            end_frame = s.get("end_frame", -1)
            
            # 处理可能的非数字类型
            if start_frame is None or start_frame == "":
                start_frame = 0
            else:
                start_frame = int(start_frame)
            
            if end_frame is None or end_frame == "":
                end_frame = float('inf')  # 如果没有 end_frame，认为到视频结束
            else:
                end_frame = int(end_frame)
            
            if start_frame <= frame <= end_frame:
                return idx, s
        except Exception:
            pass
    return None, None


def classify_episodes(video_ids: List[str], meta_path: str, output_dir: str) -> Tuple[List[str], List[str]]:
    """
    分类视频：未标注、已标注
    返回: (unannotated_list, annotated_list)
    
    优化版：只检查 output_dir 中是否存在对应的 .json 文件，不加载元数据
    """
    unannotated = []
    annotated = []
    
    for episode_id in video_ids:
        output_path = os.path.join(output_dir, f"{episode_id}.json")
        if os.path.exists(output_path):
            annotated.append(episode_id)
        else:
            unannotated.append(episode_id)
    
    return unannotated, annotated


def create_chunks(items: List[str], chunk_size: int = 50) -> Dict[str, List[str]]:
    """
    将列表分块，返回 {chunk_label: [items...]}
    """
    chunks = {}
    for i in range(0, len(items), chunk_size):
        chunk_items = items[i:i + chunk_size]
        start_idx = i + 1
        end_idx = min(i + chunk_size, len(items))
        chunk_label = f"Chunk {start_idx}-{end_idx} ({len(chunk_items)} 个)"
        chunks[chunk_label] = chunk_items
    return chunks


def get_chunk_labels(total_count: int, chunk_size: int = 50) -> List[str]:
    """
    只生成块标签列表，不实际分割数据（懒加载用）
    """
    labels = []
    for i in range(0, total_count, chunk_size):
        start_idx = i + 1
        end_idx = min(i + chunk_size, total_count)
        count = end_idx - start_idx + 1
        labels.append(f"Chunk {start_idx}-{end_idx} ({count} 个)")
    return labels


def get_chunk_labels_with_annotation_count(all_episode_ids: List[str], unannotated: List[str], chunk_size: int = 50) -> List[str]:
    """
    根据原始完整列表生成块标签，显示每个块中未标注的数量
    all_episode_ids: 所有视频的完整列表（原始顺序）
    unannotated: 未标注的视频列表
    """
    unannotated_set = set(unannotated)
    labels = []
    
    for i in range(0, len(all_episode_ids), chunk_size):
        start_idx = i + 1
        end_idx = min(i + chunk_size, len(all_episode_ids))
        chunk_items = all_episode_ids[i:i + chunk_size]
        
        # 计算这个块中未标注的数量
        unannotated_count = sum(1 for ep_id in chunk_items if ep_id in unannotated_set)
        
        # 只有当这个chunk中有未标注的视频时才添加
        if unannotated_count > 0:
            labels.append(f"Chunk {start_idx}-{end_idx} ({unannotated_count} 个)")
    
    return labels


def get_annotated_chunk_labels_with_source(all_episode_ids: List[str], annotated: List[str], chunk_size: int = 50) -> List[str]:
    """
    根据原始完整列表生成块标签，显示每个块中已标注的数量
    all_episode_ids: 所有视频的完整列表（原始顺序）
    annotated: 已标注的视频列表
    """
    annotated_set = set(annotated)
    labels = []
    
    for i in range(0, len(all_episode_ids), chunk_size):
        start_idx = i + 1
        end_idx = min(i + chunk_size, len(all_episode_ids))
        chunk_items = all_episode_ids[i:i + chunk_size]
        
        # 计算这个块中已标注的数量
        annotated_count = sum(1 for ep_id in chunk_items if ep_id in annotated_set)
        
        # 只有当这个chunk中有已标注的视频时才添加
        if annotated_count > 0:
            labels.append(f"Chunk {start_idx}-{end_idx} ({annotated_count} 个)")
    
    return labels


def get_chunk_items(items: List[str], chunk_label: str, chunk_size: int = 50) -> List[str]:
    """
    根据块标签提取对应的项目（懒加载用）
    """
    # 从标签中解析起始索引，如 "Chunk 1-50 (50 个)" -> 1
    match = re.match(r'Chunk (\d+)-', chunk_label)
    if not match:
        return []
    
    start_idx = int(match.group(1)) - 1  # 转为 0-based index
    return items[start_idx:start_idx + chunk_size]


def get_unannotated_chunk_items(all_episode_ids: List[str], unannotated: List[str], chunk_label: str, chunk_size: int = 50) -> List[str]:
    """
    根据块标签从原始列表中提取该chunk中未标注的视频
    all_episode_ids: 所有视频的完整列表（原始顺序）
    unannotated: 未标注的视频列表
    """
    if chunk_label is None:
        return []
    # 从标签中解析起始索引，如 "Chunk 1-50 (49 个)" -> 1
    match = re.match(r'Chunk (\d+)-', chunk_label)
    if not match:
        return []
    
    start_idx = int(match.group(1)) - 1  # 转为 0-based index
    chunk_items = all_episode_ids[start_idx:start_idx + chunk_size]
    
    # 只返回未标注的
    unannotated_set = set(unannotated)
    return [ep_id for ep_id in chunk_items if ep_id in unannotated_set]


def get_annotated_chunk_items_with_source(all_episode_ids: List[str], annotated: List[str], chunk_label: str, chunk_size: int = 50) -> List[Tuple[str, str]]:
    """
    根据块标签从原始列表中提取该chunk中已标注的视频，并附带原始chunk信息
    all_episode_ids: 所有视频的完整列表（原始顺序）
    annotated: 已标注的视频列表
    返回: [(episode_id, "原Chunk X-Y"), ...]
    """
    if chunk_label is None:
        return []
    # 从标签中解析起始索引
    match = re.match(r'Chunk (\d+)-', chunk_label)
    if not match:
        return []
    
    start_idx = int(match.group(1)) - 1  # 转为 0-based index
    end_idx = int(re.search(r'-(\d+)', chunk_label).group(1))
    chunk_items = all_episode_ids[start_idx:start_idx + chunk_size]
    
    # 只返回已标注的
    annotated_set = set(annotated)
    result = []
    for ep_id in chunk_items:
        if ep_id in annotated_set:
            # 显示当前chunk信息
            source_info = f"原Chunk {start_idx + 1}-{end_idx}"
            result.append((ep_id, source_info))
    
    return result


# ==========================
# Streamlit 主程序
# ==========================
def main():
    st.set_page_config(page_title="视频子任务标注工具", layout="wide")

    # 显示标题和标注者信息
    if ANNOTATOR_ID is not None:
        st.title(f"📽️ 视频子任务标注工具（v0.3） - 标注者 {ANNOTATOR_ID}")
    else:
        st.title("📽️ 视频子任务标注工具（v0.3）")

    # 在创建 widgets 之前，检查是否需要重置状态
    if st.session_state.get("_reset_selection", False):
        # 初始化下拉框的默认值为 None
        st.session_state["select_unannotated_chunk"] = None
        st.session_state["select_unannotated"] = None
        st.session_state["select_annotated_chunk"] = None
        st.session_state["select_annotated"] = None
        st.session_state["_reset_selection"] = False

    # Sidebar 配置
    st.sidebar.header("🔧 配置")
    VIDEO_DIR_LOCAL = st.sidebar.text_input("视频目录 VIDEO_DIR", value=VIDEO_DIR or "")
    ORIG_META_PATH_LOCAL = st.sidebar.text_input("原始总标注 JSON 路径", value=ORIG_META_PATH)
    OUTPUT_DIR_LOCAL = st.sidebar.text_input("输出目录 OUTPUT_DIR", value=OUTPUT_DIR)
    CHUNK_SIZE = st.sidebar.number_input("每块视频数量", min_value=10, max_value=500, value=50, step=10)

    # 优先从 meta 文件读取视频路径，如果失败则从目录扫描
    WORKSPACE_ROOT_LOCAL = st.sidebar.text_input("工作空间根目录 WORKSPACE_ROOT", value=WORKSPACE_ROOT or "")
    video_mapping = list_videos(VIDEO_DIR_LOCAL, ORIG_META_PATH_LOCAL, WORKSPACE_ROOT_LOCAL if WORKSPACE_ROOT_LOCAL else None)
    if not video_mapping:
        st.error(f"目录 {VIDEO_DIR_LOCAL} 中没有 mp4 文件")
        return

    episode_ids = list(video_mapping.keys())
    
    # 分类视频（优化版：只检查文件是否存在，不加载元数据）
    unannotated, annotated = classify_episodes(episode_ids, ORIG_META_PATH_LOCAL, OUTPUT_DIR_LOCAL)
    
    # 生成块标签（按原始列表位置分chunk）
    unannotated_chunk_labels = get_chunk_labels_with_annotation_count(episode_ids, unannotated, chunk_size=CHUNK_SIZE)
    annotated_chunk_labels = get_annotated_chunk_labels_with_source(episode_ids, annotated, chunk_size=CHUNK_SIZE)
    
    # 处理保存后的跳转逻辑
    if "_next_episode" in st.session_state and "_next_status" in st.session_state and "_next_chunk" in st.session_state:
        next_ep = st.session_state["_next_episode"]
        next_status = st.session_state["_next_status"]
        next_chunk = st.session_state["_next_chunk"]
        
        # 应用跳转（同时设置 chunk 和 episode）
        if next_status == "new":
            st.session_state["select_unannotated_chunk"] = next_chunk
            st.session_state["select_unannotated"] = next_ep
            # 确保已标注的被重置为 None
            st.session_state["select_annotated_chunk"] = None
            st.session_state["select_annotated"] = None
        elif next_status == "annotated":
            st.session_state["select_annotated_chunk"] = next_chunk
            st.session_state["select_annotated"] = next_ep
            # 确保未标注的被重置为 None
            st.session_state["select_unannotated_chunk"] = None
            st.session_state["select_unannotated"] = None
        
        # 清除标志
        del st.session_state["_next_episode"]
        del st.session_state["_next_status"]
        del st.session_state["_next_chunk"]
    
    # 处理重置到块选择的逻辑
    if "_reset_to_chunk_selection" in st.session_state and "_reset_status" in st.session_state:
        reset_status = st.session_state["_reset_status"]
        
        # 重置对应的选择为 None
        if reset_status == "new":
            st.session_state["select_unannotated_chunk"] = None
            st.session_state["select_unannotated"] = None
        elif reset_status == "annotated":
            st.session_state["select_annotated_chunk"] = None
            st.session_state["select_annotated"] = None
        
        # 清除标志
        del st.session_state["_reset_to_chunk_selection"]
        del st.session_state["_reset_status"]
    
    st.subheader("📋 选择要标注的视频")
    
    # 两栏显示不同状态的统计
    stat_col1, stat_col2 = st.columns(2)
    with stat_col1:
        st.metric("未标注的", len(unannotated))
    with stat_col2:
        st.metric("已经标注的", len(annotated))
    
    # 两栏下拉框（分块 + 视频）
    select_col1, select_col2 = st.columns(2)
    
    selected_episode = None
    current_status = None
    
    with select_col1:
        st.markdown("##### 📝 未标注的")
        if unannotated_chunk_labels:
            # 第一层：选择块
            # chunk_labels_unannotated = ["--- 选择块 ---"] + unannotated_chunk_labels
            chunk_labels_unannotated = unannotated_chunk_labels
            selected_chunk_unannotated = st.selectbox(
                "1️⃣ 选择块", 
                chunk_labels_unannotated,
                index=None,
                key="select_unannotated_chunk"
            )
            
            # 第二层：选择具体视频（懒加载：只在选择块后才提取数据）
            if selected_chunk_unannotated is not None:
                chunk_videos = get_unannotated_chunk_items(episode_ids, unannotated, selected_chunk_unannotated, chunk_size=CHUNK_SIZE)
                selected_from_unannotated = st.selectbox(
                    "2️⃣ 选择视频", 
                    # ["--- 选择 ---"] + 
                    chunk_videos,
                    index=None,
                    key="select_unannotated"
                )
                if selected_from_unannotated is not None:
                    selected_episode = selected_from_unannotated
                    current_status = "new"
                    # 清空已标注的选择，保持互斥
                    if st.session_state.get("select_annotated") is not None:
                        st.session_state["select_annotated_chunk"] = None
                        st.session_state["select_annotated"] = None
            else:
                st.info("👆 请先选择一个块")
        else:
            st.write("（无未标注的）")
    
    with select_col2:
        st.markdown("##### ✅ 已经标注的")
        if annotated_chunk_labels:
            # 第一层：选择块
            # chunk_labels_annotated = ["--- 选择块 ---"] + annotated_chunk_labels
            chunk_labels_annotated = annotated_chunk_labels
            selected_chunk_annotated = st.selectbox(
                "1️⃣ 选择块", 
                chunk_labels_annotated,
                index=None,
                key="select_annotated_chunk"
            )
            
            # 第二层：选择具体视频（懒加载：只在选择块后才提取数据）
            if selected_chunk_annotated is not None:
                chunk_videos_with_source = get_annotated_chunk_items_with_source(
                    episode_ids, annotated, selected_chunk_annotated, chunk_size=CHUNK_SIZE
                )
                # 创建显示选项，格式: "episode_id (原Chunk X-Y)"
                # video_options = ["--- 选择 ---"] + [f"{ep_id} ({source})" for ep_id, source in chunk_videos_with_source]
                video_options = [f"{ep_id} ({source})" for ep_id, source in chunk_videos_with_source]
                selected_from_annotated = st.selectbox(
                    "2️⃣ 选择视频", 
                    video_options,
                    index=None,
                    key="select_annotated"
                )
                if selected_from_annotated is not None:
                    # 提取实际的 episode_id（去掉来源信息）
                    actual_episode_id = selected_from_annotated.split(" (")[0]
                    # 只有在未标注的没有选择时才生效
                    if selected_episode is None:
                        selected_episode = actual_episode_id
                        current_status = "annotated"
                    else:
                        # 如果未标注的已有选择，将已标注的重置为 None
                        st.session_state["select_annotated_chunk"] = None
                        st.session_state["select_annotated"] = None
            else:
                st.info("👆 请先选择一个块")
        else:
            st.write("（无已标注的）")
    
    if selected_episode is None:
        st.info("👆 请从上方两个下拉框中选择一个视频开始标注")
        return
    
    # 切换 episode 时清状态
    if "current_episode" not in st.session_state:
        st.session_state["current_episode"] = None
    if st.session_state["current_episode"] != selected_episode:
        # 保存当前 episode ID 后，清空所有 step 相关的 state
        for key in list(st.session_state.keys()):
            if key.startswith("desc_") or key.startswith("start_") or key.startswith("end_") or key == "current_frame":
                del st.session_state[key]
        st.session_state["current_episode"] = selected_episode

    video_path = video_mapping[selected_episode]

    # 加载 meta 和标注状态
    meta, annotation_status = load_episode_meta(selected_episode, ORIG_META_PATH_LOCAL, OUTPUT_DIR_LOCAL)
    if meta is None:
        st.error("无法加载该视频的元数据")
        return
    
    meta["episode_id"] = selected_episode
    meta["video_path"] = video_path

    # 检查关键信息是否存在（不再依赖 img_id_list）
    task = meta.get("task", "")
    frame_count_meta = int(meta.get("frame_count", 0) or 0)
    is_data_valid = bool(task and frame_count_meta > 0)

    # 视频帧信息
    frame_count_video, fps, duration = load_video_info(video_path)
    frame_count = frame_count_video if frame_count_video > 0 else frame_count_meta
    meta["frame_count"] = frame_count

    # Step 信息
    result = meta.get("result", {})
    task_summary = result.get("task_summary", meta.get("task", ""))
    original_steps = result.get("steps", [])
    
    # 标准化 steps（使用真实的视频帧数）
    original_steps = normalize_steps(original_steps, frame_count)

    # 如果没有 steps 且 frame_count > 0，创建默认 step
    if not original_steps and frame_count > 0:
        original_steps = [{
            "step_description": "",
            "start_frame": 0,
            "end_frame": frame_count - 1,
        }]

    num_steps = len(original_steps)

    # 当前 steps（带 session_state）
    current_steps = []
    for i in range(num_steps):
        base = original_steps[i]
        # 使用包含 episode_id 的 key，确保不同视频的数据互不干扰
        desc_key = f"desc_{selected_episode}_{i}"
        start_key = f"start_{selected_episode}_{i}"
        end_key = f"end_{selected_episode}_{i}"
        
        if desc_key in st.session_state:
            desc = st.session_state[desc_key]
        else:
            desc = base.get("step_description", "")
        
        if start_key in st.session_state:
            start = st.session_state[start_key]
        else:
            # 增强鲁棒性：处理缺失或非数字的 start_frame
            start_raw = base.get("start_frame", 0)
            try:
                start = int(start_raw) if start_raw not in (None, "") else 0
            except (ValueError, TypeError):
                start = 0
        
        if end_key in st.session_state:
            end = st.session_state[end_key]
        else:
            # 增强鲁棒性：处理缺失或非数字的 end_frame
            end_raw = base.get("end_frame")
            try:
                end = int(end_raw) if end_raw not in (None, "") else (frame_count - 1)
            except (ValueError, TypeError):
                end = frame_count - 1
        
        current_steps.append({
            "step_description": desc,
            "start_frame": int(start),
            "end_frame": int(end) if i < num_steps - 1 else frame_count - 1,
        })
    
    st.markdown("---")
    # 初始化 current_frame
    current_frame = st.session_state.get("current_frame", 0)

    # 上半：左图像右信息
    col1, col2 = st.columns([1, 1])


    with col1:
        st.subheader("🎞 视频预览（视频条 & 帧滑条已对齐）")

        # 把视频读成 base64，嵌入 HTML5 video
        b64 = get_video_base64(video_path)
        if not b64:
            st.error("无法读取视频文件，请检查路径和权限。")
        else:
            # 有些视频 meta fps 可能是 0，这里兜底成 30
            effective_fps = fps if fps and fps > 0 else 30.0

            html = f"""
                    <div style="width: 500px;">
                    <video id="video" width="500" controls>
                        <source src="data:video/mp4;base64,{b64}" type="video/mp4" />
                        您的浏览器不支持 HTML5 视频。
                    </video>

                    <!-- 帧滑条：与视频进度条严格同步 -->
                    <input
                        type="range"
                        id="frameSlider"
                        min="0"
                        max="{max(frame_count - 1, 0)}"
                        value="{current_frame}"
                        style="width: 500px; margin-top: 8px;"
                    />

                    <div style="margin-top: 4px; font-size: 18px; font-weight: bold;">
                        当前帧: 
                        <span id="frameLabel" style="color: red; font-size: 22px; font-weight: bold;">
                            {current_frame}
                        </span>
                        / {max(frame_count - 1, 0)}
                    </div>


                    <div style="margin-top: 4px; font-size: 12px; color: #666;">
                        提示：暂停后看上面的“当前帧”数字，在下方 Step 的 start_frame / end_frame 中手动填写该帧号。
                    </div>
                    </div>

                    <script>
                    (function() {{
                    const fps = {effective_fps:.6f};
                    const video = document.getElementById("video");
                    const slider = document.getElementById("frameSlider");
                    const label = document.getElementById("frameLabel");

                    if (!video || !slider || !label) {{
                        return;
                    }}

                    let isSyncFromSlider = false;

                    // 视频播放时，根据时间更新滑条和帧号
                    video.addEventListener("timeupdate", function() {{
                        if (isSyncFromSlider) return;
                        const frame = Math.round(video.currentTime * fps);
                        slider.value = frame;
                        label.textContent = frame;
                    }});

                    // 拖动帧滑条时，跳转到对应帧
                    slider.addEventListener("input", function() {{
                        const frame = parseInt(slider.value);
                        const time = frame / fps;
                        isSyncFromSlider = true;
                        video.currentTime = time;
                        label.textContent = frame;
                        // 短暂延时解除“滑条驱动”标记，避免相互触发
                        setTimeout(() => {{ isSyncFromSlider = false; }}, 100);
                    }});
                    }})();
                    </script>
                    """
            components.html(html, height=580, scrolling=False)



    with col2:
        st.subheader("ℹ️ 视频信息")
        st.write(f"**Episode:** `{selected_episode}`")
        st.write(f"**视频路径:** `{video_path}`")
        st.write(f"**Frames:** {frame_count}")
        if fps > 0:
            st.write(f"**FPS:** {fps:.2f}")
            st.write(f"**Duration:** {duration:.2f}s")
        st.markdown("""
            <style>
            .custom-label {
                font-weight: 600;
                margin-top: 4px;
                margin-bottom: 4px;
            }
            .custom-textbox {
                background-color: #f2f4f8;      /* 比默认灰更干净 */
                padding: 10px 16px;
                border-radius: 8px;
                text-align: center;             /* 文本居中 */
                font-size: 14px;
                color: #333;
                border: 1px solid #d0d0d0;
                margin-bottom: 16px;            /* 和下面内容拉开距离 */
            }
            </style>
            """, unsafe_allow_html=True)

        # Task 原始描述
        st.markdown("<div class='custom-label'>Task 原始描述：</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='custom-textbox'>{meta.get('task','')}</div>",
            unsafe_allow_html=True,
        )

        # Task LLM 处理后描述
        st.markdown("<div class='custom-label'>Task LLM 处理后描述：</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='custom-textbox'>{task_summary}</div>",
            unsafe_allow_html=True,
        )


    # Step 编辑区
    st.subheader("🧩 子任务 Step 标注（可编辑）")
    
    # 初始化 session_state 中的 steps 列表（如果不存在）
    if f"steps_{selected_episode}" not in st.session_state:
        st.session_state[f"steps_{selected_episode}"] = current_steps.copy()
    
    # 获取当前 episode 的 steps
    working_steps = st.session_state[f"steps_{selected_episode}"]
    
    # 添加新 Step 按钮
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("➕ 新增 Step", disabled=not is_data_valid):
            # 添加一个新的空 step
            last_end = working_steps[-1]["end_frame"] if working_steps else 0
            new_step = {
                "step_description": "",
                "start_frame": min(last_end + 1, max(frame_count - 1, 0)),
                "end_frame": max(frame_count - 1, 0),
            }
            working_steps.append(new_step)
            st.rerun()
    
    updated_steps = []

    for i, s in enumerate(working_steps):
        # st.markdown(f"#### Step {i + 1}")
        # 左右两列：左侧是表单，右侧是预览视频
        left_col, right_col = st.columns([3, 2])

        # ===== 左侧：step_description / start_frame / end_frame（纵向排布）=====
        with left_col:
            # Step 标题和删除按钮在同一行
            header_col1, header_col2 = st.columns([4, 1])
            with header_col1:
                st.markdown(f"<div style='font-size:18px; font-weight:700;'>Step {i+1}</div>", unsafe_allow_html=True)
            with header_col2:
                # 删除按钮（至少保留一个 step）
                if len(working_steps) > 1:
                    delete_key = f"delete_{selected_episode}_{i}"
                    if st.button("🗑️", key=delete_key, disabled=not is_data_valid, help="删除此 Step"):
                        # 删除当前 step
                        working_steps.pop(i)
                        # 清空相关的 session_state
                        for key in list(st.session_state.keys()):
                            if key.startswith(f"desc_{selected_episode}_") or \
                               key.startswith(f"start_{selected_episode}_") or \
                               key.startswith(f"end_{selected_episode}_"):
                                del st.session_state[key]
                        st.rerun()
            
            desc_key = f"desc_{selected_episode}_{i}"
            start_key = f"start_{selected_episode}_{i}"
            end_key = f"end_{selected_episode}_{i}"

            desc = st.text_input(
                "description",
                s["step_description"],
                key=desc_key,
                disabled=not is_data_valid,
            )

            start = st.number_input(
                "start_frame",
                min_value=0,
                max_value=max(frame_count - 1, 0),
                value=s["start_frame"],
                step=1,
                key=start_key,
                disabled=not is_data_valid,
            )

            end = st.number_input(
                "end_frame",
                min_value=0,
                max_value=max(frame_count - 1, 0),
                value=min(s["end_frame"], frame_count - 1),
                step=1,
                key=end_key,
                disabled=not is_data_valid,
            )

        # ===== 右侧：片段播放器（只播放 start→end 区间）=====
        with right_col:

            # 防止 start > end 或越界
            safe_start = max(0, int(start))
            safe_end = min(int(end), max(frame_count - 1, 0))
            if safe_end < safe_start:
                safe_end = safe_start

            fps_effective = fps if fps and fps > 0 else 30.0
            clip_start_sec = float(safe_start) / float(fps_effective)
            clip_end_sec = float(safe_end) / float(fps_effective)

            video_b64 = get_video_base64(video_path)

            html_clip = f"""
                <div style="width: 260px;">
                <video id="clip_video_{i}"  style="width: 350px; height: 240px; object-fit: cover; border-radius: 6px;"  controls>
                    <source src="data:video/mp4;base64,{video_b64}" type="video/mp4" />
                </video>
                <div style="font-size: 12px; color:#666; margin-top:4px;">
                    播放帧区间: {safe_start} → {safe_end}
                </div>
                </div>

                <script>
                (function() {{
                const v = document.getElementById("clip_video_{i}");
                if (!v) return;

                const start = {clip_start_sec};
                const end = {clip_end_sec};

                // 元数据加载完成后跳到 start
                v.addEventListener("loadedmetadata", function() {{
                    v.currentTime = start;
                }});

                // 播放过程中超出 end 就暂停并回到 start
                v.addEventListener("timeupdate", function() {{
                    if (v.currentTime < start) {{
                    v.currentTime = start;
                    }}
                    if (v.currentTime > end) {{
                    v.pause();
                    v.currentTime = start;
                    }}
                }});

                // 用户拖动进度条时，限制在 [start, end] 范围内
                v.addEventListener("seeking", function() {{
                    if (v.currentTime < start) {{
                    v.currentTime = start;
                    }}
                    if (v.currentTime > end) {{
                    v.currentTime = end;
                    }}
                }});
                }})();
                </script>
    """
            components.html(html_clip, height=280, scrolling=False)

        # ===== 收集更新后的 step 信息 =====
        updated_steps.append({
            "step_description": desc,
            "start_frame": int(start),
            "end_frame": int(end),
        })
        
        st.markdown("<hr style='margin:6px 0; border:0; border-top:1px solid #ddd;'>", unsafe_allow_html=True)

        # ==========================
        # Step 区间合法性校验
        # 1) 每个 step: start_frame < end_frame
        # 2) 相邻 step: end_i < start_{i+1}
        # ==========================
        has_step_error = False
        prev_end = None

        for i, step in enumerate(updated_steps):
            sf = step["start_frame"]
            ef = step["end_frame"]

            # 规则 1：start_frame 必须小于 end_frame
            if sf >= ef:
                st.error(f"Step {i+1}: start_frame 必须小于 end_frame，请重新输入。")
                has_step_error = True
                # 清空当前 step 对应的 session_state，使其在下次渲染时回到默认
                start_key = f"start_{selected_episode}_{i}"
                end_key = f"end_{selected_episode}_{i}"
                if start_key in st.session_state:
                    del st.session_state[start_key]
                if end_key in st.session_state:
                    del st.session_state[end_key]

            # 规则 2：相邻 step 区间互斥，上一段的 end 必须小于当前 start
            if prev_end is not None and prev_end >= sf:
                st.error(f"Step {i} 的 end_frame 必须小于 Step {i+1} 的 start_frame，区间不能重叠。")
                has_step_error = True
                # 清空本 step 的 start_frame，让用户重填
                start_key = f"start_{selected_episode}_{i}"
                if start_key in st.session_state:
                    del st.session_state[start_key]

            prev_end = ef



    # 保存
    if has_step_error:
        st.warning("存在不合法的 Step 区间（start/end 或相邻 Step 有重叠），请根据上面的提示修改后再保存。")
    else:
        
        if st.button("💾 保存当前视频标注", disabled=not is_data_valid):
            meta["task"] = st.session_state.get("task_text", meta.get("task", ""))
            
            # 直接使用 updated_steps，因为它们已经是界面上的真实帧号
            meta["result"] = {
                "task_summary": st.session_state.get("task_summary_text", task_summary),
                "steps": updated_steps,
            }
            
            # 标记为已标注
            meta["annotation_status"] = "annotated"
            save_episode_meta(meta, OUTPUT_DIR_LOCAL)
            
            # 智能跳转：查找当前块的下一个视频
            next_episode = None
            next_episode_display = None
            
            # 判断当前选择的是未标注的还是已标注的
            if current_status == "new":
                # 从未标注的列表中查找
                selected_chunk = st.session_state.get("select_unannotated_chunk", None)
                if selected_chunk is not None:
                    # 重新获取当前块的视频列表（保存后会变化）
                    unannotated_new, _ = classify_episodes(episode_ids, ORIG_META_PATH_LOCAL, OUTPUT_DIR_LOCAL)
                    chunk_videos = get_unannotated_chunk_items(episode_ids, unannotated_new, selected_chunk, chunk_size=CHUNK_SIZE)
                    
                    # 找到当前视频在列表中的位置
                    if selected_episode in chunk_videos:
                        current_idx = chunk_videos.index(selected_episode)
                        # 查找下一个视频
                        if current_idx + 1 < len(chunk_videos):
                            next_episode = chunk_videos[current_idx + 1]
                            next_episode_display = next_episode
                    elif len(chunk_videos) > 0:
                        # 当前视频已标注，取第一个未标注的
                        next_episode = chunk_videos[0]
                        next_episode_display = next_episode
            
            elif current_status == "annotated":
                # 从已标注的列表中查找
                selected_chunk = st.session_state.get("select_annotated_chunk", None)
                if selected_chunk is not None:
                    _, annotated_new = classify_episodes(episode_ids, ORIG_META_PATH_LOCAL, OUTPUT_DIR_LOCAL)
                    chunk_videos_with_source = get_annotated_chunk_items_with_source(
                        episode_ids, annotated_new, selected_chunk, chunk_size=CHUNK_SIZE
                    )
                    chunk_videos = [ep_id for ep_id, _ in chunk_videos_with_source]
                    
                    # 找到当前视频在列表中的位置
                    if selected_episode in chunk_videos:
                        current_idx = chunk_videos.index(selected_episode)
                        # 查找下一个视频
                        if current_idx + 1 < len(chunk_videos):
                            next_episode = chunk_videos[current_idx + 1]
                            # 找到完整的显示字符串
                            for ep_id, source in chunk_videos_with_source:
                                if ep_id == next_episode:
                                    next_episode_display = f"{ep_id} ({source})"
                                    break
                    elif len(chunk_videos) > 0:
                        # 取第一个
                        next_episode = chunk_videos[0]
                        for ep_id, source in chunk_videos_with_source:
                            if ep_id == next_episode:
                                next_episode_display = f"{ep_id} ({source})"
                                break
            
            # 清空所有 step 相关的 state
            for key in list(st.session_state.keys()):
                if key.startswith("desc_") or key.startswith("start_") or key.startswith("end_") or \
                   key.startswith("steps_") or key.startswith("delete_") or key == "current_frame":
                    del st.session_state[key]
            
            # 根据是否有下一个视频，决定跳转策略
            # 使用 _next_episode、_next_chunk 和 _next_status 作为标志，在 rerun 后应用
            if next_episode:
                # 有下一个视频，设置标志位（包括 chunk 选择）
                st.session_state["_next_episode"] = next_episode_display
                st.session_state["_next_status"] = current_status
                # 保存当前的 chunk 选择
                if current_status == "new":
                    st.session_state["_next_chunk"] = st.session_state.get("select_unannotated_chunk", None)
                elif current_status == "annotated":
                    st.session_state["_next_chunk"] = st.session_state.get("select_annotated_chunk", None)
                
                st.session_state["current_episode"] = None  # 重置，下次会重新加载
                st.success(f"✅ 保存成功！正在跳转到下一个视频：{next_episode}")
            else:
                # 当前块没有更多视频，设置重置标志
                st.session_state["_reset_to_chunk_selection"] = True
                st.session_state["_reset_status"] = current_status
                st.session_state["current_episode"] = None
                st.success("✅ 保存成功！当前块已完成，请选择新的块继续标注。")
            
            time.sleep(1)
            st.rerun()  # 保存后自动刷新


if __name__ == "__main__":
    main()