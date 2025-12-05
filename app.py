import os
import json
import glob
import ast
import copy
import re
import time
from typing import Dict, Any, List, Tuple

import base64 
import streamlit as st
import streamlit.components.v1 as components 

try:
    import cv2
except ImportError:
    cv2 = None
    st.warning("⚠ 未安装 OpenCV（cv2），无法自动读取视频帧数和 FPS，请先 `pip install opencv-python`。")


# ==========================
# 默认配置（可被 sidebar 覆盖）
# ==========================
VIDEO_DIR = "/home/unitree/桌面/label_task/episode_videos/head"
ORIG_META_PATH = "/home/unitree/桌面/label_task/galaxea_subtask_label/part1_r1_lite/results_cleaned.json"
OUTPUT_DIR = "/home/unitree/桌面/label_task/opt"


# ==========================
# 工具函数
# ==========================
def list_videos(video_dir: str) -> Dict[str, str]:
    if not os.path.isdir(video_dir):
        return {}
    files = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    mapping = {}
    for f in files:
        eid = os.path.splitext(os.path.basename(f))[0]
        mapping[eid] = f
    return mapping


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


def load_episode_from_original_meta(episode_id: str, meta_path: str):
    if not os.path.exists(meta_path):
        return {}

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        st.error(f"读取原始标注文件失败: {e}")
        return {}

    entry = None
    if isinstance(data, dict):
        # 1. 尝试直接用 episode_id 匹配 (e.g., "part1_r1_lite_ep10")
        entry = data.get(episode_id)

        # 2. 如果直接匹配失败，尝试构建几种可能的 key
        if entry is None:
            try:
                # 从 episode_id 中提取最后的数字
                num_part = re.findall(r'\d+', episode_id)
                if num_part:
                    episode_num = int(num_part[-1])
                    
                    # 尝试 key A: "part1_r1_lite_ep10" 格式
                    prefix = os.path.basename(os.path.dirname(meta_path))
                    key1 = f"{prefix}_ep{episode_num}"
                    if key1 != episode_id:
                        entry = data.get(key1)

                    # 尝试 key B: "episode_000010" 格式
                    if entry is None:
                        key2 = f"episode_{episode_num:06d}"
                        if key2 != episode_id:
                            entry = data.get(key2)
            except (ValueError, TypeError, IndexError):
                pass  # 如果解析或构建 key 失败，则忽略
    
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and item.get("episode_id") == episode_id:
                entry = item
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

    return {
        "episode_id": episode_id,
        "task": task,
        "frame_count": frame_count,
        "video_path": video_path,
        "result": {
            "task_summary": result.get("task_summary", ""),
            "steps": result.get("steps", []),
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
    for idx, s in enumerate(steps):
        try:
            if int(s.get("start_frame", 0)) <= frame <= int(s.get("end_frame", -1)):
                return idx, s
        except Exception:
            pass
    return None, None


def classify_episodes(video_ids: List[str], meta_path: str, output_dir: str) -> Tuple[List[str], List[str]]:
    """
    分类视频：未标注、已标注
    返回: (unannotated_list, annotated_list)
    """
    unannotated = []
    annotated = []
    
    for episode_id in video_ids:
        _, status = load_episode_meta(episode_id, meta_path, output_dir)
        if status == "annotated":
            annotated.append(episode_id)
        else:
            unannotated.append(episode_id)
    
    return unannotated, annotated


# ==========================
# Streamlit 主程序
# ==========================
def main():
    st.set_page_config(page_title="视频子任务标注工具", layout="wide")

    st.title("📽️ 视频子任务标注工具（v0.2）")

    # 在创建 widgets 之前，检查是否需要重置状态
    if st.session_state.get("_reset_selection", False):
        # 初始化下拉框的默认值，使其显示 "--- 选择 ---"
        st.session_state["select_unannotated"] = "--- 选择 ---"
        st.session_state["select_annotated"] = "--- 选择 ---"
        st.session_state["_reset_selection"] = False

    # Sidebar 配置
    st.sidebar.header("🔧 配置")
    VIDEO_DIR_LOCAL = st.sidebar.text_input("视频目录 VIDEO_DIR", value=VIDEO_DIR)
    ORIG_META_PATH_LOCAL = st.sidebar.text_input("原始总标注 JSON 路径", value=ORIG_META_PATH)
    OUTPUT_DIR_LOCAL = st.sidebar.text_input("输出目录 OUTPUT_DIR", value=OUTPUT_DIR)

    video_mapping = list_videos(VIDEO_DIR_LOCAL)
    if not video_mapping:
        st.error(f"目录 {VIDEO_DIR_LOCAL} 中没有 mp4 文件")
        return

    episode_ids = list(video_mapping.keys())
    
    # 分类视频
    unannotated, annotated = classify_episodes(episode_ids, ORIG_META_PATH_LOCAL, OUTPUT_DIR_LOCAL)
    
    st.subheader("📋 选择要标注的视频")
    
    # 两栏显示不同状态的统计
    stat_col1, stat_col2 = st.columns(2)
    with stat_col1:
        st.metric("未标注的", len(unannotated))
    with stat_col2:
        st.metric("已经标注的", len(annotated))
    
    # 两栏下拉框
    select_col1, select_col2 = st.columns(2)
    
    selected_episode = None
    current_status = None
    
    with select_col1:
        if unannotated:
            selected_from_unannotated = st.selectbox("📝 未标注的", ["--- 选择 ---"] + unannotated, key="select_unannotated")
            if selected_from_unannotated != "--- 选择 ---":
                selected_episode = selected_from_unannotated
                current_status = "new"
                # 清空已标注的选择，保持互斥
                if st.session_state.get("select_annotated") != "--- 选择 ---":
                    st.session_state["select_annotated"] = "--- 选择 ---"
        else:
            st.write("（无未标注的）")
    
    with select_col2:
        if annotated:
            selected_from_annotated = st.selectbox("✅ 已经标注的", ["--- 选择 ---"] + annotated, key="select_annotated")
            if selected_from_annotated != "--- 选择 ---":
                # 只有在未标注的没有选择时才生效
                if selected_episode is None:
                    selected_episode = selected_from_annotated
                    current_status = "annotated"
                else:
                    # 如果未标注的已有选择，将已标注的重置为"--- 选择 ---"
                    st.session_state["select_annotated"] = "--- 选择 ---"
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
            start = int(base.get("start_frame", 0))
        
        if end_key in st.session_state:
            end = st.session_state[end_key]
        else:
            end = int(base.get("end_frame", frame_count - 1))
        
        current_steps.append({
            "step_description": desc,
            "start_frame": int(start),
            "end_frame": int(end),
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
    # Step 编辑区
    st.subheader("🧩 子任务 Step 标注（可编辑）")
    updated_steps = []

    for i, s in enumerate(current_steps):
        # st.markdown(f"#### Step {i + 1}")
        # 左右两列：左侧是表单，右侧是预览视频
        left_col, right_col = st.columns([3, 2])

        # ===== 左侧：step_description / start_frame / end_frame（纵向排布）=====
        with left_col:
            st.markdown(f"<div style='font-size:18px; font-weight:700;'>Step {i+1}</div>", unsafe_allow_html=True)
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
                value=s["end_frame"],
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
            
            # 设置重置标志，下次运行时会重置下拉框
            st.session_state["_reset_selection"] = True
            
            # 清空所有 step 相关的 state
            for key in list(st.session_state.keys()):
                if key.startswith("desc_") or key.startswith("start_") or key.startswith("end_") or key == "current_frame":
                    del st.session_state[key]
            
            st.session_state["current_episode"] = None
            
            time.sleep(1)
            st.rerun()  # 保存后自动刷新，重置所有状态


if __name__ == "__main__":
    main()