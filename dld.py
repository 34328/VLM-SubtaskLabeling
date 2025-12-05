import os
from huggingface_hub import snapshot_download

# 配置阿里 Hugging Face 镜像（关键）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

local_dir = snapshot_download(
    repo_id="OpenGalaxea/Galaxea-Open-World-Dataset",
    repo_type="dataset",
    local_dir="./galaxea_part1",
    resume_download=True, 
    allow_patterns=["rlds/part1_r1_lite/*"],

)
