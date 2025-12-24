import json
import ast
import os
import argparse
from pathlib import Path

def safe_literal_eval(s: any) -> any:
    """
    Safely evaluates a string containing a Python literal (list, dict, etc.).
    Returns the original input if it's not a string or if parsing fails.
    """
    if isinstance(s, str):
        try:
            # First, try to parse as a Python literal (handles lists like '[1, 2, 3]')
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            try:
                # If that fails, it might be a JSON string (handles dicts like '{"key": "value"}')
                return json.loads(s)
            except json.JSONDecodeError:
                print(f"警告：无法解析字符串，将保留原样: {s[:100]}...")
                return s  # Return original string if all parsing fails
    return s

def preprocess_json_file(input_path: Path, output_path: Path):
    """
    Loads a JSON file, cleans up stringified fields ('result'),
    uses img_id_list to convert frame indices to actual image IDs,
    and removes the img_id_list field from output.
    """
    if not input_path.exists():
        print(f"错误：输入文件未找到: {input_path}")
        return

    print(f"正在从 {input_path} 加载数据...")
    with open(input_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"错误：无法解析输入的 JSON 文件。请检查文件格式。 {e}")
            return
    cleaned_data = {}
    for key, value in data.items():
        if isinstance(value, dict):
            cleaned_entry = value.copy()
            
            # 先解析 'img_id_list' 字段
            img_id_list = []
            if 'img_id_list' in cleaned_entry:
                img_id_list = safe_literal_eval(cleaned_entry['img_id_list'])
                if not isinstance(img_id_list, list):
                    print(f"警告：{key} 的 img_id_list 不是列表，跳过转换")
                    img_id_list = []
            
            # 清理 'result' 字段并转换 frame 索引
            if 'result' in cleaned_entry:
                result = safe_literal_eval(cleaned_entry['result'])
                if isinstance(result, dict) and 'steps' in result:
                    # 遍历每个 step，将 start_frame 和 end_frame 从索引转换为实际 ID
                    for step in result['steps']:
                        if 'start_frame' in step and img_id_list:
                            start_idx = step['start_frame']
                            if 0 <= start_idx < len(img_id_list):
                                step['start_frame'] = img_id_list[start_idx]
                        
                        if 'end_frame' in step and img_id_list:
                            end_idx = step['end_frame']
                            if 0 <= end_idx < len(img_id_list):
                                step['end_frame'] = img_id_list[end_idx]
                
                cleaned_entry['result'] = result
            
            # 删除 'img_id_list' 字段
            cleaned_entry.pop('img_id_list', None)
            
            cleaned_data[key] = cleaned_entry
        else:
            cleaned_data[key] = value

    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"正在将清理后的数据保存到 {output_path} (JSONL 格式)...")
    with open(output_path, 'w', encoding='utf-8') as f:
        # 将每个任务写成独立的 JSON 行，便于后续流式处理
        for key, entry in cleaned_data.items():
            json_line = {
                "key": key,
                "data": entry,
            }
            f.write(json.dumps(json_line, ensure_ascii=False) + '\n')
    
    
    print(f"✅ 预处理完成！已生成干净的数据文件。")

if __name__ == '__main__':
    # 将工作目录设置为项目根目录，以确保相对路径正确
    # /home/user/project/preprocess_results.py -> /home/user/project/
    os.chdir(Path(__file__).parent)

    parser = argparse.ArgumentParser(description="清理并预处理子任务标注 JSON 文件。")
    parser.add_argument(
        '--input', 
        type=str, 
        default='galaxea_subtask_label/part1_r1_lite/results.json',
        help='输入的 JSON 文件路径或目录路径（如果是目录，会处理目录下所有 JSON 文件）。'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default=None,
        help='保存清理后数据的 JSONL 文件路径或目录（输入是文件时指定文件路径，输入是目录时指定输出目录，必须提供）。'
    )
    args = parser.parse_args()

    # 使用 Path 对象处理路径
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"错误：输入路径不存在: {input_path}")
        exit(1)
    
    # 如果输入是目录，处理目录下所有 JSON 文件
    if input_path.is_dir():
        if args.output is None:
            print(f"错误：输入是目录时，必须指定 --output 输出目录")
            exit(1)
        
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"检测到目录输入，将处理目录下所有 JSON 文件...")
        json_files = list(input_path.rglob('*.json'))
        
        if not json_files:
            print(f"警告：在目录 {input_path} 下未找到 JSON 文件")
            exit(1)
        
        print(f"找到 {len(json_files)} 个 JSON 文件")
        print(f"输出目录: {output_dir}")
        
        for json_file in json_files:
            # 计算相对于输入目录的路径，在输出目录中保持相同的目录结构
            relative_path = json_file.relative_to(input_path)
            output_file = output_dir / relative_path.with_suffix('.jsonl')
            print(f"\n处理: {json_file} -> {output_file}")
            preprocess_json_file(json_file, output_file)
        
        print(f"\n✅ 所有文件处理完成！共处理 {len(json_files)} 个文件")
    
    # 如果输入是文件，使用原有逻辑
    else:
        if args.output is None:
            # 如果没有指定输出，默认在同目录下生成同名文件，扩展名改为 .jsonl
            output_file = input_path.with_suffix('.jsonl')
        else:
            output_file = Path(args.output)
        
        preprocess_json_file(input_path, output_file)
