# conda activate /nas-mmu/zhoubb/envs/glm_ocr
# vllm 0.18.0 torch 2.10.0+cu128 transformers 5.3.0
# CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /nas-mmu/zhoubb/download_models/Logics-Parsing-v2 --max-model-len 32768 --trust-remote-code --served-model-name logicsparingv2 --tensor-parallel-size 4 --port 8118 &
# CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve /nas-mmu/zhoubb/download_models/Logics-Parsing-v2 --max-model-len 32768 --trust-remote-code --served-model-name logicsparingv2 --tensor-parallel-size 4 --port 8119

import argparse
import json
import concurrent.futures
import re
import os
import sys
from tqdm import tqdm
from PIL import Image
from openai import OpenAI
import multiprocessing
import random
import base64

# --- 配置部分 ---
ports = [8118, 8119]
MODEL_NAME = "logicsparingv2"
INPUT_FILE = "MPDocBench.json"
OUTPUT_PATH = f"./markdown/{MODEL_NAME}"

# --- 并行配置 ---
# 1. 文档间并行：同时处理多少个不同的文档（使用多进程）
MAX_DOCUMENT_PROCESSES = 8
# 2. 文档内并行：处理单个文档时，同时处理多少个页面（使用多线程）
MAX_PAGE_WORKERS = 8

# --- HTML后处理函数 (保持不变) ---
def remove_lines_starting_with(text):
    lines = text.splitlines(keepends=True)
    filtered = []
    prefixes_to_remove = ('Z:')
    for line in lines:
        stripped = line.lstrip()
        if not stripped.strip():
            continue
        if stripped.startswith(prefixes_to_remove):
            continue
        filtered.append(line)
    return "".join(filtered)

def process_code_content(content: str) -> str:
    content = content.replace('```', '')
    content = re.sub(r'^\s*<pre[^>]*>', '', content, flags=re.IGNORECASE)
    content = re.sub(r'</pre>\s*$', '', content, flags=re.IGNORECASE)
    content = re.sub(r'^\s*<code[^>]*>', '', content, flags=re.IGNORECASE)
    content = re.sub(r'</code>\s*$', '', content, flags=re.IGNORECASE)
    return f"```code\n{content.strip()}\n```"

def process_pseudocode_content(content: str) -> str:
    content = content.replace('```', '')
    content = re.sub(r'^\s*<(pre|code)[^>]*>', '', content, flags=re.IGNORECASE | re.MULTILINE)
    content = re.sub(r'</(pre|code)>\s*$', '', content, flags=re.IGNORECASE | re.MULTILINE)
    math_blocks = []
    def save_math(match):
        placeholder = f"___MATH_ID_{len(math_blocks)}___"
        math_blocks.append(match.group(0))
        return placeholder
    math_pattern = r'(\$\$.*?\$\$|\$.*?\$)'
    protected_content = re.sub(math_pattern, save_math, content, flags=re.DOTALL)
    protected_content = protected_content.replace(' ', '&nbsp;')
    protected_content = protected_content.replace('\t', '&nbsp;&nbsp;&nbsp;&nbsp;')
    protected_content = protected_content.replace('\n', '<br>')
    final_content = protected_content
    for i, original_math in enumerate(math_blocks):
        placeholder = f"___MATH_ID_{i}___"
        final_content = final_content.replace(placeholder, original_math)
    return f"___\n<br>{final_content.strip()}<br>\n___"

def qwenvl_cast_html_tag(input_text: str, page_id: int) -> str:
    output = input_text
    # output = re.sub(
    #     r'<img\b[^>]*\bdata-bbox\s*=\s*"?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*"?[^>]*\/?>',
    #     rf'![](page{page_id}_\1_\2_\3_\4.jpg)',
    #     output,
    #     flags=re.IGNORECASE
    # )

    pattern = r'(<div\b[^>]*\bclass\s*=\s*"image"[^>]*>)(.*?)(<img\b[^>]*\bdata-bbox\s*=\s*"?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*"?[^>]*\/?>)(.*?)(<\/div>)'

    #    定义新的替换字符串，使用回溯引用
    #    - \1: 第1个捕获组 (<div ...>)
    #    - \2: 第2个捕获组 (<img> 之前的内容)
    #    - \8: 第8个捕获组 (<img> 之后的内容)
    #    - \9: 第9个捕获组 (</div>)
    #    - \4, \5, \6, \7: 是 img 内部的坐标捕获组 (注意序号变化了！)
    replacement_string = rf'\1\2![](page{page_id}_\4_\5_\6_\7.jpg)\8\9'

    # 3. 执行替换，仍然需要 re.DOTALL 和 re.IGNORECASE
    output = re.sub(
        pattern,
        replacement_string,
        output,
        flags=re.IGNORECASE | re.DOTALL
    )

    IMG_RE = re.compile(
        r'<img\b[^>]*\bdata-bbox\s*=\s*"?\d+,\d+,\d+,\d+"?[^>]*\/?>',
        flags=re.IGNORECASE,
    )
    output = IMG_RE.sub('', output)
    def replace_code(match):
        content = match.group(1)
        processed_content = process_code_content(content)
        return f"\n\n{processed_content}\n\n"
    code_pattern = re.compile(r'<div\b[^>]*class="code"[^>]*>(.*?)</div>', flags=re.DOTALL | re.IGNORECASE)
    output = code_pattern.sub(replace_code, output)
    def replace_pseudocode(match):
        content = match.group(1)
        processed_content = process_pseudocode_content(content)
        return f"\n\n{processed_content}\n\n"
    pseudocode_pattern = re.compile(r'<div\b[^>]*class="pseudocode"[^>]*>(.*?)</div>', flags=re.DOTALL | re.IGNORECASE)
    output = pseudocode_pattern.sub(replace_pseudocode, output)
    def strip_div(class_name: str, txt: str) -> str:
        if class_name in ['code', 'pseudocode']: return txt
        def replace_func(match):
            content = match.group(1)
            if class_name == 'chart':
                content = re.sub(r'^\s*(click\s+|style\s+|linkStyle\s+|stroke|classDef\s+|class\s+)\b.*\n?', '', content, flags=re.MULTILINE | re.IGNORECASE)
                content = re.sub(r'^\s*(?:%%|::icon).*\n?', '', content, flags=re.MULTILINE)
                content = content.strip()
                if content.startswith('mermaid'): content = '```' + content
                elif re.match(r'^```\s*mermaid', content): pass
                else: content = '```mermaid\n' + content
                if not content.endswith('```'): content += '\n```'
            if class_name == 'music':
                content = remove_lines_starting_with(content)
                content = content.strip()
                if content.startswith('abc'): content = '```' + content
                elif re.match(r'^```\s*abc', content): pass
                else: content = '```abc\n' + content
                if not content.endswith('```'): content += '\n```'
            return f"\n\n{content}\n\n"
        pattern = re.compile(rf'\s*<div\b[^>]*class="{class_name}"[^>]*>(.*?)</div>\s*', flags=re.DOTALL | re.IGNORECASE)
        return pattern.sub(replace_func, txt)
    other_classes = ['image', 'chemistry', 'table', 'formula', 'image caption', 'table caption']
    for cls in other_classes: output = strip_div(cls, output)
    output = re.sub(r'<p\b[^>]*>(.*?)</p>', r'\n\n\1\n\n', output, flags=re.DOTALL | re.IGNORECASE)
    output = output.replace(" </td>", "</td>")
    return output

# --- 并行处理逻辑 ---

def process_page_worker(args):
    """
    [内层并行] 处理单个页面的工作函数，供线程池调用。
    返回 (页面索引, 处理后的Markdown内容)。
    """
    image_path, page_idx, model_name = args
    query = "QwenVL HTML"
    openai_api_key = "EMPTY"
    port = random.choice(ports)
    openai_api_base = f"http://localhost:{port}/v1"

    # try:
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
    
    ext = image_path.lower().split('.')[-1]
    if ext == 'png': img_format = "png"
    elif ext in ('jpg', 'jpeg'): img_format = "jpeg"
    elif ext == 'webp': img_format = "webp"
    else: raise ValueError(f"Unsupported image format: {ext}")

    with open(image_path, "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode("utf-8")
    base64_qwen = f"data:image/{img_format};base64,{encoded_image}"

    chat_response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": base64_qwen}, "min_pixels": 3136, "max_pixels": 7200 * 32 * 32},
                {"type": "text", "text": query},
            ]},
        ],
        max_tokens=16384,
        temperature=0.0,
    )
    html_content = chat_response.choices[0].message.content
    normalized_content = qwenvl_cast_html_tag(html_content, page_idx + 1)
    
    return (page_idx, html_content, normalized_content)

def process_single_document(args):
    """
    [外层并行] 处理单个文档的函数，由多进程池调用。
    内部使用线程池并行处理该文档的所有页面。
    """
    image_paths, save_dir, pdf_name, model_name = args
    output_filepath = os.path.join(save_dir, pdf_name + ".md")
    mmd_output_filepath = os.path.join(save_dir, pdf_name + ".mmd")

    if os.path.exists(output_filepath) and os.path.exists(mmd_output_filepath):
        return f"Skipped (already exists): {pdf_name}"

    try:
        page_tasks = [(path, idx, model_name) for idx, path in enumerate(image_paths)]
        page_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PAGE_WORKERS) as executor:
            future_to_page = {executor.submit(process_page_worker, task): task for task in page_tasks}
            
            # 由于此函数在多进程中运行，内部进度条可能会相互干扰。
            # 这里我们不使用内部tqdm，只在最外层显示文档级别的进度。
            for future in concurrent.futures.as_completed(future_to_page):
                page_results.append(future.result())

        page_results.sort(key=lambda x: x[0])
        full_md_content = "\n\n".join([content for idx, mmd_content, content in page_results])
        full_mmd_content = "\n\n".join([mmd_content for idx, mmd_content, content in page_results])
        
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(full_md_content)
        with open(mmd_output_filepath, "w", encoding="utf-8") as f:
            f.write(full_mmd_content)

        return f"Success: {pdf_name}"
    except Exception as e:
        error_msg = f"Failed to process document {pdf_name}: {e}"
        print(error_msg, file=sys.stderr)
        return f"Error: {pdf_name}"

def get_pdf_images(json_path):
    data = json.load(open(json_path))
    raw_data = {}
    for item in data:
        page_info = item["page_info"]
        images_list = page_info["images_list"]
        annotations_list = page_info["annotations_list"]
        image_path = page_info["image_path"]
        pdf_name = os.path.splitext(image_path)[0]
        if pdf_name not in raw_data:
            raw_data[pdf_name] = []
            page_id = 0
            for img, ann in zip(images_list, annotations_list):
                raw_data[pdf_name].append((page_id, img, ann))
                page_id += 1
        else:
            print(f"Warning: duplicate pdf_name {pdf_name} found. Skipping.")
            continue
    return raw_data

if __name__ == '__main__':
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    raw_data = get_pdf_images(INPUT_FILE)

    data = [[img_path for (index, img_path, json_path) in v] for k, v in raw_data.items()]
    pdf_names = [k for k in raw_data.keys()]
    
    all_doc_tasks = [
        (image_paths, OUTPUT_PATH, pdf_name, MODEL_NAME) 
        for image_paths, pdf_name in zip(data, pdf_names)
    ]
    
    print(f"找到 {len(all_doc_tasks)} 个文档任务。")
    print(f"启动嵌套并行处理：")
    print(f" - {MAX_DOCUMENT_PROCESSES} 个文档将同时处理。")
    print(f" - 每个文档内部，{MAX_PAGE_WORKERS} 个页面将同时处理。")
    
    # 使用多进程池并行处理所有文档
    with multiprocessing.Pool(processes=MAX_DOCUMENT_PROCESSES) as pool:
        results = []
        # 使用tqdm显示文档级别的处理进度
        progress_bar = tqdm(
            pool.imap_unordered(process_single_document, all_doc_tasks),
            total=len(all_doc_tasks),
            desc="Processing Documents"
        )
        for result in progress_bar:
            results.append(result)
            # progress_bar.set_postfix_str(f"Last completed: {result.split(':')[-1].strip()}")

    print("\n所有文档处理完毕。")
    success_count = sum(1 for r in results if r.startswith("Success") or r.startswith("Skipped"))
    error_count = len(results) - success_count
    print(f"摘要: {success_count} 个成功/跳过, {error_count} 个失败。")