# conda activate /nas-alinlp/zhoubb/envs/internvl35
# lmdeploy 0.10.1 torch 2.8.0+cu128 transformers 4.57.0
# lmdeploy serve api_server /nas-alinlp/zhoubb/download_models/InternVL3_5-38B --model-name internvl35_38b --server-port 8000 --tp 8

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
ports = [8000]
MODEL_NAME = "internvl35_38b"
INPUT_FILE = "MPDocBench.json"
OUTPUT_PATH = f"./markdown/{MODEL_NAME}_md"

# --- 并行配置 ---
# 1. 文档间并行：同时处理多少个不同的文档（使用多进程）
MAX_DOCUMENT_PROCESSES = 2
# 2. 文档内并行：处理单个文档时，同时处理多少个页面（使用多线程）
MAX_PAGE_WORKERS = 8

# --- 固定的模型指令 ---
QUERY_PROMPT = r'''You are an AI assistant specialized in converting document images to structured Markdown. Please follow these instructions for the conversion:

            1. Text Processing:
            - Accurately recognize all text from the image.
            - Convert recognized text into Markdown, maintaining the original structure (headings, paragraphs, lists, etc.).

            2. Mathematical Formula Processing:
            - Convert all mathematical formulas to LaTeX format.
            - Enclose inline formulas with `\(...\)`.
            - Enclose block formulas with `\[...\]`.

            3. Table Processing:
            - Convert tables into HTML format, wrapped within `<table>` and `</table>` tags.

            4. Figure Handling:
            - Detect figures and images that are part of the main document content.
            - Do not detect or output images located in the page's header or footer.
            - For each detected content figure, determine its bounding box coordinates, normalized to a 1000x1000 grid.
            - Represent each figure using this specific Markdown syntax: `![](x1_y1_x2_y2.jpg)`.

            5. Output Format:
            - Ensure the final Markdown document has a clear structure.
            - Place the generated figure tags (`![](...).jpg`) in the document flow where the original figures were located.

            Your task is to accurately convert the content of the document image into a single Markdown string, without adding any extra explanations or comments.
            '''

def clean_markdown(markdown_text: str) -> str:
    t = (markdown_text or "").strip()
    if t.startswith("```markdown"):
        t = t[len("```markdown"):].strip()
    t = t.strip()
    if t.endswith("```"):
        t = t[:-len("```")].strip()
    return t

_SINGLE_FIG_RE = re.compile(r'!\[\]\((\d+_\d+_\d+_\d+)\.jpg\)')

def add_pageidx_to_singlepage_fig_tags(md: str, pageidx_1based: int) -> str:
    def _rep(m):
        coords = m.group(1)
        return f"![](page{pageidx_1based}_{coords}.jpg)"
    return _SINGLE_FIG_RE.sub(_rep, md)

def process_page_worker(args):
    """
    [内层并行] 处理单个页面的工作函数，供线程池调用。
    返回 (页面索引, 处理后的Markdown内容)。
    """
    image_path, page_idx, model_name, query = args
    openai_api_key = "EMPTY"
    port = random.choice(ports)
    openai_api_base = f"http://localhost:{port}/v1"

    try:
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
                    {"type": "image_url", "image_url": {"url": base64_qwen}},
                    {"type": "text", "text": query},
                ]},
            ],
            max_tokens=16384,
            temperature=0.0,
        )
        response = chat_response.choices[0].message.content
        response = clean_markdown(response)
        response = add_pageidx_to_singlepage_fig_tags(response, page_idx + 1)
        return (page_idx, response)
        
    except Exception as e:
        error_message = f"Error processing page {page_idx} ({os.path.basename(image_path)}): {e}"
        print(error_message, file=sys.stderr)
        return (page_idx, f"<!-- Page {page_idx+1} failed to process. Error: {e} -->")

def process_single_document(args):
    """
    [外层并行] 处理单个文档的函数，由多进程池调用。
    内部使用线程池并行处理该文档的所有页面。
    """
    image_paths, save_dir, pdf_name, model_name = args
    output_filepath = os.path.join(save_dir, pdf_name + ".md")

    if os.path.exists(output_filepath):
        return f"Skipped (already exists): {pdf_name}"

    try:
        page_tasks = [(path, idx, model_name, QUERY_PROMPT) for idx, path in enumerate(image_paths)]
        page_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PAGE_WORKERS) as executor:
            # 提交所有页面任务到线程池
            future_to_page = {executor.submit(process_page_worker, task): task for task in page_tasks}
            
            # 收集结果
            for future in concurrent.futures.as_completed(future_to_page):
                page_results.append(future.result())

        # 根据页面索引对结果进行排序，确保文档顺序正确
        page_results.sort(key=lambda x: x[0])
        
        # 拼接所有页面的内容
        full_md_content = "\n\n".join([content for idx, content in page_results])
        
        # 写入最终文件
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(full_md_content)

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
    print(f" - {MAX_DOCUMENT_PROCESSES} 个文档将同时处理 (进程池)。")
    print(f" - 每个文档内部，{MAX_PAGE_WORKERS} 个页面将同时处理 (线程池)。")
    
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
            

    print("\n所有文档处理完毕。")
    # 打印处理结果摘要
    success_count = sum(1 for r in results if r.startswith("Success") or r.startswith("Skipped"))
    error_count = len(results) - success_count
    print(f"摘要: {success_count} 个成功/跳过, {error_count} 个失败。")
