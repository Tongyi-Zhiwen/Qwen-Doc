# 推理和部署使用同一个环境 conda activate /nas-alinlp/zhoubb/envs/minicpm
# torch 2.8.0+cu128 vllm 0.10.2 transformers 4.57.0
# CUDA_VISIBLE_DEVICES=0 vllm serve /nas-mmu/zhoubb/download_models/Qianfan-OCR --trust-remote-code --served-model-name qianfan-ocr --port 8118 &
# CUDA_VISIBLE_DEVICES=1 vllm serve /nas-mmu/zhoubb/download_models/Qianfan-OCR --trust-remote-code --served-model-name qianfan-ocr --port 8119 &
# CUDA_VISIBLE_DEVICES=2 vllm serve /nas-mmu/zhoubb/download_models/Qianfan-OCR --trust-remote-code --served-model-name qianfan-ocr --port 8120 &
# CUDA_VISIBLE_DEVICES=3 vllm serve /nas-mmu/zhoubb/download_models/Qianfan-OCR --trust-remote-code --served-model-name qianfan-ocr --port 8121 &
# CUDA_VISIBLE_DEVICES=4 vllm serve /nas-mmu/zhoubb/download_models/Qianfan-OCR --trust-remote-code --served-model-name qianfan-ocr --port 8122 &
# CUDA_VISIBLE_DEVICES=5 vllm serve /nas-mmu/zhoubb/download_models/Qianfan-OCR --trust-remote-code --served-model-name qianfan-ocr --port 8123 &
# CUDA_VISIBLE_DEVICES=6 vllm serve /nas-mmu/zhoubb/download_models/Qianfan-OCR --trust-remote-code --served-model-name qianfan-ocr --port 8124 &
# CUDA_VISIBLE_DEVICES=7 vllm serve /nas-mmu/zhoubb/download_models/Qianfan-OCR --trust-remote-code --served-model-name qianfan-ocr --port 8125 


import argparse
import json
import concurrent.futures
import re
import os
import sys
from tqdm import tqdm
from PIL import Image
from openai import OpenAI
import random
import base64
from functools import partial
import threading

# ================= 配置区域 =================
PORTS = list(range(8118, 8126))
MODEL_NAME = "qianfan-ocr" 
INPUT_FILE =  "MPDocBench.json"
OUTPUT_PATH = f"./markdown/{MODEL_NAME}_md"
MAX_WORKERS = 64
OUTER_CONCURRENCY = 8

# ===========================================
def normalize_bbox(x1, y1, x2, y2, image_width, image_height):
    """
    根据你的原始逻辑，这里似乎不需要归一化回像素坐标
    原始代码注释掉了归一化计算，直接返回 int。
    如果需要归一化，请取消注释并修改逻辑。
    """
    # 假设输入已经是像素坐标或者不需要转换
    return [int(x1), int(y1), int(x2), int(y2)]

def extract_box_label(text):
    """提取 <box> 和 <label> 内容"""
    pattern = r'<box>(.*?)</box>\s*<label>(.*?)</label>'
    matches = re.findall(pattern, text, re.DOTALL)
    
    results = []
    for bbox, label in matches:
        try:
            if label.strip() == "image":
                coords = convert_coords(bbox.strip())
                if coords: # 确保坐标解析成功
                    results.append([label.strip(), coords])
        except Exception:
            continue
    return results

def convert_coords(coord_str):
    """解析 <COORD_123> 格式"""
    numbers = re.findall(r'<COORD_(\d+)>', coord_str)
    return [int(n) for n in numbers]


import re

def replace_image_placeholders(md_content: str, page_index: int) -> str:
    """
    在给定的 Markdown 文本中，查找并替换特定的图片占位符。

    Args:
        md_content (str): 包含占位符的原始 Markdown 文本。
        page_index (int): 当前页面的索引 (从 0 开始)。

    Returns:
        str: 替换占位符后的新 Markdown 文本。
    """
    pattern = re.compile(r'!\[image\]\(<box>\[\[<COORD_(\d+)>, <COORD_(\d+)>, <COORD_(\d+)>, <COORD_(\d+)>\]\]</box>\)')
    replacer = lambda m: f"![](page{page_index + 1}_{m.group(1)}_{m.group(2)}_{m.group(3)}_{m.group(4)}.jpg)"
    new_md_content = pattern.sub(replacer, md_content)
    
    return new_md_content



def encode_image_to_base64(image_path):
    """将图片转换为 base64 字符串"""
    with open(image_path, "rb") as f:
        encoded_image = base64.b64encode(f.read())
    return encoded_image.decode("utf-8")

def get_image_format(image_path):
    ext = image_path.split('.')[-1].lower()
    if ext == 'png':
        return "png"
    elif ext in ['jpg', 'jpeg']:
        return "jpeg"
    elif ext == 'webp':
        return "webp"
    else:
        return None

def process_single_page(args):
    image_path, page_index, model_name, port = args
    
    query = """Parse this document to Markdown.<think>"""
    
    openai_api_key = "EMPTY"
    openai_api_base = f"http://localhost:{port}/v1"
    
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    
    try:
        # 1. 准备图片数据
        img_format = get_image_format(image_path)
        if not img_format:
            print(f"[Error: Unsupported format for {image_path}]")
            return page_index, f"\n[Error: Unsupported format for {image_path}]\n", ""
        
        base64_image = encode_image_to_base64(image_path)
        data_url = f"data:image/{img_format};base64,{base64_image}"
        
        img = Image.open(image_path)
        width, height = img.size
        
        # 2. 调用 API
        chat_response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": query},
                    ],
                }
            ],
            extra_body={
                "skip_special_tokens": False  # <--- 正确的用法在这里
            },
            max_tokens=16384,
            temperature=0.0,
        )
        
        response = chat_response.choices[0].message.content
        
        md_text = ""
        image_tags = ""
        # print(response)
        # 3. 解析响应
        if "</think>" in response:
            thinking, content = response.split("</think>", 1)
            md_text = content.strip()
            # # 提取图片标注
            # image_results = extract_box_label(thinking)
            # for label, bbox in image_results:
            #     if len(bbox) >= 4:
            #         x1, y1, x2, y2 = normalize_bbox(bbox[0], bbox[1], bbox[2], bbox[3], width, height)
            #         # 生成图片引用标记，稍后合并到主 markdown
            #         image_tags += f"![](page{page_index + 1}_{x1}_{y1}_{x2}_{y2}.jpg)\n\n"
        else:
            md_text = response
        md_text = replace_image_placeholders(md_text, page_index)    
        return page_index, md_text + "\n\n", image_tags
        
    except Exception as e:
        return page_index, f"\n[Error processing page {page_index}: {str(e)}]\n", ""

def process_single_pdf_task(pdf_info):
    image_paths, save_dir, pdf_name, model_name = pdf_info
    output_path = os.path.join(save_dir, f"{pdf_name}.md")

    if os.path.exists(output_path):
        return f"Success: {pdf_name}"
    
    page_args = []
    for idx, img_path in enumerate(image_paths):
        port = random.choice(PORTS)
        page_args.append((img_path, idx, model_name, port))
    
    # 存储结果，key 为 page_index，保证顺序
    results_map = {}
    
    local_workers = min(MAX_WORKERS, len(image_paths))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=local_workers) as executor:
        future_to_idx = {executor.submit(process_single_page, args): args[1] for args in page_args}
        
        for future in concurrent.futures.as_completed(future_to_idx):
            page_idx = future_to_idx[future]
            try:
                _, md_part, img_tags = future.result()
                results_map[page_idx] = (md_part, img_tags)
            except Exception as exc:
                print(f'Page generated an exception: {exc}')
                results_map[page_idx] = (f"\n[Error: Page failed]\n", "")

    # === 合并结果 ===
    # 按页码排序，保证 Markdown 内容顺序正确
    sorted_pages = sorted(results_map.items())
    
    final_md_content = ""
    for _, (md_part, img_tags) in sorted_pages:
        final_md_content += md_part
        if img_tags:
            final_md_content += img_tags # 图片标签通常插在对应位置或末尾，视需求而定
            
    # 清理末尾多余换行
    if final_md_content.endswith("\n\n"):
        final_md_content = final_md_content[:-2]
        
    # 保存文件
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_md_content)
        return f"Success: {pdf_name}"
    except Exception as e:
        return f"Error saving {pdf_name}: {str(e)}"



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
        
    tasks = []
    for pdf_name, pages_info in raw_data.items():
        img_paths = [item[1] for item in pages_info] 
        tasks.append((img_paths, OUTPUT_PATH, pdf_name, MODEL_NAME))

    print(f"共找到 {len(tasks)} 个 PDF 任务")
    print(f"全局最大并发线程数: {MAX_WORKERS}")
    print("开始处理...")
    
    
    INNER_CONCURRENCY_PER_PDF = MAX_WORKERS // OUTER_CONCURRENCY
    
    import multiprocessing
    
    def worker_wrapper(task):
        return process_single_pdf_task(task)

    num_processes = min(multiprocessing.cpu_count(), OUTER_CONCURRENCY)
    
    print(f"启动 {num_processes} 个进程，每个进程内部处理 PDF 的多页并发")
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(worker_wrapper, tasks), total=len(tasks), desc="PDF 处理进度"))
        
    print("所有任务完成。")
    for res in results:
        if res.startswith("Error"):
            print(res)