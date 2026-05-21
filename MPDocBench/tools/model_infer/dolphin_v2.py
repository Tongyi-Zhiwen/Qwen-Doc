# conda activate /nas-alinlp/zhoubb/envs/zhoubb_swift
# vllm 0.8.5.post1 torch 2.6.0+cu124 transformers 4.51.3
# 项目存储在./tools/idp_tools/Dolphin中, Dolphin如果哪里报错说缺少什么package,直接pip install那个package就行了
# CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /nas-mmu/zhoubb/download_models/Dolphin-v2 --max-model-len 32768 --trust-remote-code --served-model-name Dolphin_v2 --tensor-parallel-size 4 --port 8118 &
# CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve /nas-mmu/zhoubb/download_models/Dolphin-v2 --max-model-len 32768 --trust-remote-code --served-model-name Dolphin_v2 --tensor-parallel-size 4 --port 8119

"""
Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: MIT
"""

import os
import base64
import random
import json
import sys
import multiprocessing
from functools import partial
from io import BytesIO
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor  # <-- 新增导入

import torch
from PIL import Image
from openai import OpenAI

# 添加模型代码路径,也可以在这个项目中使用pip install -e .安装成包的形式
sys.path.insert(0, "./tools/idp_tools/Dolphin")
from utils.utils import *


# ==================== 配置参数 ====================
MODEL_NAME = "Dolphin_v2"  # vLLM部署的模型名称
INPUT_FILE = "MPDocBench.json"
OUTPUT_PATH = f"./markdown/{MODEL_NAME}"
PORTS = [8118, 8119]  # vLLM服务端口列表
MAX_BATCH_SIZE = 64  # 元素处理的最大批次大小
NUM_PROCESSES = 16  # 并行进程数
POST_PROCESS = False  # 是否应用后处理
# =================================================


class DOLPHIN:
    def __init__(self, model_name, ports):
        """Initialize the vLLM model with OpenAI API
        
        Args:
            model_name: Name of the model deployed on vLLM
            ports: List of available ports for vLLM service
        """
        self.model_name = model_name
        self.ports = ports
        self.openai_api_key = "EMPTY"
        
    def _get_client(self):
        """Get OpenAI client with random port selection"""
        port = random.choice(self.ports)
        openai_api_base = f"http://localhost:{port}/v1"
        return OpenAI(
            api_key=self.openai_api_key,
            base_url=openai_api_base,
        )

    def _image_to_base64(self, image):
        """Convert PIL Image to base64 string"""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        encoded_image = base64.b64encode(buffered.getvalue())
        encoded_image_text = encoded_image.decode("utf-8")
        return f"data:image/png;base64,{encoded_image_text}"

    def chat(self, prompt, image):       
        # Check if we're dealing with a batch
        is_batch = isinstance(image, list)
        
        if not is_batch:
            images = [image]
            prompts = [prompt]
        else:
            images = image
            prompts = prompt if isinstance(prompt, list) else [prompt] * len(images)
        
        assert len(images) == len(prompts), "Number of images and prompts must match"
        
        client = self._get_client()
        results = []
        
        # Process each image
        for img, question in zip(images, prompts):
            # Resize image if needed
            processed_img = resize_img(img)
            
            # Convert to base64
            base64_image = self._image_to_base64(processed_img)
            
            try:
                chat_response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": base64_image},
                                    "min_pixels": 784,
                                    "max_pixels": 2560000
                                },
                                {"type": "text", "text": question},
                            ],
                        }
                    ],
                    max_tokens=16384,
                    temperature=0.0,
                )
                response = chat_response.choices[0].message.content
                results.append(response)
            except Exception as e:
                print(f"Error calling API: {str(e)}")
                results.append("")
        
        # Return single result for single image input
        if not is_batch:
            return results[0]
        return results


def process_pdf_pages(image_paths, save_dir, pdf_name, model_name, ports, max_batch_size=4, post_process=False):
    # Initialize model for this process
    model = DOLPHIN(model_name, ports)
    save_dir = os.path.join(save_dir, pdf_name)
    setup_output_dirs(save_dir)

    
    all_results = []
    
    # Process each page in order
    for page_idx, image_path in enumerate(image_paths):
        try:
            # Load image
            pil_image = Image.open(image_path).convert("RGB")
            
            # Generate output name for this page
            page_name = f"{pdf_name}_page_{page_idx + 1}"
            
            # Process this page (don't save individual page results)
            _, recognition_results = process_single_image(
                pil_image, model, save_dir, page_name, max_batch_size, 
                save_individual=True, post_process=post_process
            )
            
            # Add page information to results
            page_results = {
                "page_number": page_idx + 1,
                "image_path": image_path,
                "elements": recognition_results
            }
            all_results.append(page_results)
            
        except Exception as e:
            print(f"Error processing {pdf_name} page {page_idx + 1} ({image_path}): {str(e)}")
            all_results.append({
                "page_number": page_idx + 1,
                "image_path": image_path,
                "elements": [],
                "error": str(e)
            })
            continue
    
    # Save combined results
    combined_json_path = save_combined_pdf_results(
        all_results, pdf_name + ".pdf", save_dir, post_process=post_process
    )
    
    return combined_json_path, all_results


def process_single_task(task_args):
    image_paths, save_dir, pdf_name, model_name, ports, max_batch_size, post_process = task_args

    if os.path.exists(f"{save_dir}/{pdf_name}/markdown/{pdf_name}.md") and os.path.exists(f"{save_dir}/{pdf_name}/recognition_json/{pdf_name}.json"):
        return {"pdf_name": pdf_name, "status": "success", "json_path": ""}
    
    try:
        print(f"开始处理 {pdf_name}，共 {len(image_paths)} 页")
        json_path, all_results = process_pdf_pages(
            image_paths=image_paths,
            save_dir=save_dir,
            pdf_name=pdf_name,
            model_name=model_name,
            ports=ports,
            max_batch_size=max_batch_size,
            post_process=post_process
        )
        return {"pdf_name": pdf_name, "status": "success", "json_path": json_path}
    except Exception as e:
        print(f"处理 {pdf_name} 时出错: {str(e)}")
        return {"pdf_name": pdf_name, "status": "error", "error": str(e)}



def process_single_image(image, model, save_dir, image_name, max_batch_size=None, save_individual=True, post_process=False):
    # Stage 1: Page-level layout and reading order parsing
    layout_output = model.chat("Parse the reading order of this document.", image)

    # Stage 2: Element-level content parsing
    recognition_results = process_elements(layout_output, image, model, max_batch_size, save_dir, image_name)

    # Save outputs only if requested (skip for PDF pages)
    json_path = None
    if save_individual:
        json_path = save_outputs(recognition_results, image, image_name, save_dir, post_process=post_process)

    return json_path, recognition_results


def process_elements(layout_results, image, model, max_batch_size, save_dir=None, image_name=None):
    """
    Parse all document elements with parallel decoding.
    【此函数已被修改为使用线程池并行处理】
    """
    layout_results_list = parse_layout_string(layout_results)
    if not layout_results_list or not (layout_results.startswith("[") and layout_results.endswith("]")):
        layout_results_list = [([0, 0, *image.size], 'distorted_page', [])]
    elif len(layout_results_list) > 1 and check_bbox_overlap(layout_results_list, image):
        print("Falling back to distorted_page mode due to high bbox overlap")
        layout_results_list = [([0, 0, *image.size], 'distorted_page', [])]
        
    tab_elements = []      
    equ_elements = []     
    code_elements = []    
    text_elements = []     
    figure_results = []    
    reading_order = 0

    # 1. 收集和分组元素
    for bbox, label, tags in layout_results_list:
        try:
            if label == "distorted_page":
                x1, y1, x2, y2 = 0, 0, *image.size
                pil_crop = image
            else:
                x1, y1, x2, y2 = process_coordinates(bbox, image)
                pil_crop = image.crop((x1, y1, x2, y2))

            if pil_crop.size[0] > 3 and pil_crop.size[1] > 3:
                if label == "fig":
                    figure_filename = save_figure_to_local(pil_crop, save_dir, image_name, reading_order)
                    figure_results.append({
                        "label": label,
                        "text": f"![Figure](figures/{figure_filename})",
                        "figure_path": f"figures/{figure_filename}",
                        "bbox": [x1, y1, x2, y2],
                        "normalized_bbox": list(map(int, bbox)),
                        "reading_order": reading_order,
                        "tags": tags,
                    })
                else:
                    element_info = {
                        "crop": pil_crop,
                        "label": label,
                        "bbox": [x1, y1, x2, y2],
                        "normalized_bbox": list(map(int, bbox)),
                        "reading_order": reading_order,
                        "tags": tags,
                    }
                    if label == "tab":
                        tab_elements.append(element_info)
                    elif label == "equ":
                        equ_elements.append(element_info)
                    elif label == "code":
                        code_elements.append(element_info)
                    else:
                        text_elements.append(element_info)
            reading_order += 1
        except Exception as e:
            print(f"Error processing bbox with label {label}: {str(e)}")
            continue

    recognition_results = figure_results.copy()
    
    # 2. 使用线程池并行处理不同类型的元素
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        
        # 提交表格处理任务
        if tab_elements:
            futures.append(executor.submit(process_element_batch, tab_elements, model, "Parse the table in the image.", max_batch_size))
        
        # 提交公式处理任务
        if equ_elements:
            futures.append(executor.submit(process_element_batch, equ_elements, model, "Read formula in the image.", max_batch_size))
        
        # 提交代码处理任务
        if code_elements:
            futures.append(executor.submit(process_element_batch, code_elements, model, "Read code in the image.", max_batch_size))
        
        # 提交文本处理任务
        if text_elements:
            futures.append(executor.submit(process_element_batch, text_elements, model, "Read text in the image.", max_batch_size))
            
        # 3. 收集并行任务的结果
        for future in futures:
            try:
                batch_results = future.result()
                recognition_results.extend(batch_results)
            except Exception as e:
                print(f"A batch processing task failed in ThreadPool: {e}")

    # 4. 按阅读顺序对所有结果进行排序
    recognition_results.sort(key=lambda x: x.get("reading_order", 0))

    return recognition_results


def process_element_batch(elements, model, prompt, max_batch_size=None):
    """Process elements of the same type in batches"""
    results = []
    
    # Determine batch size
    batch_size = len(elements)
    if max_batch_size is not None and max_batch_size > 0:
        batch_size = min(batch_size, max_batch_size)
    
    # Process in batches
    for i in range(0, len(elements), batch_size):
        batch_elements = elements[i:i+batch_size]
        crops_list = [elem["crop"] for elem in batch_elements]
        
        # Use the same prompt for all elements in the batch
        prompts_list = [prompt] * len(crops_list)
        
        # Batch inference
        batch_results = model.chat(prompts_list, crops_list)
        
        # Add results
        for j, result in enumerate(batch_results):
            elem = batch_elements[j]
            results.append({
                "label": elem["label"],
                "bbox": elem["bbox"],
                "normalized_bbox": elem["normalized_bbox"],
                "text": result.strip(),
                "reading_order": elem["reading_order"],
                "tags": elem["tags"],
            })
    
    return results


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


def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Load input data
    print(f"加载输入文件: {INPUT_FILE}")
    raw_data = get_pdf_images(INPUT_FILE)
    
    data = [[img_path for (index, img_path, json_path) in v] for k, v in raw_data.items()]
    pdf_names = [k for k in raw_data.keys()]
    
    remaining_data = [
        (image_paths, OUTPUT_PATH, pdf_name, MODEL_NAME, PORTS, MAX_BATCH_SIZE, POST_PROCESS)
        for image_paths, pdf_name in zip(data, pdf_names)
    ]
    
    print(f"处理 {len(remaining_data)} 个PDF任务...")
    print(f"使用 {NUM_PROCESSES} 个进程并行处理")
    
    # Process with multiprocessing
    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_single_task, remaining_data), 
            total=len(remaining_data), 
            desc="处理进度"
        ))
    
    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = sum(1 for r in results if r["status"] == "error")
    
    print(f"\n处理完成!")
    print(f"成功: {success_count}/{len(results)}")
    print(f"失败: {error_count}/{len(results)}")
    

if __name__ == '__main__':
    main()

