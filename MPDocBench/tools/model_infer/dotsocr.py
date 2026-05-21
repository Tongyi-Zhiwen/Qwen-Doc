# 部署使用这个 conda activate /nas-mmu/zhoubb/envs/chandra 客户端推理使用这个 source /nas-mmu/zhoubb/envs/ppocr_vl/bin/activate
# vllm 0.17.1 torch 2.10.0+cu12.8 transformers 4.57.6,
# 我们修改了项目的原始代码, ./tools/idp_tools/dots.ocr-master, 如果哪里报错说缺少什么package,直接pip install那个package就行了
# CUDA_VISIBLE_DEVICES=0 vllm serve --model /nas-mmu/zhoubb/download_models/DOTSOCR --served-model-name dotsocr --async-scheduling --trust-remote-code --chat-template-content-format string --tensor-parallel-size 1 --port 8118 --gpu-memory-utilization 0.9 & \
# CUDA_VISIBLE_DEVICES=1 vllm serve --model /nas-mmu/zhoubb/download_models/DOTSOCR --served-model-name dotsocr --async-scheduling --trust-remote-code --chat-template-content-format string --tensor-parallel-size 1 --port 8119 --gpu-memory-utilization 0.9 & \
# CUDA_VISIBLE_DEVICES=2 vllm serve --model /nas-mmu/zhoubb/download_models/DOTSOCR --served-model-name dotsocr --async-scheduling --trust-remote-code --chat-template-content-format string --tensor-parallel-size 1 --port 8120 --gpu-memory-utilization 0.9 & \
# CUDA_VISIBLE_DEVICES=3 vllm serve --model /nas-mmu/zhoubb/download_models/DOTSOCR --served-model-name dotsocr --async-scheduling --trust-remote-code --chat-template-content-format string --tensor-parallel-size 1 --port 8121 --gpu-memory-utilization 0.9 & \
# CUDA_VISIBLE_DEVICES=4 vllm serve --model /nas-mmu/zhoubb/download_models/DOTSOCR --served-model-name dotsocr --async-scheduling --trust-remote-code --chat-template-content-format string --tensor-parallel-size 1 --port 8122 --gpu-memory-utilization 0.9 & \
# CUDA_VISIBLE_DEVICES=5 vllm serve --model /nas-mmu/zhoubb/download_models/DOTSOCR --served-model-name dotsocr --async-scheduling --trust-remote-code --chat-template-content-format string --tensor-parallel-size 1 --port 8123 --gpu-memory-utilization 0.9 & \
# CUDA_VISIBLE_DEVICES=6 vllm serve --model /nas-mmu/zhoubb/download_models/DOTSOCR --served-model-name dotsocr --async-scheduling --trust-remote-code --chat-template-content-format string --tensor-parallel-size 1 --port 8124 --gpu-memory-utilization 0.9 & \
# CUDA_VISIBLE_DEVICES=7 vllm serve --model /nas-mmu/zhoubb/download_models/DOTSOCR --served-model-name dotsocr --async-scheduling --trust-remote-code --chat-template-content-format string --tensor-parallel-size 1 --port 8125 --gpu-memory-utilization 0.9 &

import os
import json
import sys
from tqdm import tqdm
import random
from multiprocessing.pool import ThreadPool, Pool
import multiprocessing
import argparse

# 添加模型代码路径,也可以在这个项目中使用pip install -e .安装成包的形式
sys.path.insert(0, "./tools/idp_tools/dots.ocr-master")

from dots_ocr.model.inference import inference_with_vllm
from dots_ocr.utils.consts import image_extensions, MIN_PIXELS, MAX_PIXELS
from dots_ocr.utils.image_utils import get_image_by_fitz_doc, fetch_image, smart_resize
from dots_ocr.utils.doc_utils import fitz_doc_to_image, load_images_from_pdf
from dots_ocr.utils.prompts import dict_promptmode_to_prompt
from dots_ocr.utils.layout_utils import post_process_output, draw_layout_on_image, pre_process_bboxes
from dots_ocr.utils.format_transformer import layoutjson2md


# ==================== 配置参数 ====================
INPUT_FILE = "MPDocBench.json"
MODEL_NAME = "dotsocr"  # 模型名称
OUTPUT_PATH = f"./markdown/{MODEL_NAME}"
PROTOCOL = "http"  # 协议
IP = "localhost"  # IP地址
PORTS = list(range(8118, 8118+8))  # 端口
TEMPERATURE = 0.0  # 温度
TOP_P = 1.0  # top_p
MAX_COMPLETION_TOKENS = 16384  # 最大生成token数
NUM_PROCESSES = 8  # 并行进程数
NUM_THREAD_PER_PDF = 8  # 每个PDF内部的线程数
DPI = 144  # PDF转图片的DPI
PROMPT_MODE = "prompt_layout_all_en"  # prompt模式
CONFIG_MIN_PIXELS = None
CONFIG_MAX_PIXELS = None
USE_HF = False  # 是否使用HuggingFace模型
FITZ_PREPROCESS = False  # 是否使用fitz预处理
# =================================================


class DotsOCRParser:
    def __init__(self, 
            protocol='http',
            ip='localhost',
            port=8000,
            model_name='model',
            temperature=0.1,
            top_p=1.0,
            max_completion_tokens=16384,
            num_thread=64,
            dpi = 200, 
            output_dir="./output", 
            min_pixels=None,
            max_pixels=None,
            use_hf=False,
        ):
        self.dpi = dpi

        # default args for vllm server
        self.protocol = protocol
        self.ip = ip
        self.port = port
        self.model_name = model_name
        # default args for inference
        self.temperature = temperature
        self.top_p = top_p
        self.max_completion_tokens = max_completion_tokens
        self.num_thread = num_thread
        self.output_dir = output_dir
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        self.use_hf = use_hf
        if self.use_hf:
            self._load_hf_model()
            # print(f"use hf model, num_thread will be set to 1")
        else:
            pass
            # print(f"use vllm model, num_thread will be set to {self.num_thread}")
        assert self.min_pixels is None or self.min_pixels >= MIN_PIXELS
        assert self.max_pixels is None or self.max_pixels <= MAX_PIXELS

    def _load_hf_model(self):
        pass

    def _inference_with_hf(self, image, prompt):
        pass

    def _inference_with_vllm(self, image, prompt):
        response = inference_with_vllm(
            image,
            prompt, 
            model_name=self.model_name,
            protocol=self.protocol,
            ip=self.ip,
            port=self.port,
            temperature=self.temperature,
            top_p=self.top_p,
            max_completion_tokens=self.max_completion_tokens,
        )
        return response

    def get_prompt(self, prompt_mode, bbox=None, origin_image=None, image=None, min_pixels=None, max_pixels=None):
        prompt = dict_promptmode_to_prompt[prompt_mode]
        if prompt_mode == 'prompt_grounding_ocr':
            assert bbox is not None
            bboxes = [bbox]
            bbox = pre_process_bboxes(origin_image, bboxes, input_width=image.width, input_height=image.height, min_pixels=min_pixels, max_pixels=max_pixels)[0]
            prompt = prompt + str(bbox)
        # print(prompt)
        return prompt

    def _parse_single_image(
        self, 
        origin_image, 
        prompt_mode, 
        save_dir, 
        save_name, 
        source="image", 
        page_idx=0, 
        bbox=None,
        fitz_preprocess=False,
        ):
        min_pixels, max_pixels = self.min_pixels, self.max_pixels
        if prompt_mode == "prompt_grounding_ocr":
            min_pixels = min_pixels or MIN_PIXELS  # preprocess image to the final input
            max_pixels = max_pixels or MAX_PIXELS
        if min_pixels is not None: assert min_pixels >= MIN_PIXELS, f"min_pixels should >= {MIN_PIXELS}"
        if max_pixels is not None: assert max_pixels <= MAX_PIXELS, f"max_pixels should <= {MAX_PIXELS}"

        if source == 'image' and fitz_preprocess:
            image = get_image_by_fitz_doc(origin_image, target_dpi=self.dpi)
            image = fetch_image(image, min_pixels=min_pixels, max_pixels=max_pixels)
        else:
            image = fetch_image(origin_image, min_pixels=min_pixels, max_pixels=max_pixels)
        input_height, input_width = smart_resize(image.height, image.width)
        prompt = self.get_prompt(prompt_mode, bbox, origin_image, image, min_pixels=min_pixels, max_pixels=max_pixels)
        if self.use_hf:
            response = self._inference_with_hf(image, prompt)
        else:
            response = self._inference_with_vllm(image, prompt)
            
        result = {'page_no': page_idx,
            "input_height": input_height,
            "input_width": input_width
        }
        if source == 'pdf':
            save_name = f"{save_name}_page_{page_idx}"
        if prompt_mode in ['prompt_layout_all_en', 'prompt_layout_only_en', 'prompt_grounding_ocr']:
            cells, filtered = post_process_output(
                response, 
                prompt_mode, 
                origin_image, 
                image,
                min_pixels=min_pixels, 
                max_pixels=max_pixels,
                )
            if filtered and prompt_mode != 'prompt_layout_only_en':  # model output json failed, use filtered process
                json_file_path = os.path.join(save_dir, f"{save_name}.json")
                with open(json_file_path, 'w', encoding="utf-8") as w:
                    json.dump(response, w, ensure_ascii=False)

                image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
                origin_image.save(image_layout_path)
                result.update({
                    'layout_info_path': json_file_path,
                    'layout_image_path': image_layout_path,
                })

                md_file_path = os.path.join(save_dir, f"{save_name}.md")
                with open(md_file_path, "w", encoding="utf-8") as md_file:
                    md_file.write(cells)
                result.update({
                    'md_content_path': md_file_path
                })
                result.update({
                    'filtered': True
                })
            else:
                try:
                    image_with_layout = draw_layout_on_image(origin_image, cells)
                except Exception as e:
                    print(f"Error drawing layout on image: {e}")
                    image_with_layout = origin_image

                json_file_path = os.path.join(save_dir, f"{save_name}.json")
                with open(json_file_path, 'w', encoding="utf-8") as w:
                    json.dump(cells, w, ensure_ascii=False)

                image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
                image_with_layout.save(image_layout_path)
                result.update({
                    'layout_info_path': json_file_path,
                    'layout_image_path': image_layout_path,
                })
                if prompt_mode != "prompt_layout_only_en":  # no text md when detection only
                    md_content = layoutjson2md(origin_image, cells, text_key='text', page_id = page_idx+1)
                    md_content_no_hf = layoutjson2md(origin_image, cells, text_key='text', no_page_hf=True, page_id = page_idx+1) # used for clean output or metric of omnidocbench、olmbench 
                    md_file_path = os.path.join(save_dir, f"{save_name}.md")
                    with open(md_file_path, "w", encoding="utf-8") as md_file:
                        md_file.write(md_content)
                    md_nohf_file_path = os.path.join(save_dir, f"{save_name}_nohf.md")
                    with open(md_nohf_file_path, "w", encoding="utf-8") as md_file:
                        md_file.write(md_content_no_hf)
                    result.update({
                        'md_content_path': md_file_path,
                        'md_content_nohf_path': md_nohf_file_path,
                    })
        else:
            image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
            origin_image.save(image_layout_path)
            result.update({
                'layout_image_path': image_layout_path,
            })

            md_content = response
            md_file_path = os.path.join(save_dir, f"{save_name}.md")
            with open(md_file_path, "w", encoding="utf-8") as md_file:
                md_file.write(md_content)
            result.update({
                'md_content_path': md_file_path,
            })

        return result
    

    def parse_image_list(self, image_paths, pdf_name, prompt_mode, save_dir):
        """Parse a list of images (PDF pages) in order"""
        print(f"Processing {pdf_name} with {len(image_paths)} pages")
        
        tasks = [
            {
                "origin_image": fetch_image(image_path),
                "prompt_mode": prompt_mode,
                "save_dir": save_dir,
                "save_name": pdf_name,
                "source": "pdf",
                "page_idx": i,
            } for i, image_path in enumerate(image_paths)
        ]

        def _execute_task(task_args):
            return self._parse_single_image(**task_args)

        if self.use_hf:
            num_thread = 1
        else:
            num_thread = min(len(image_paths), self.num_thread)

        results = []
        with ThreadPool(num_thread) as pool:
            for result in pool.imap(_execute_task, tasks):
                results.append(result)

        results.sort(key=lambda x: x["page_no"])
        for i in range(len(results)):
            results[i]['file_path'] = f"{pdf_name}_page_{results[i]['page_no']}"

        merged_md_content = []
        for result in results:
            md_path = result.get('md_content_path')
            if md_path:
                try:
                    with open(md_path, 'r', encoding='utf-8') as f:
                        merged_md_content.append(f.read().strip())
                except Exception as e:
                    print(f"Error reading {md_path}: {e}")
            else:
                print(f"No markdown found for {result['file_path']}")

        if merged_md_content:
            merged_md_path = os.path.join(save_dir, f"{pdf_name}.md")
            with open(merged_md_path, 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(merged_md_content).strip())
            print(f"Merged markdown saved to: {merged_md_path}")
        
        merged_nohf_md_content = []
        for result in results:
            md_path = result.get('md_content_nohf_path')
            if md_path:
                try:
                    with open(md_path, 'r', encoding='utf-8') as f:
                        merged_nohf_md_content.append(f.read().strip())
                except Exception as e:
                    print(f"Error reading {md_path}: {e}")
            else:
                print(f"No markdown found for {result['file_path']}")
        if merged_nohf_md_content:
            merged_nohf_md_path = os.path.join(save_dir, f"{pdf_name}_nohf.md")
            with open(merged_nohf_md_path, 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(merged_nohf_md_content).strip())
            print(f"Merged markdown saved to: {merged_nohf_md_path}")

        return results


def process_single_pdf_task(task_args):
    """Process a single PDF task (wrapper for multiprocessing)"""
    image_paths, output_path, pdf_name, config = task_args
    if os.path.exists(os.path.join(output_path, pdf_name, f"{pdf_name}.md")) or os.path.exists(os.path.join(output_path, pdf_name, f"{pdf_name}_nohf.md")):
        return {"pdf_name": pdf_name, "status": "skipped", "num_pages": 0}
    
    try:
        # Create parser instance for this process
        parser = DotsOCRParser(
            protocol=config['protocol'],
            ip=config['ip'],
            port=random.choice(config['port']),
            model_name=config['model_name'],
            temperature=config['temperature'],
            top_p=config['top_p'],
            max_completion_tokens=config['max_completion_tokens'],
            num_thread=config['num_thread_per_pdf'],
            dpi=config['dpi'],
            output_dir=output_path,
            min_pixels=config['min_pixels'],
            max_pixels=config['max_pixels'],
            use_hf=config['use_hf'],
        )
        
        # Create save directory
        save_dir = os.path.join(output_path, pdf_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # Parse image list
        results = parser.parse_image_list(
            image_paths=image_paths,
            pdf_name=pdf_name,
            prompt_mode=config['prompt_mode'],
            save_dir=save_dir
        )
        
        # Save results
        with open(os.path.join(output_path, pdf_name, f"{pdf_name}.jsonl"), 'w', encoding="utf-8") as w:
            for result in results:
                w.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        print(f"完成处理 {pdf_name}")
        return {"pdf_name": pdf_name, "status": "success", "num_pages": len(results)}
        
    except Exception as e:
        print(f"处理 {pdf_name} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"pdf_name": pdf_name, "status": "error", "error": str(e)}


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

    print(f"加载输入文件: {INPUT_FILE}")
    raw_data = get_pdf_images(INPUT_FILE)

    # Parse data structure: {pdf_name: [(index, img_path, json_path), ...]}
    data = [[img_path for (index, img_path, json_path) in v] for k, v in raw_data.items()]
    pdf_names = [k for k in raw_data.keys()]
    
    # Prepare configuration
    config = {
        'protocol': PROTOCOL,
        'ip': IP,
        'port': PORTS,
        'model_name': MODEL_NAME,
        'temperature': TEMPERATURE,
        'top_p': TOP_P,
        'max_completion_tokens': MAX_COMPLETION_TOKENS,
        'num_thread_per_pdf': NUM_THREAD_PER_PDF,
        'dpi': DPI,
        'min_pixels': CONFIG_MIN_PIXELS,
        'max_pixels': CONFIG_MAX_PIXELS,
        'use_hf': USE_HF,
        'prompt_mode': PROMPT_MODE,
    }
    
    # Prepare task list
    remaining_data = [
        (image_paths, OUTPUT_PATH, pdf_name, config)
        for image_paths, pdf_name in zip(data, pdf_names)
    ]
    
    print(f"处理 {len(remaining_data)} 个PDF任务...")
    print(f"使用 {NUM_PROCESSES} 个进程并行处理")
    print(f"每个PDF内部使用 {NUM_THREAD_PER_PDF} 个线程")
    
    # Process with multiprocessing
    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_single_pdf_task, remaining_data), 
            total=len(remaining_data), 
            desc="处理进度"
        ))
    
    # Summary
    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = sum(1 for r in results if r["status"] == "error")
    
    print(f"\n处理完成!")
    print(f"成功: {success_count}/{len(results)}")
    print(f"失败: {error_count}/{len(results)}")
    
if __name__ == "__main__":
    main()
