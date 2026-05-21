# source /nas-mmu/zhoubb/envs/ppocr_vl/bin/activate
# vllm 0.15.0rc2.dev66+g31b25f651 torch 2.9.1+cu129 transformers 4.57.6
# 项目存储在./tools/idp_tools/DeepSeek-OCR2-main/DeepSeek-OCR2-master/DeepSeek-OCR2-vllm
import os
import io
import re
import json
import fitz
from tqdm import tqdm
import torch
import sys
from concurrent.futures import ThreadPoolExecutor
 
os.environ['VLLM_USE_V1'] = '0'

# 添加模型代码路径
sys.path.insert(0, './tools/idp_tools/DeepSeek-OCR-2-main/DeepSeek-OCR2-master/DeepSeek-OCR2-vllm')
from deepseek_ocr2.config import MODEL_PATH, PROMPT, SKIP_REPEAT, MAX_CONCURRENCY, NUM_WORKERS, CROP_MODE

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from vllm import LLM, SamplingParams

INPUT_file = "MPDocBench.json"  # 下载的json文件
OUTPUT_PATH = "./markdown/deepseek_ocr2_md"

llm = LLM(
    model=MODEL_PATH,
    trust_remote_code=True, 
    max_model_len=8192,
    swap_space=0,
    max_num_seqs=MAX_CONCURRENCY,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
)


sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    skip_special_tokens=False,
)

def re_match(text):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)


    mathes_image = []
    mathes_other = []
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            mathes_image.append(a_match)
        else:
            mathes_other.append(a_match)
    return matches, mathes_image, mathes_other


def extract_coordinates_and_label(ref_text, image_width, image_height):
    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])[0]
        if len(cor_list) != 4:
            return None
        x1, y1, x2, y2 = cor_list

        cor_list = [x1, y1, x2, y2]
    except Exception as e:
        print(e)
        return None

    return (label_type, cor_list)


def process_single_image(prompt, image):
    """single image"""
    prompt_in = prompt
    cache_item = {
        "prompt": prompt_in,
        "multi_modal_data": {"image": image},
    }
    return cache_item

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

if __name__ == "__main__":
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    raw_data = get_pdf_images(INPUT_file)
    data     = [[img_path for (index, img_path, json_path) in v] for k, v in raw_data.items()]
    pdf_names = [k for k in raw_data.keys()]


    prompt = PROMPT

    for pdf_name, img_list in zip(pdf_names, data):
        md_path = os.path.join(OUTPUT_PATH, pdf_name + '.md')
        if os.path.exists(md_path):
            continue

        images = [Image.open(img_path) for img_path in img_list]
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:  
            batch_inputs = list(tqdm(
                executor.map(process_single_image, [prompt] * len(images), images),
                total=len(images),
                desc="Pre-processed images"
            ))

            outputs_list = llm.generate(
                batch_inputs,
                sampling_params=sampling_params
            )

            contents = ''
            draw_images = []
            jdx = 1
            for output, img in zip(outputs_list, images):
                width, height = img.size
                content = output.outputs[0].text
                # print(content)

                matches_ref, matches_images, mathes_other = re_match(content)

                for idx, a_match_image in enumerate(matches_images):
                    result = extract_coordinates_and_label(a_match_image, width, height)
                    if result is None:
                        bbox = str(jdx)
                    else:
                        label_type, points_list = result
                        [x1, y1, x2, y2] = points_list
                        bbox = str(x1) + '_' + str(y1) + '_' + str(x2) + '_' + str(y2)
                    content = content.replace(a_match_image[0], f'![](page' + str(jdx) + '_' + bbox + '.jpg)\n')

                for idx, a_match_other in enumerate(mathes_other):
                    content = content.replace(a_match_other[0], '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:').replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n')
                contents += content + "\n\n"
                jdx += 1
            
            if contents[-2:] == "\n\n":
                contents = contents[:-2]

            with open(md_path, 'w', encoding='utf-8') as afile:
                afile.write(contents)



