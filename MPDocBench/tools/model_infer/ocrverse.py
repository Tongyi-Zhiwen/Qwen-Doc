# 推理和部署使用同一个环境 conda activate /nas-mmu/zhoubb/envs/glm_ocr
# vllm 0.18.0 torch 2.10.0+cu128 transformers 5.3.0
# CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /nas-mmu/zhoubb/download_models/OCRVerse --max-model-len 32768 --trust-remote-code --served-model-name ocrverse --tensor-parallel-size 4 --port 8118 &
# CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve /nas-mmu/zhoubb/download_models/OCRVerse --max-model-len 32768 --trust-remote-code --served-model-name ocrverse --tensor-parallel-size 4 --port 8119

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
from functools import partial

ports = [8118, 8119]
INPUT_FILE =  "MPDocBench.json"
MODELNAME = "ocrverse"
OUTPUTPATH = f"./markdown/{MODELNAME}_md"

# 新增：处理单张图片的函数，供线程池调用
def process_single_page(imagepath, modelname):
    query = "Extract the main content from the document in the image, keeping the original structure. Convert all formulas to LaTeX and all tables to HTML."
    if imagepath.endswith('.png'):
        imgformat = "png"
    elif imagepath.endswith(('.jpg', '.jpeg')):
        imgformat = "jpeg"
    elif imagepath.endswith('.webp'):
        imgformat = "webp"
    else: 
        print(f"Not support image format {imagepath.split('.')[-1]}")
        return ""

    with open(imagepath, "rb") as f:
        encodedimage = base64.b64encode(f.read())
    encodedimagetext = encodedimage.decode("utf-8")
    base64qwen = f"data:image/{imgformat};base64,{encodedimagetext}"
    
    port = random.choice(ports)
    client = OpenAI(
        api_key="EMPTY",
        base_url=f"http://localhost:{port}/v1",
    )
    
    # try: 
    chatresponse = client.chat.completions.create(
        model=modelname,
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": base64qwen},
                    },
                    {"type": "text", "text": query},
                ],
            }
        ],
        max_tokens=16384,
        temperature=0.0,
    )
    return chatresponse.choices[0].message.content
    # except Exception as e:
    #     print(f"Error processing page {imagepath}: {e}")
    #     return ""

def calllocalvllm(imagepaths, modelname):
    max_threads = 8
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        process_func = partial(process_single_page, modelname=modelname)
        results = list(executor.map(process_func, imagepaths))
    
    # 过滤掉可能因为报错返回的空字符串，并拼接
    valid_results = [res for res in results if res]
    mdcontent = "\n\n".join(valid_results)
    return mdcontent

def processsingleimage(args):
    imagepaths, savedir, pdfname, modelname = args
    try:
        if os.path.exists(os.path.join(savedir, pdfname + ".md")):
            return f"Skipped: {pdfname}"

        mdcontent = calllocalvllm(imagepaths, modelname)
       
        with open(os.path.join(savedir, pdfname + ".md"), "w") as f:
           f.write(mdcontent)

        return f"Success: {pdfname}"
    except Exception as e:
       print(f"Error processing {pdfname}: {str(e)}")
       return f"Error: {pdfname}"


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
    os.makedirs(OUTPUTPATH, exist_ok=True)

    raw_data = get_pdf_images(INPUT_FILE)
    data     = [[imgpath for (index, imgpath, jsonpath) in v] for k, v in raw_data.items()]
    pdfnames = [k for k in raw_data.keys()]

    remainingdata = [(imagepaths, savedir, pdfname, modelname) 
                     for imagepaths, savedir, pdfname, modelname 
                     in zip(data, [OUTPUTPATH] * len(data), pdfnames, [MODELNAME] * len(data))]
                     
    print(f"处理 {len(remainingdata)} 个文档任务...")
    
    taskfunc = partial(processsingleimage)
    
    with multiprocessing.Pool(processes=8) as pool:
        results = list(tqdm(pool.imap_unordered(taskfunc, remainingdata), total=len(remainingdata), desc="处理进度"))
