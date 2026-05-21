# 推理和模型部署在一个命令下 conda activate /nas-mmu/zhoubb/envs/MonkeyOCR_Pro
# magic_pdf 1.1.0 lmdeploy 0.9.2 torch 2.6.0+cu118 transformers 4.51.0 paddlepaddle-gpu 3.0.0 paddlex 3.3.0
# 在部署模型之前，将目录cd到MonkeyOCR_Pro路径下,执行以下命令部署模型
# CUDA_VISIBLE_DEVICES=0 uvicorn api.main:app --port 8000 & CUDA_VISIBLE_DEVICES=1 uvicorn api.main:app --port 8001 &
# CUDA_VISIBLE_DEVICES=2 uvicorn api.main:app --port 8002 & CUDA_VISIBLE_DEVICES=3 uvicorn api.main:app --port 8003 &
# CUDA_VISIBLE_DEVICES=4 uvicorn api.main:app --port 8004 & CUDA_VISIBLE_DEVICES=5 uvicorn api.main:app --port 8005 &
# CUDA_VISIBLE_DEVICES=6 uvicorn api.main:app --port 8006 & CUDA_VISIBLE_DEVICES=7 uvicorn api.main:app --port 8007
# 在部署好模型之后，新开一个bash, cd到MPDocBench路径下，再运行本脚本

import os
import subprocess
import json
import fitz
from tqdm import tqdm
import random
from pathlib import Path
from PIL import Image, ImageOps
from io import BytesIO
import shutil
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

ports = list(range(8000, 8000+8))
threads = min(16, 5 * len(ports))  # 并行线程数，根据 CPU / vllm-server 并发能力调整
INPUT_FILE =  "MPDocBench.json"
OUTPUT_PATH = "./markdown/monkeyocrpro_3b"



def images_bytes_to_pdf_bytes(image_bytes_list, dpi=200):
    pdf_buffer = BytesIO()
    images = []
    for image_bytes in image_bytes_list:
        image = Image.open(BytesIO(image_bytes))
        image = ImageOps.exif_transpose(image) or image
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        images.append(image)
    if not images:
        raise ValueError("图片列表为空")
    images[0].save(
        pdf_buffer,
        format="PDF",
        save_all=True,
        append_images=images[1:],
        resolution=dpi,
    )
    pdf_bytes = pdf_buffer.getvalue()
    pdf_buffer.close()
    return pdf_bytes


def images_to_pdf_fitz(image_bytes_list, dpi=200):
    """
    使用 PyMuPDF(fitz) 创建一个PDF，可以精确控制尺寸。
    这个函数将替代原来的基于Pillow的函数。
    """
    doc = fitz.open()  # 创建一个空的PDF文档

    for image_bytes in image_bytes_list:
        img = Image.open(BytesIO(image_bytes))
        width_px, height_px = img.size
        width_pt = width_px * 72 / dpi
        height_pt = height_px * 72 / dpi
        page = doc.new_page(width=width_pt, height=height_pt)        
        page.insert_image(page.rect, stream=image_bytes)

    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


def run_doc_parser(image_paths, pdf_name, save_path_dir):
    # 初始化
    port = random.choice(ports)
    save_dir = Path(save_path_dir) / pdf_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # 创建 PDF（如果不存在）
    pdf_path = Path("/tmp/dpi200") / f"{pdf_name}.pdf"
    pdf_path.parent.mkdir(exist_ok=True)
    if not pdf_path.exists():
        file_bytes = [open(img, "rb").read() for img in image_paths]
        try:
            pdf_path.write_bytes(images_to_pdf_fitz(file_bytes, dpi=200))
        except Exception as e:
            print(f"Error: {e} in creating PDF {pdf_name}. Trying Pillow-based method...")
            pdf_path.write_bytes(images_bytes_to_pdf_bytes(file_bytes, dpi=200))


    md_name = ""
    for file in os.listdir(save_dir):
        if file.endswith(".md"):
            md_name = os.path.splitext(file)[0]
            break

    if md_name and os.path.exists(save_dir / f"{md_name}.md") and os.path.exists(save_dir / f"{md_name}_middle.json"):
        return (image_paths, True, "skipped")
    
    try:
        with open(pdf_path, 'rb') as f:
            resp = requests.post(
                f"http://localhost:{port}/parse",
                files={'file': (pdf_path.name, f)},
                timeout=3000
            ).json()
        
        # 检查结果
        if not resp.get("success"):
            return (image_paths, False, resp.get("message", "Parse failed"))
        
        # 复制结果文件
        output_dir = Path(resp["output_dir"])
        
        if not output_dir.exists():
            print(f"Output directory not found: {output_dir}")
            return (image_paths, False, f"Output directory not found: {output_dir}")
        
        shutil.copytree(output_dir, save_dir, dirs_exist_ok=True)
        
        return (image_paths, True, "")
        
    except Exception as e:
        print(f"Error: {e}")
        return (image_paths, False, str(e))


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
    raw_data = get_pdf_images(INPUT_FILE)

    # 修正数据提取逻辑，确保 image_paths 是字符串列表
    data = []
    pdf_names = []
    for k, v in raw_data.items():
        image_paths_for_pdf = [item[1] for item in v] # item[1] 是 img_path
        data.append(image_paths_for_pdf)
        pdf_names.append(k)


    print(f"Total pdfs:   {len(data)}")
    print(f"Total images: {sum([len(d) for d in data])}")

    success = []
    failed  = []
    skipped = []

    try:
        with ThreadPoolExecutor(max_workers=threads) as executor:
            future_to_pdf = {
                executor.submit(run_doc_parser, image_paths, pdf_name, OUTPUT_PATH): pdf_name
                for pdf_name, image_paths in zip(pdf_names, data)
            }
            
            progress_bar = tqdm(
                as_completed(future_to_pdf), 
                total=len(data), 
                desc="Processing"
            )

            for future in progress_bar:
                pdf_name = future_to_pdf[future]
                # try:
                img_paths, ok, msg = future.result()
                if msg == "skipped":
                    skipped.append(pdf_name)
                elif ok:
                    success.append(pdf_name)
                else:
                    print(f"File:   {pdf_name}: {msg}")
                    failed.append((pdf_name, msg))
                # except Exception as exc:
                #     print(f"File:   {pdf_name}: {str(exc)}")
                #     failed.append((pdf_name, str(exc)))
                
                progress_bar.set_postfix({
                    "success": len(success),
                    "failed":  len(failed),
                    "skipped": len(skipped),
                })

    finally:
        print("所有pipeline已关闭")

    print(f"\nDone!")
    print(f"Success: {len(success)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Failed:  {len(failed)}")


if __name__ == "__main__":
    main()
