# conda activate /nas-mmu/zhoubb/envs/chandra
# chandra-ocr 0.2.0  vllm 0.17.1 torch 2.10.0+cu12.8 transformers 4.57.6,
# 安装过程参考原始代码: ./tools/idp_tools/chandra,这个代码和原始项目代码略有不同
# CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve --model /nas-mmu/zhoubb/download_models/chandra-ocr-2 --served-model-name chandra --tensor-parallel-size 4 --trust-remote-code --gpu-memory-utilization 0.9 --max-model-len 20480 --max-num-batched-tokens 20480 --port 8118 &
# CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve --model /nas-mmu/zhoubb/download_models/chandra-ocr-2 --served-model-name chandra --tensor-parallel-size 4 --trust-remote-code --gpu-memory-utilization 0.9 --max-model-len 20480 --max-num-batched-tokens 20480 --port 8119


import os
import subprocess
import json
from tqdm import tqdm
import random
import fitz
from pathlib import Path
from PIL import Image, ImageOps
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed


ports = [8118, 8119] 
threads = 16  # 并行线程数，根据 CPU / vllm-server 并发能力调整
INPUT_FILE = "MPDocBench.json"  # 下载的json文件
OUTPUT_PATH = "./markdown/chandra"


def images_bytes_to_pdf_bytes(image_bytes_list, dpi=144):
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

def images_to_pdf_fitz(image_bytes_list, dpi=144):
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
    port = random.choice(ports)

    save_dir = Path(save_path_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    pdf_demo_dir = Path("/tmp/dpi144")
    pdf_demo_dir.mkdir(exist_ok=True)
    pdf_path = pdf_demo_dir / f"{pdf_name}.pdf"

    if not pdf_path.exists():
        file_bytes = [open(img, "rb").read() for img in image_paths]
        try:
            pdf_path.write_bytes(images_to_pdf_fitz(file_bytes, dpi=144))
        except Exception as e:
            print(f"Error: {e} in creating PDF {pdf_name}. Trying Pillow-based method...")
            pdf_path.write_bytes(images_bytes_to_pdf_bytes(file_bytes, dpi=144))

    d = save_dir /  pdf_name
    if d.exists():
        suffixs = ["_metadata.json", ".html", ".md"]
        ok = all((d / f"{pdf_name}{s}").exists() for s in suffixs)
        if ok:
            return f"skipped (already exists): {pdf_name}"

    try:
        cmd = [
        "chandra",
        pdf_path,
        save_dir,
        "--method", "vllm",
        "--vllm_api_base", f"http://localhost:{port}/v1", "--paginate_output"
        ]
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.decode('utf-8', errors='ignore'))
        print(result.stderr.decode('utf-8', errors='ignore'))
        return (image_paths, True, "")

    except subprocess.CalledProcessError as e:
        print(f"Failed to parse {pdf_path}: {e.stderr.decode('utf-8', errors='ignore')}")
        return (image_paths, False, e.stderr.decode("utf-8", errors="ignore"))


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
                try:
                    img_paths, ok, msg = future.result()
                    if msg == "skipped":
                        skipped.append(pdf_name)
                    elif ok:
                        success.append(pdf_name)
                    else:
                        failed.append((pdf_name, msg))
                except Exception as exc:
                    failed.append((pdf_name, str(exc)))
                
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

    # if failed:
    #     print("\n--- Failed Tasks ---")
    #     for pdf_name, reason in failed:
    #         print(f"File:   {pdf_name}")
    #         print(f"Reason: {str(reason)[:300]}") # 打印更长的错误信息
    #         print("-" * 40)


if __name__ == "__main__":
    main()
