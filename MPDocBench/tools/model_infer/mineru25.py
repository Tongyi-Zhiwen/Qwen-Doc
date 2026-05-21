
# 部署source /nas-mmu/zhoubb/envs/ppocr_vl/bin/activate
# vllm 0.15.0rc2.dev66+g31b25f651 torch 2.9.1+cu129 transformers 4.57.6
# CUDA_VISIBLE_DEVICES=0 vllm serve /nas-alinlp/zhoubb/download_models/MinerU2.5-2509-1.2B --host 127.0.0.1 --port 9001 --logits-processors mineru_vl_utils:MinerULogitsProcessor &
# CUDA_VISIBLE_DEVICES=1 vllm serve /nas-alinlp/zhoubb/download_models/MinerU2.5-2509-1.2B --host 127.0.0.1 --port 9002 --logits-processors mineru_vl_utils:MinerULogitsProcessor &
# CUDA_VISIBLE_DEVICES=2 vllm serve /nas-alinlp/zhoubb/download_models/MinerU2.5-2509-1.2B --host 127.0.0.1 --port 9003 --logits-processors mineru_vl_utils:MinerULogitsProcessor &
# CUDA_VISIBLE_DEVICES=3 vllm serve /nas-alinlp/zhoubb/download_models/MinerU2.5-2509-1.2B --host 127.0.0.1 --port 9004 --logits-processors mineru_vl_utils:MinerULogitsProcessor &
# CUDA_VISIBLE_DEVICES=4 vllm serve /nas-alinlp/zhoubb/download_models/MinerU2.5-2509-1.2B --host 127.0.0.1 --port 9005 --logits-processors mineru_vl_utils:MinerULogitsProcessor &
# CUDA_VISIBLE_DEVICES=5 vllm serve /nas-alinlp/zhoubb/download_models/MinerU2.5-2509-1.2B --host 127.0.0.1 --port 9006 --logits-processors mineru_vl_utils:MinerULogitsProcessor &
# CUDA_VISIBLE_DEVICES=6 vllm serve /nas-alinlp/zhoubb/download_models/MinerU2.5-2509-1.2B --host 127.0.0.1 --port 9007 --logits-processors mineru_vl_utils:MinerULogitsProcessor &
# CUDA_VISIBLE_DEVICES=7 vllm serve /nas-alinlp/zhoubb/download_models/MinerU2.5-2509-1.2B --host 127.0.0.1 --port 9008 --logits-processors mineru_vl_utils:MinerULogitsProcessor

import os
import re
import subprocess
import json
import fitz
from tqdm import tqdm
import random
from pathlib import Path
from PIL import Image, ImageOps
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 配置 ---
ports = [p for p in range(9001, 9001 + 8)]
threads = min(64, 8 * len(ports))
INPUT_FILE = "MPDocBench.json"
OUTPUT_PATH = "./markdown/mineru_2.5"

# ============================================================
# MODIFIED: 将 md 中的图片链接转换为带归一化坐标的格式
#           (尺寸信息从外部 image_paths 获取)
# ============================================================
def fill_image_captions_in_md(md_path, json_path, middle_json_path, output_md_path=None):
    """
    将 md 文件中 ![](...) 的 alt-text 填充为归一化的 bbox 信息。
    bbox 来自 json_path，图片尺寸来自 image_paths。
    """
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    with open(json_path, "r", encoding="utf-8") as f:
        content_list = json.load(f)
    
    with open(middle_json_path, "r", encoding="utf-8") as f:
        middle_content_list = json.load(f)

    # --- 步骤 1: 创建 page_idx -> image_size 的映射 ---
    page_to_size = {}
    # print("正在读取图片尺寸...")
    for page_res in middle_content_list["pdf_info"]:
        page_idx = page_res["page_idx"]
        page_size = page_res.get("page_size")
        if page_idx in page_to_size:
            print(f"[WARN] 已经找到 page_idx {page_idx} 对应的图片尺寸，跳过此页的 bbox 映射。")
        else:
            page_to_size[page_idx] = page_size

    # --- 步骤 2: 构建 img_path -> 页面和BBox信息的映射 ---
    img_map = {}
    for page_res in middle_content_list["pdf_info"]:
        page_idx = page_res["page_idx"]

        para_blocks = page_res.get("para_blocks", None)
        if not para_blocks:  # 该页中没有图片
            continue

        for blocks in para_blocks:
            for block in blocks:
                for line in block["lines"]:
                    for item in line["spans"]:
                        if item.get("type") == "image":
                            img_path = item.get("image_path")    
                            bbox = item.get("bbox")
                            
                            if img_path and isinstance(page_idx, int) and bbox and len(bbox) == 4:
                                img_map[f"images/{img_path}"] = {
                                    "page_idx": page_idx,
                                    "bbox": bbox,
                                }
                        else:
                            print(f"[WARN] {middle_json_path} JSON中图片条目信息不完整，已跳过: {item}")


    # --- 步骤 3: 修改替换逻辑 ---
    def replace_image(match):
        img_path = match.group(2)  # () 中的内容

        if img_path in img_map:
            info = img_map[img_path]
            page_idx = info["page_idx"]
            x0, y0, x1, y1 = info["bbox"]

            if page_idx in page_to_size:
                width, height = page_to_size[page_idx]
                
                if width == 0 or height == 0:
                    print(f"[WARN] 图片 {img_path} (page {page_idx}) 的尺寸为零，无法归一化。")
                    return match.group(0)

                # 计算归一化坐标 (0-1000)
                norm_x0 = round((x0 / width) * 1000)
                norm_y0 = round((y0 / height) * 1000)
                norm_x1 = round((x1 / width) * 1000)
                norm_y1 = round((y1 / height) * 1000)

                final_filename = f"page{page_idx+1}_{int(norm_x0)}_{int(norm_y0)}_{int(norm_x1)}_{int(norm_y1)}.jpg"
                alt_text = ""
                return f"![{alt_text}]({final_filename})"
            else:
                print(f"[WARN] 找不到 page_idx {page_idx} img_path {img_path} 对应的图片尺寸。")
        else:
            print(f"[WARN] 未找到匹配: img_path={img_path}")      
        return match.group(0)

    new_content = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', replace_image, md_content)

    output_path = Path(output_md_path or md_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"文件 {output_path} 的图片链接已更新。")


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

def run_doc_parser(img_list, pdf_name, save_path):
    port = random.choice(ports)
    
    pdf_demo_path = Path("/tmp/dpi200")
    pdf_demo_path.mkdir(exist_ok=True)
    pdf_file = pdf_demo_path / f"{pdf_name}.pdf"

    output_dir = Path(save_path) / pdf_name / "vlm"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_file.exists():
        file_bytes = [open(img, "rb").read() for img in img_list]
        try:
            pdf_file.write_bytes(images_to_pdf_fitz(file_bytes, dpi=200))
        except Exception as e:
            print(f"Error: {e} in creating PDF {pdf_name}. Trying Pillow-based method...")
            pdf_file.write_bytes(images_bytes_to_pdf_bytes(file_bytes, dpi=200))

    if (output_dir / pdf_name / "vlm" / f"{pdf_name}_middle.json").exists() and \
        (output_dir / pdf_name / "vlm" / f"{pdf_name}_content_list.json").exists() and \
        (output_dir / pdf_name / "vlm" / f"{pdf_name}.md").exists():
        return (pdf_name, True, "skipped")

    cmd = [
        "mineru",
        "--path", str(pdf_file),
        "--output", str(Path(save_path)),
        "--backend", "vlm-http-client",
        "--url", f"http://127.0.0.1:{port}"
    ]

    try:
        md_path = output_dir / f"{pdf_name}.md"
        json_path = output_dir / f"{pdf_name}_content_list.json"
        middle_json_path = output_dir / f"{pdf_name}_middle.json"

        if not(md_path.exists() and json_path.exists() and middle_json_path.exists()):
            print(f"正在处理 {pdf_name}...")
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
                
        if not md_path.exists() or not json_path.exists():
            raise FileNotFoundError(f"Mineru 未能成功生成 {md_path} 或 {json_path}")

        # fill_image_captions_in_md(md_path, json_path, middle_json_path, output_md_path=md_final_path)
        
        return (pdf_name, True, "")
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode("utf-8", errors="ignore")
        print(f"[ERROR] {pdf_name} (CalledProcessError): {error_msg}")
        return (pdf_name, False, error_msg)
    except FileNotFoundError as e:
        print(f"[ERROR] {pdf_name} (FileNotFoundError): {e}")
        return (pdf_name, False, str(e))
    except Exception as e:
        print(f"[ERROR] {pdf_name} (General Exception): {e}")        
        return (pdf_name, False, str(e))


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

    data = [[item[1] for item in v] for k, v in raw_data.items()]
    pdf_names = list(raw_data.keys())

    print(f"Total pdfs:   {len(data)}")
    print(f"Total images: {sum([len(d) for d in data])}")

    success = []
    failed  = []
    skipped = []

    with ThreadPoolExecutor(max_workers=threads) as executor:
        future_to_pdf = {
            executor.submit(run_doc_parser, img_list, pdf_name, OUTPUT_PATH): pdf_name
            for pdf_name, img_list in zip(pdf_names, data)
        }
        
        progress_bar = tqdm(
            as_completed(future_to_pdf), 
            total=len(data), 
            desc="Processing"
        )

        for future in progress_bar:
            pdf_name = future_to_pdf[future]
            try:
                name, ok, msg = future.result()
                if msg == "skipped":
                    skipped.append(name)
                elif ok:
                    success.append(name)
                else:
                    failed.append((name, msg))
            except Exception as exc:
                print(f"[ERROR] {pdf_name} (Exception): {exc}")
                failed.append((pdf_name, str(exc)))
            
            progress_bar.set_postfix({
                "success": len(success),
                "failed":  len(failed),
                "skipped": len(skipped),
            })

    print("\nDone!")
    print(f"Success: {len(success)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Failed:  {len(failed)}")

    # if failed:
    #     print("\n--- Failed Tasks ---")
    #     for pdf_name, reason in failed:
    #         print(f"File:   {pdf_name}")
    #         print(f"Reason: {str(reason)[:300]}")
    #         print("-" * 40)


if __name__ == "__main__":
    main()
