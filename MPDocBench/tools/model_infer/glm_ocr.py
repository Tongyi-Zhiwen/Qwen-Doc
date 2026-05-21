# 推理和部署使用同一个环境 conda activate /nas-mmu/zhoubb/envs/glm_ocr
# 环境安装参考: https://github.com/zai-org/GLM-OCR/tree/v0.1.4
# glmocr 0.1.4 vllm 0.18.0 torch 2.10.0+cu128 transformers 5.3.0
# MODEL="/nas-mmu/xhd/checkpoints/official_models/ZhipuAI/GLM-OCR"
# SPEC='{"method": "mtp", "num_speculative_tokens": 1}'
# for i in {1..7}; do
#     PORT=$((7080 + i))
#     CUDA_VISIBLE_DEVICES=$i vllm serve $MODEL \
#         --host 0.0.0.0 \
#         --port $PORT \
#         --allowed-local-media-path / \
#         --served-model-name glm-ocr \
#         --speculative-config "$SPEC" &
#     echo "✅ 启动 GPU$i -> 端口$PORT"
# done
# # 等待所有进程
# wait
# echo "🎉 全部启动完成"


import os
import subprocess
import json
import fitz
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
from io import BytesIO
from PIL import Image, ImageOps

# --- 配置 ---
# 7个可用的端口，将作为资源池
ports = [p for p in range(7081, 7088)] 
# 并发线程数应等于或略大于端口数。这里设置为端口数，因为实际并发上限是端口数量。
threads = len(ports)
INPUT_FILE = "MPDocBench.json"
OUTPUT_PATH = "./markdown/glm_ocr"


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


def run_doc_parser(image_paths, pdf_name, save_path_dir, port_queue):
    """
    处理单个PDF的任务函数。
    它会从 port_queue 获取一个独占端口，并在完成后归还。
    """
    port = None
    try:
        # 从队列中获取一个可用端口，如果队列为空则阻塞等待
        port = port_queue.get()
        save_dir = Path(save_path_dir)
    
        if os.path.exists(save_dir / pdf_name / f"{pdf_name}.md") and os.path.exists(save_dir / pdf_name / f"{pdf_name}.json"):
            return (image_paths, True, "skipped")

        # 统一存放生成的PDF文件，避免重复生成
        pdf_demo_dir = Path("/nas-mmu/zhoubb/pdf_demo/dpi144")
        pdf_demo_dir.mkdir(exist_ok=True)
        pdf_path = pdf_demo_dir / f"{pdf_name}.pdf"

        if not pdf_path.exists():
            file_bytes = [open(img, "rb").read() for img in image_paths]
            try:
                pdf_path.write_bytes(images_to_pdf_fitz(file_bytes, dpi=144))
            except Exception as e:
                print(f"Error: {e} in creating PDF {pdf_name}. Trying Pillow-based method...")
                pdf_path.write_bytes(images_bytes_to_pdf_bytes(file_bytes, dpi=144))
            
        # 使用获取到的独占端口
        cmd = [
            "glmocr",
            "parse",
            str(pdf_path), # 确保路径是字符串
            "--config", f"./tools/model_infer/glm_ocr_configs/{port}.yaml",
            "--layout-device", f"cuda:0", # 动态分配GPU ID，假设GPU ID从0开始
            "--output", str(save_dir) # 确保路径是字符串
        ]
        # 使用subprocess.run执行命令
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        
        if result.returncode != 0:
            # 如果命令执行失败，抛出异常
            raise subprocess.CalledProcessError(result.returncode, cmd, output=result.stdout, stderr=result.stderr)

        return (image_paths, True, "")

    except Exception as e:
        error_info = f"Stderr: {e.stderr}" if isinstance(e, subprocess.CalledProcessError) else str(e)
        print(f"任务 {pdf_name} 在端口 {port} 上失败: {error_info}")
        return (image_paths, False, error_info)
    finally:
        if port is not None:
            port_queue.put(port)



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
        image_paths_for_pdf = [item[1] for item in v]
        data.append(image_paths_for_pdf)
        pdf_names.append(k)

    print(f"总任务数 (PDFs): {len(data)}")
    print(f"总图片数: {sum([len(d) for d in data])}")
    print(f"使用 {len(ports)} 个端口进行并行处理...")

    # 创建端口池
    port_pool_queue = queue.Queue()
    for p in ports:
        port_pool_queue.put(p)

    success = []
    failed  = []
    skipped = []

    with ThreadPoolExecutor(max_workers=threads) as executor:
        future_to_pdf = {
            executor.submit(run_doc_parser, image_paths, pdf_name, OUTPUT_PATH, port_pool_queue): pdf_name
            for pdf_name, image_paths in zip(pdf_names, data)
        }
        
        progress_bar = tqdm(
            as_completed(future_to_pdf), 
            total=len(data), 
            desc="Processing PDFs"
        )

        for future in progress_bar:
            pdf_name = future_to_pdf[future]
            try:
                img_paths, ok, msg = future.result()
                if msg == "skipped":
                    skipped.append(pdf_name)
                elif ok:
                    print(f"成功处理: {pdf_name}")
                    success.append(pdf_name)
                else:
                    failed.append((pdf_name, msg))
            except Exception as exc:
                failed.append((pdf_name, str(exc)))
            
            # 更新进度条的后缀信息
            progress_bar.set_postfix({
                "success": len(success),
                "failed":  len(failed),
                "skipped": len(skipped),
            })

    print("\n处理完成!")
    print(f"成功: {len(success)}")
    print(f"跳过: {len(skipped)}")
    print(f"失败: {len(failed)}")

    if failed:
        print("\n--- 失败的任务列表 ---")
        for pdf_name, reason in failed:
            print(f"- 文件: {pdf_name}")
            print(f"  原因: {str(reason)[:400]}...") # 打印部分错误信息
        
        


if __name__ == "__main__":
    main()


