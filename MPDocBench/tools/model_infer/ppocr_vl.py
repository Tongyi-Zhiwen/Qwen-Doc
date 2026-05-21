# conda activate /nas-mmu/zhoubb/envs/venv_paddleocr_deploy 用于部署    conda activate /nas-mmu/zhoubb/envs/venv_paddleocr 用于推理
# CUDA_VISIBLE_DEVICES=1 paddleocr genai_server --model_name PaddleOCR-VL-1.5-0.9B --backend vllm --port 8119 --backend_config /nas-mmu/zhoubb/code/IDP_tools/ppocr-vl/vllm_config.yaml &
# CUDA_VISIBLE_DEVICES=2 paddleocr genai_server --model_name PaddleOCR-VL-1.5-0.9B --backend vllm --port 8120 --backend_config /nas-mmu/zhoubb/code/IDP_tools/ppocr-vl/vllm_config.yaml &
# CUDA_VISIBLE_DEVICES=3 paddleocr genai_server --model_name PaddleOCR-VL-1.5-0.9B --backend vllm --port 8121 --backend_config /nas-mmu/zhoubb/code/IDP_tools/ppocr-vl/vllm_config.yaml &
# CUDA_VISIBLE_DEVICES=4 paddleocr genai_server --model_name PaddleOCR-VL-1.5-0.9B --backend vllm --port 8122 --backend_config /nas-mmu/zhoubb/code/IDP_tools/ppocr-vl/vllm_config.yaml &
# CUDA_VISIBLE_DEVICES=5 paddleocr genai_server --model_name PaddleOCR-VL-1.5-0.9B --backend vllm --port 8123 --backend_config /nas-mmu/zhoubb/code/IDP_tools/ppocr-vl/vllm_config.yaml &
# CUDA_VISIBLE_DEVICES=6 paddleocr genai_server --model_name PaddleOCR-VL-1.5-0.9B --backend vllm --port 8124 --backend_config /nas-mmu/zhoubb/code/IDP_tools/ppocr-vl/vllm_config.yaml &
# CUDA_VISIBLE_DEVICES=7 paddleocr genai_server --model_name PaddleOCR-VL-1.5-0.9B --backend vllm --port 8125 --backend_config /nas-mmu/zhoubb/code/IDP_tools/ppocr-vl/vllm_config.yaml

# CUDA_VISIBLE_DEVICES=1 paddleocr genai_server --model_name PaddleOCR-VL-0.9B --backend vllm --port 8119 --backend_config /nas-mmu/zhoubb/code/IDP_tools/ppocr-vl/vllm_config.yaml &
# CUDA_VISIBLE_DEVICES=2 paddleocr genai_server --model_name PaddleOCR-VL-0.9B --backend vllm --port 8120 --backend_config /nas-mmu/zhoubb/code/IDP_tools/ppocr-vl/vllm_config.yaml &
# CUDA_VISIBLE_DEVICES=3 paddleocr genai_server --model_name PaddleOCR-VL-0.9B --backend vllm --port 8121 --backend_config /nas-mmu/zhoubb/code/IDP_tools/ppocr-vl/vllm_config.yaml &
# CUDA_VISIBLE_DEVICES=4 paddleocr genai_server --model_name PaddleOCR-VL-0.9B --backend vllm --port 8122 --backend_config /nas-mmu/zhoubb/code/IDP_tools/ppocr-vl/vllm_config.yaml &
# CUDA_VISIBLE_DEVICES=5 paddleocr genai_server --model_name PaddleOCR-VL-0.9B --backend vllm --port 8123 --backend_config /nas-mmu/zhoubb/code/IDP_tools/ppocr-vl/vllm_config.yaml &
# CUDA_VISIBLE_DEVICES=6 paddleocr genai_server --model_name PaddleOCR-VL-0.9B --backend vllm --port 8124 --backend_config /nas-mmu/zhoubb/code/IDP_tools/ppocr-vl/vllm_config.yaml &
# CUDA_VISIBLE_DEVICES=7 paddleocr genai_server --model_name PaddleOCR-VL-0.9B --backend vllm --port 8125 --backend_config /nas-mmu/zhoubb/code/IDP_tools/ppocr-vl/vllm_config.yaml

import json
import os
import re
import fitz
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
from PIL import Image, ImageOps
from io import BytesIO
from paddleocr import PaddleOCRVL
from pathlib import Path  # 推荐使用 pathlib

# --- 配置项 --- 如果时1.5版本，请将 PIPELINE_VERSION 设为 "v1.5", 如果时1.0版本，请将 PIPELINE_VERSION 设为 "v1"
START_PORT = 8119
PORT_RANGE = 7
tasks_per_port = 5
VL_SERVEL_URL = "http://127.0.0.1:{}/v1"
PIPELINE_VERSION = "v1.5"
name = "ppocr_vl_1.5"
INPUT_FILE =  "MPDocBench.json"
OUTPUT_PATH = f"./markdown/{name}"


class PortPool:
    def __init__(self, ports: list[int], tasks_per_port: int = 4):
        self._queue = queue.Queue()
        self._all_pipelines = []
        
        print("初始化所有pipeline...")
        for port in ports:
            for i in range(tasks_per_port):
                pipeline = PaddleOCRVL(
                    pipeline_version=PIPELINE_VERSION,
                    vl_rec_backend="vllm-server",
                    vl_rec_server_url=VL_SERVEL_URL.format(port)
                )
                self._queue.put((port, pipeline))
                self._all_pipelines.append(pipeline)
                print(f"  port={port}, slot={i+1} 初始化完成")
        
        print(f"所有pipeline初始化完成！共{len(self._all_pipelines)}个")
    
    def acquire(self):
        """获取一个空闲的(port, pipeline)"""
        return self._queue.get()
    
    def release(self, port: int, pipeline):
        """归还(port, pipeline)"""
        self._queue.put((port, pipeline))

    def close_all(self):
        for pipeline in self._all_pipelines:
            pipeline.close()

# ============================================================
# MODIFIED: 将 md 中的 <img> 标签转换为归一化坐标的 Markdown 格式
# ============================================================
def convert_div_img_to_markdown(md_path, json_paths, image_paths, output_md_path=None):
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()
    
    # --- 新增步骤 1: 创建 page_id 到 image_size 的映射 ---
    page_to_size = {}
    print("正在读取图片尺寸...")
    for i, img_path in enumerate(image_paths):
        page_id = i + 1  # 页码从1开始
        try:
            with Image.open(img_path) as img:
                page_to_size[page_id] = img.size  # img.size 是 (width, height)
        except Exception as e:
            print(f"[WARN] 无法读取图片 {img_path} 的尺寸: {e}")


    bbox_to_page = defaultdict(list)
    for json_path in json_paths:
        with open(json_path, "r", encoding="utf-8") as f:
            res = json.load(f)
        page_id = res.get("page_index")
        if page_id is not None and isinstance(page_id, int):
            page_id += 1
        else:
            print(f"[WARN] {json_path} 的 page_index 不是有效整数")
            continue
        
        if page_id not in page_to_size:
            print(f"[WARN] 未找到 page_id {page_id} 对应的图片尺寸，跳过此页的 bbox 映射。")
            continue

        for block in res.get("parsing_res_list", []):
            bbox = block.get("block_bbox")
            label = block.get("block_label")
            if bbox and label:
                bbox_key = tuple(int(v) for v in bbox)
                bbox_to_page[(label, *bbox_key)].append(page_id)

    for key, page_id_lists in bbox_to_page.items():
        page_id_lists.sort()

    div_img_pattern = re.compile(
        r'<div\s+style="text-align:\s*center;">'
        r'<img\s+src="imgs/img_in_([a-zA-Z0-9_]+)_box_(\d+)_(\d+)_(\d+)_(\d+)\.[a-zA-Z]+"'
        r'[^/]*/></div>'
    )

    replaced_count = 0
    count = 0
    cur_page_id = 1
    
    def replace_div_img(match):
        nonlocal replaced_count, cur_page_id, count
        count += 1
        label = match.group(1)
        x0, y0, x1, y1 = map(int, match.groups()[1:])
        key = (label, x0, y0, x1, y1)
        
        page_id_list = bbox_to_page.get(key, [])
        if page_id_list:
            for page_id in page_id_list:
                if page_id >= cur_page_id and page_id in page_to_size:
                    width, height = page_to_size[page_id]
                    if width == 0 or height == 0:
                        print(f"[WARN] page {page_id} 尺寸为0，无法归一化。")
                        continue

                    # 计算归一化坐标 (0-1000)
                    norm_x0 = round((x0 / width) * 1000)
                    norm_y0 = round((y0 / height) * 1000)
                    norm_x1 = round((x1 / width) * 1000)
                    norm_y1 = round((y1 / height) * 1000)
                    
                    # --- 新增步骤 3: 返回新的 Markdown 格式 ---
                    cur_page_id = page_id
                    replaced_count += 1
                    return f"![](page{page_id}_{norm_x0}_{norm_y0}_{norm_x1}_{norm_y1}.jpg)"
                else:
                    print(f"[WARN] 找不到 page_id {page_id} 的尺寸信息。")
        
        print(f"[WARN] 未找到匹配: label={label}, bbox=({x0},{y0},{x1},{y1})")
        return match.group(0)

    new_content = div_img_pattern.sub(replace_div_img, md_content)
    output_path = Path(output_md_path or md_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"处理完成，共{count}个图像，替换 {replaced_count} 处 <div><img> 标签")


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


def run_doc_parser(image_paths, pdf_name, save_path_dir, port_pool: PortPool):
    save_dir = Path(save_path_dir) / pdf_name
    save_dir.mkdir(parents=True, exist_ok=True)

    md_final_path = save_dir / f"{pdf_name}.md" # 使用新的文件名以区分
    if md_final_path.exists():
        return (image_paths, True, "skipped")

    port, pipeline = port_pool.acquire()
    
    pdf_demo_dir = Path("/tmp/dpi144") # ppocrvl会以144dpi从pdf中创建图像
    pdf_demo_dir.mkdir(exist_ok=True)
    pdf_path = pdf_demo_dir / f"{pdf_name}.pdf"

    if not pdf_path.exists():
        file_bytes = [open(img, "rb").read() for img in image_paths]
        try:
            pdf_path.write_bytes(images_to_pdf_fitz(file_bytes, dpi=144))
        except Exception as e:
            print(f"Error: {e} in creating PDF {pdf_name}. Trying Pillow-based method...")
            pdf_path.write_bytes(images_bytes_to_pdf_bytes(file_bytes, dpi=144))

    try:
        md_raw_path = save_dir / f"{pdf_name}.md"
        json_paths = [save_dir / f'{pdf_name}_{idx}_res.json' for idx in range(len(image_paths))]
        render_img_path = [save_dir / f'{pdf_name}_layout_det_res_{idx}.png' for idx in range(len(image_paths))]
        all_parsed = md_raw_path.exists() and all(p.exists() for p in json_paths) and all(p.exists() for p in render_img_path)

        if not all_parsed:
            print(f"正在解析 {pdf_name}...")
            output = pipeline.predict(input=str(pdf_path))
            pages_res = list(output)
            for page in pages_res:
                page.save_to_json(save_path=str(save_dir))
            
            output = pipeline.restructure_pages(
                pages_res, 
                merge_tables=True, 
                relevel_titles=True, 
                concatenate_pages=True
            )
            for res in output:
                res.save_to_markdown(save_path=str(save_dir))
                res.save_to_json(save_path=str(save_dir))
                res.save_to_img(save_path=str(save_dir))

            # print(f"正在为 {pdf_name} 进行坐标归一化...")
            convert_div_img_to_markdown(md_raw_path, json_paths, render_img_path, output_md_path=md_final_path)
            return (image_paths, True, "")
        else:
            # print(f"{pdf_name} 的原始解析文件已存在，跳过解析。")
            return (image_paths, True, "skipped")

        

    except Exception as e:
        import traceback
        print(f"[ERROR] 在处理 {pdf_name} 时发生错误: {e}")
        traceback.print_exc() # 打印详细的错误堆栈
        return (image_paths, False, str(e))
        
    finally:
        port_pool.release(port, pipeline)


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
    ports = list(range(START_PORT, START_PORT + PORT_RANGE))
    
    port_pool = PortPool(ports, tasks_per_port=tasks_per_port)
    threads = len(ports) * tasks_per_port
    print(f"端口数: {len(ports)}, 每端口并发: {tasks_per_port}, 总线程数: {threads}")

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    raw_data = get_pdf_images(INPUT_FILE)

    # 修正数据提取逻辑，确保 image_paths 是字符串列表
    data = []
    pdf_names = []
    for k, v in raw_data.items():
        # v 是一个元组列表 [(index, img_path, json_path), ...]
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
                # 提交任务时，image_paths 是一个列表
                executor.submit(run_doc_parser, image_paths, pdf_name, OUTPUT_PATH, port_pool): pdf_name
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
        port_pool.close_all()
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
