# 可以多运行几遍这个脚本，直到所有文档都成功（因为有些可能会偶尔失败或被内容审查）。每次运行它都会跳过已经成功的页面和文档，所以是安全的。
import os
import sys
import json
import re
import base64
import multiprocessing
import concurrent.futures
from tqdm import tqdm
from openai import OpenAI

# ---------------- 配置区 ----------------
MODEL_NAME = "qwen3-vl-235b-a22b-instruct"

INPUT_FILE =  "MPDocBench.json"
OUTPUT_PATH = f"./markdown/{MODEL_NAME}"

# 文档间并行（多进程）
MAX_DOCUMENT_PROCESSES = 2
# 文档内页面并行（多线程）
MAX_PAGE_WORKERS = 8

# 商业 API（OpenAI compatible）
API_KEY = os.getenv("MODELSCOPE_KEY", "sk-xxxxx")
API_BASE = os.getenv("MODELSCOPE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# ---------------- Prompt ----------------
IMAGE__PROMPT = r'''You are an AI assistant specialized in converting document images to structured Markdown. Please follow these instructions for the conversion:

            1. Text Processing:
            - Accurately recognize all text from the image.
            - Convert recognized text into Markdown, maintaining the original structure (headings, paragraphs, lists, etc.).

            2. Mathematical Formula Processing:
            - Convert all mathematical formulas to LaTeX format.
            - Enclose inline formulas with `\(...\)`.
            - Enclose block formulas with `\[...\]`.

            3. Table Processing:
            - Convert tables into HTML format, wrapped within `<table>` and `</table>` tags.

            4. Figure Handling:
            - Detect figures and images that are part of the main document content.
            - Do not detect or output images located in the page's header or footer.
            - For each detected content figure, determine its bounding box coordinates, normalized to a 1000x1000 grid.
            - Represent each figure using this specific Markdown syntax: `![](x1_y1_x2_y2.jpg)`.

            5. Output Format:
            - Ensure the final Markdown document has a clear structure.
            - Place the generated figure tags (`![](...).jpg`) in the document flow where the original figures were located.

            Your task is to accurately convert the content of the document image into a single Markdown string, without adding any extra explanations or comments.
            '''

# ---------------- 清理函数 ----------------
def clean_markdown(markdown_text: str) -> str:
    t = (markdown_text or "").strip()
    if t.startswith("```markdown"):
        t = t[len("```markdown"):].strip()
    t = t.strip()
    if t.endswith("```"):
        t = t[:-len("```")].strip()
    return t

# ---------------- 工具函数 ----------------
def _guess_image_mime(path: str) -> str:
    ext = path.lower().split(".")[-1]
    if ext == "png":
        return "image/png"
    if ext in ("jpg", "jpeg"):
        return "image/jpeg"
    if ext == "webp":
        return "image/webp"
    raise ValueError(f"Unsupported image format: {ext}")

def image_to_data_url(path: str) -> str:
    mime = _guess_image_mime(path)
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

# 把 ![](x1_y1_x2_y2.jpg) -> ![](pageidx_x1_y1_x2_y2.jpg)
_SINGLE_FIG_RE = re.compile(r'!\[\]\((\d+_\d+_\d+_\d+)\.jpg\)')

def add_pageidx_to_singlepage_fig_tags(md: str, pageidx_1based: int) -> str:
    def _rep(m):
        coords = m.group(1)
        return f"![](page{pageidx_1based}_{coords}.jpg)"
    return _SINGLE_FIG_RE.sub(_rep, md)

def page_md_path(doc_dir: str, page_idx0: int) -> str:
    return os.path.join(doc_dir, f"{page_idx0+1:04d}.md")

def final_md_path(doc_dir: str) -> str:
    return os.path.join(doc_dir, "_full.md")

# ---------------- 单页解析 worker（成功就落盘） ----------------
def process_page_worker(args):
    """
    args: (doc_dir, image_path, page_idx0, model_name)
    成功：写入 doc_dir/0000.md ... 并返回 (page_idx0, True)
    失败：不写或写错误标记，并返回 (page_idx0, False)
    """
    doc_dir, image_path, page_idx0, model_name = args
    out_page_path = page_md_path(doc_dir, page_idx0)

    # 已经成功过就跳过（支持重复运行脚本）
    if os.path.exists(out_page_path):
        return (page_idx0, True, "cached")

    client = OpenAI(api_key=API_KEY, base_url=API_BASE)
    try:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                {"type": "text", "text": IMAGE__PROMPT},
            ]
        }]
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=16384,
            temperature=0.0,
        )
        md = resp.choices[0].message.content
        md = clean_markdown(md)
        md = add_pageidx_to_singlepage_fig_tags(md, page_idx0 + 1)

        # 原子写入：先写 tmp 再 replace，避免并发/中途崩溃导致空文件
        tmp_path = out_page_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(md)
        os.replace(tmp_path, out_page_path)

        return (page_idx0, True, "ok")

    except Exception as e:
        def _is_inappropriate_content_error(obj) -> bool:
            INAPPROPRIATE_PREFIX = "Output data may contain inappropriate content."
            """
            兼容两类情况：
            1) 返回的是 dict/JSON 字符串：{'error': {'message': 'Output data may contain inappropriate content...'}}
            2) Exception 的字符串里带上述 message
            """
            try:
                if isinstance(obj, dict):
                    msg = (((obj.get("error") or {}).get("message")) or "")
                    return msg.startswith(INAPPROPRIATE_PREFIX)

                if isinstance(obj, str):
                    # 可能是 JSON 字符串，也可能是普通文本/报错文本
                    s = obj.strip()
                    if s.startswith("{") and '"error"' in s:
                        j = json.loads(s)
                        msg = (((j.get("error") or {}).get("message")) or "")
                        return msg.startswith(INAPPROPRIATE_PREFIX)
                    return INAPPROPRIATE_PREFIX in s
                return False
            except Exception:
                return False

        if _is_inappropriate_content_error(str(e)):
            tmp_path = out_page_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write("")
            os.replace(tmp_path, out_page_path)
            return (page_idx0, True, "blocked(inappropriate_content)")

        print(f"[PAGE FAIL] {image_path} page={page_idx0+1}: {e}", file=sys.stderr)
        return (page_idx0, False, str(e))

# ---------------- 文档处理（线程池并发页；全成功才合并） ----------------
def process_single_document(task):
    """
    对每个 pdf_name：
    - 在 OUTPUT_PATH 下建子目录 OUTPUT_PATH/pdf_name/
    - 每页成功后写入 0000.md, 0001.md ...
    - 当所有页面都存在且非空：合并成 _full.md
    """
    image_paths, save_root, pdf_name, model_name = task
    doc_dir = os.path.join(save_root, pdf_name)
    os.makedirs(doc_dir, exist_ok=True)

    full_path = final_md_path(doc_dir)

    # 如果已经有 full 且非空，直接跳过
    if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
        return f"Skipped(full exists): {pdf_name}"

    # 并发处理页面
    page_tasks = [(doc_dir, p, i, model_name) for i, p in enumerate(image_paths)]
    results = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PAGE_WORKERS) as ex:
            futs = [ex.submit(process_page_worker, t) for t in page_tasks]
            for fut in concurrent.futures.as_completed(futs):
                results.append(fut.result())
    except Exception as e:
        print(f"[DOC FAIL submit/collect] {pdf_name}: {e}", file=sys.stderr)

    # 检查是否所有页面都成功落盘
    missing = []
    for i in range(len(image_paths)):
        p = page_md_path(doc_dir, i)
        if not (os.path.exists(p)): # and os.path.getsize(p) > 0):
            missing.append(i + 1)

    if missing:
        return f"Partial: {pdf_name} (missing pages: {len(missing)}/{len(image_paths)})"

    # 全部存在：合并
    try:
        parts = []
        for i in range(len(image_paths)):
            p = page_md_path(doc_dir, i)
            with open(p, "r", encoding="utf-8") as f:
                parts.append(f.read().rstrip())

        full_md = "\n\n".join(parts).strip() + "\n"

        tmp_full = full_path + ".tmp"
        with open(tmp_full, "w", encoding="utf-8") as f:
            f.write(full_md)
        os.replace(tmp_full, full_path)

        return f"Success: {pdf_name}"

    except Exception as e:
        print(f"[DOC FAIL merge] {pdf_name}: {e}", file=sys.stderr)
        return f"Error(merge): {pdf_name}"



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


# ---------------- main ----------------
if __name__ == "__main__":
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    raw_data = get_pdf_images(INPUT_FILE)

    pdf_names = list(raw_data.keys())
    docs = [[img_path for (index, img_path, json_path) in raw_data[pdf_name]] for pdf_name in pdf_names]

    all_doc_tasks = [(image_paths, OUTPUT_PATH, pdf_name, MODEL_NAME)
                     for image_paths, pdf_name in zip(docs, pdf_names)]

    print(f"找到 {len(all_doc_tasks)} 个文档任务。")
    print("策略：单页并发调用；每页成功即写入子目录；全部页齐全后合并为 _full.md。可重复运行直到全部完成。")
    print(f"文档并行进程: {MAX_DOCUMENT_PROCESSES}, 文档内页面线程: {MAX_PAGE_WORKERS}")
    print(f"API_BASE={API_BASE}, MODEL={MODEL_NAME}")

    with multiprocessing.Pool(processes=MAX_DOCUMENT_PROCESSES) as pool:
        results = []
        for r in tqdm(pool.imap_unordered(process_single_document, all_doc_tasks),
                      total=len(all_doc_tasks),
                      desc="Processing Documents"):
            results.append(r)

    success = sum(1 for r in results if r.startswith("Success") or r.startswith("Skipped"))
    partial = sum(1 for r in results if r.startswith("Partial"))
    errors = len(results) - success - partial
    print("\n所有文档处理完毕。")
    print(f"摘要: success/skip={success}, partial={partial}, error={errors}")
