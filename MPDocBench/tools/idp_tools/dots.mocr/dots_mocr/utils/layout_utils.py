from PIL import Image
from typing import Dict, List

import fitz
from io import BytesIO
import json
import re

from dots_mocr.utils.image_utils import smart_resize
from dots_mocr.utils.consts import MIN_PIXELS, MAX_PIXELS
from dots_mocr.utils.output_cleaner import OutputCleaner


# Define a color map (using RGBA format)
dict_layout_type_to_color = {
    "Text": (0, 128, 0, 256),  # Green, translucent
    "Picture": (255, 0, 255, 256),  # Magenta, translucent
    "Caption": (255, 165, 0, 256),  # Orange, translucent
    "Section-header": (0, 255, 255, 256),  # Cyan, translucent
    "Footnote": (0, 128, 0, 256),  # Green, translucent
    "Formula": (128, 128, 128, 256),  # Gray, translucent
    "Table": (255, 192, 203, 256),  # Pink, translucent
    "Title": (255, 0, 0, 256),  # Red, translucent
    "List-item": (0, 0, 255, 256),  # Blue, translucent
    "Page-header": (0, 128, 0, 256),  # Green, translucent
    "Page-footer":  (128, 0, 128, 256),  # Purple, translucent
    "Other": (165, 42, 42, 256),  # Brown, translucent
    "Unknown": (0, 0, 0, 0),
}


def draw_layout_on_image(image, cells, resized_height=None, resized_width=None, fill_bbox=True, draw_bbox=True):
    """
    Draw transparent boxes on an image.
    
    Args:
        image: The source PIL Image.
        cells: A list of cells containing bounding box information.
        resized_height: The resized height.
        resized_width: The resized width.
        fill_bbox: Whether to fill the bounding box.
        draw_bbox: Whether to draw the bounding box.
        
    Returns:
        PIL.Image: The image with drawings.
    """
    # origin_image = Image.open(image_path)
    original_width, original_height = image.size
        
    # Create a new PDF document
    doc = fitz.open()
    
    # Get image information
    img_bytes = BytesIO()
    image.save(img_bytes, format='PNG')
    # pix = fitz.Pixmap(image_path)
    pix = fitz.Pixmap(img_bytes)
    
    # Create a page
    page = doc.new_page(width=pix.width, height=pix.height)
    page.insert_image(
        fitz.Rect(0, 0, pix.width, pix.height), 
        # filename=image_path
        pixmap=pix
        )

    for i, cell in enumerate(cells):
        bbox = cell['bbox']
        layout_type = cell['category']
        order = i
        
        top_left = (bbox[0], bbox[1])
        down_right = (bbox[2], bbox[3])
        if resized_height and resized_width:
            scale_x = resized_width / original_width
            scale_y = resized_height / original_height
            top_left = (int(bbox[0] / scale_x), int(bbox[1] / scale_y))
            down_right = (int(bbox[2] / scale_x), int(bbox[3] / scale_y))
            
        color = dict_layout_type_to_color.get(layout_type, (0, 128, 0, 256))
        color = [col/255 for col in color[:3]]

        x0, y0, x1, y1 = top_left[0], top_left[1], down_right[0], down_right[1]
        rect_coords = fitz.Rect(x0, y0, x1, y1)
        if draw_bbox:
            if fill_bbox:
                page.draw_rect(
                    rect_coords,
                    color=None,
                    fill=color,
                    fill_opacity=0.3,
                    width=0.5,
                    overlay=True,
                )  # Draw the rectangle
            else:
                page.draw_rect(
                    rect_coords,
                    color=color,
                    fill=None,
                    fill_opacity=1,
                    width=0.5,
                    overlay=True,
                )  # Draw the rectangle
        order_cate = f"{order}_{layout_type}"
        page.insert_text(
            (x1, y0 + 20), order_cate, fontsize=20, color=color
        )  # Insert the index in the top left corner of the rectangle

    # Convert to a Pixmap (maintaining original dimensions)
    mat = fitz.Matrix(1.0, 1.0)
    pix = page.get_pixmap(matrix=mat)

    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def pre_process_bboxes(
    origin_image,
    bboxes,
    input_width,
    input_height,
    factor: int = 28,
    min_pixels: int = 3136, 
    max_pixels: int = 11289600
):
    assert isinstance(bboxes, list) and len(bboxes) > 0 and isinstance(bboxes[0], list)
    min_pixels = min_pixels or MIN_PIXELS
    max_pixels = max_pixels or MAX_PIXELS
    original_width, original_height = origin_image.size

    input_height, input_width = smart_resize(input_height, input_width, min_pixels=min_pixels, max_pixels=max_pixels)
    
    scale_x = original_width / input_width
    scale_y = original_height / input_height

    bboxes_out = []
    for bbox in bboxes:
        bbox_resized = [
            int(float(bbox[0]) / scale_x), 
            int(float(bbox[1]) / scale_y),
            int(float(bbox[2]) / scale_x), 
            int(float(bbox[3]) / scale_y)
        ]
        bboxes_out.append(bbox_resized)
    
    return bboxes_out

def post_process_cells(
    origin_image: Image.Image, 
    cells: List[Dict], 
    input_width,  # server input width, also has smart_resize in server
    input_height,
    factor: int = 28,
    min_pixels: int = 3136, 
    max_pixels: int = 11289600
) -> List[Dict]:
    """
    Post-processes cell bounding boxes, converting coordinates from the resized dimensions back to the original dimensions.
    
    Args:
        origin_image: The original PIL Image.
        cells: A list of cells containing bounding box information.
        input_width: The width of the input image sent to the server.
        input_height: The height of the input image sent to the server.
        factor: Resizing factor.
        min_pixels: Minimum number of pixels.
        max_pixels: Maximum number of pixels.
        
    Returns:
        A list of post-processed cells.
    """
    assert isinstance(cells, list) and len(cells) > 0 and isinstance(cells[0], dict)
    min_pixels = min_pixels or MIN_PIXELS
    max_pixels = max_pixels or MAX_PIXELS
    original_width, original_height = origin_image.size

    input_height, input_width = smart_resize(input_height, input_width, min_pixels=min_pixels, max_pixels=max_pixels)
    
    scale_x = input_width / original_width
    scale_y = input_height / original_height
    
    cells_out = []
    for cell in cells:
        bbox = cell['bbox']
        bbox_resized = [
            int(float(bbox[0]) / scale_x), 
            int(float(bbox[1]) / scale_y),
            int(float(bbox[2]) / scale_x), 
            int(float(bbox[3]) / scale_y)
        ]
        cell_copy = cell.copy()
        cell_copy['bbox'] = bbox_resized
        cells_out.append(cell_copy)
    
    return cells_out

def is_legal_bbox(cells):
    for cell in cells:
        bbox = cell['bbox']
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            return False
    return True

def post_process_output(response, prompt_mode, origin_image, input_image, min_pixels=None, max_pixels=None):
    if prompt_mode in ["prompt_ocr", "prompt_table_html", "prompt_table_latex", "prompt_formula_latex"]:
        return response

    json_load_failed = False
    cells = response
    try:
        cells = json.loads(cells)
        cells = post_process_cells(
            origin_image, 
            cells,
            input_image.width,
            input_image.height,
            min_pixels=min_pixels,
            max_pixels=max_pixels
        )
        return cells, False
    except Exception as e:
        print(f"cells post process error: {e}, when using {prompt_mode}")
        json_load_failed = True

    if json_load_failed:
        cleaner = OutputCleaner()
        response_clean = cleaner.clean_model_output(cells)
        if isinstance(response_clean, list):
            response_clean = "\n\n".join([cell['text'] for cell in response_clean if 'text' in cell])
        return response_clean, True


# ========== Scene Text Visualization ==========
def parse_scene_text_output(response: str) -> List[Dict]:
    """解析场景文本输出，格式: (x1, y1), (x2, y2), (x3, y3), (x4, y4) text"""
    pattern = re.compile(
        r'\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*'  # 点1
        r'\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*'  # 点2
        r'\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*'  # 点3
        r'\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*'      # 点4
        r'(.*?)'                               # 文本内容（非贪婪）
        r'(?=\(\s*\d+\s*,\s*\d+\s*\)|\Z)',     # 下一个坐标点开始或结束
        re.DOTALL
    )

    results: List[Dict] = []
    for m in pattern.finditer(response):
        coords = [int(m.group(i)) for i in range(1, 9)]
        text = m.group(9).strip()
        if text:
            results.append({'points': coords, 'text': text})
    return results

def post_process_scene_text(response, origin_image, input_image, min_pixels=None, max_pixels=None):
    """坐标反算：从模型输出坐标映射回原图"""
    instances = parse_scene_text_output(response)
    if not instances:
        return response, True
    
    min_pixels = min_pixels or MIN_PIXELS
    max_pixels = max_pixels or MAX_PIXELS

    original_width, original_height = origin_image.size
    input_h, input_w = smart_resize(
        original_height, original_width, 
        min_pixels=min_pixels, 
        max_pixels=max_pixels
    )
    
    scale_x, scale_y = input_w / origin_image.width, input_h / origin_image.height
    
    for inst in instances:
        pts = inst['points']
        inst['points'] = [int(pts[i] / (scale_x if i % 2 == 0 else scale_y)) for i in range(8)]
    
    return instances, False

def draw_scene_text_on_image(image, instances):
    """绘制场景文本：绿色多边形框 + 白色文字标签"""
    import cv2
    import numpy as np
    
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    for i, inst in enumerate(instances):
        pts = np.array(inst['points']).reshape(4, 2).astype(np.int32)
        x, y = pts[0]
        text = inst['text'][:40] + ('...' if len(inst['text']) > 40 else '')
        
        cv2.polylines(img, [pts], True, (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x, y - th - 8), (x + tw, y - 3), (0, 0, 0), -1)
        cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def format_scene_text_to_markdown(instances):
    """格式化为 Markdown，保留原始坐标格式"""
    if isinstance(instances, str):
        return instances
    lines = []
    for inst in instances:
        pts = inst['points']
        # 格式: (x1, y1), (x2, y2), (x3, y3), (x4, y4) text
        coord_str = f"({pts[0]}, {pts[1]}), ({pts[2]}, {pts[3]}), ({pts[4]}, {pts[5]}), ({pts[6]}, {pts[7]})"
        lines.append(f"{coord_str} {inst['text']}\n")
    return "# Scene Text Results\n\n" + "\n".join(lines)

