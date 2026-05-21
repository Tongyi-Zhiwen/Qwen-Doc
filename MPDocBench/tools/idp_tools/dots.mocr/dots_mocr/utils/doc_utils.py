import fitz
import numpy as np
import enum
from pydantic import BaseModel, Field
from PIL import Image


class SupportedPdfParseMethod(enum.Enum):
    OCR = 'ocr'
    TXT = 'txt'


class PageInfo(BaseModel):
    """The width and height of page
    """
    w: float = Field(description='the width of page')
    h: float = Field(description='the height of page')


def get_matrix(page, dpi_default=200, max_pixels=11289600):
    rect = page.rect
    if rect.width * rect.height > max_pixels:
        factor = (max_pixels / (rect.width * rect.height)) ** 0.5
    else:
        factor = dpi_default / 72
    mat = fitz.Matrix(factor, factor)
    return mat

def is_page_safe_to_render(page, max_image_pixels=30_000_000):
    """
    检查一个页面是否包含可能导致内存问题的超大图片。
    
    Args:
        page (pymupdf.Page): 要检查的页面对象。
        max_image_pixels (int): 单个图片允许的最大像素数 (宽*高)。
                                默认3000万像素，约对应 5000x6000 的图片，
                                解压后约 120MB (RGBA)，是一个比较安全的上限。

    Returns:
        bool: 如果页面安全则返回 True，否则返回 False。
        str: 包含原因的描述信息。
    """
    image_list = page.get_images(full=True)
    if not image_list:
        return True, "页面不含图片。"

    for img_index, img in enumerate(image_list):
        xref = img[0]
        if xref == 0:  # 内联图片，通常较小，但也可以检查
            continue
        
        try:
            # 只获取图片信息，不解压！这是关键！
            width = img[2]  # 直接从元数据获取宽度
            height = img[3] # 直接从元数据获取高度

            if width * height > max_image_pixels:
                reason = (
                    f"页面包含一个超大尺寸的内嵌图片 (xref: {xref}, "
                    f"尺寸: {width}x{height})，像素数超过阈值 {max_image_pixels}。"
                )
                return False, reason
        
        except Exception as e:
            # 如果连获取信息都失败，也标记为不安全
            reason = f"检查图片 xref:{xref} 的元信息时出错: {e}"
            return False, reason
            
    return True, "页面所有图片尺寸都在安全范围内。"

def fitz_doc_to_image(doc, target_dpi=200, origin_dpi=None) -> dict:
    """Convert fitz.Document to image, Then convert the image to numpy array.

    Args:
        doc (_type_): pymudoc page
        dpi (int, optional): reset the dpi of dpi. Defaults to 200.

    Returns:
        dict:  {'img': numpy array, 'width': width, 'height': height }
    """
    from PIL import Image
    # mat = fitz.Matrix(target_dpi / 72, target_dpi / 72)
    mat = get_matrix(doc, target_dpi)
    pm = doc.get_pixmap(matrix=mat, alpha=False)
    if pm.width == 0 or pm.height == 0:
        print(f"image is empty loading from pdf, skip")
        return None
        
    if pm.width > 4500 or pm.height > 4500:
        mat = fitz.Matrix(72 / 72, 72 / 72)  # use fitz default dpi
        pm = doc.get_pixmap(matrix=mat, alpha=False)

    image = Image.frombytes('RGB', (pm.width, pm.height), pm.samples)
    return image


def load_images_from_pdf(pdf_file, dpi=200, start_page_id=0, end_page_id=None) -> list:
    images = []
    with fitz.open(pdf_file) as doc:
        pdf_page_num = doc.page_count
        end_page_id = (
            end_page_id
            if end_page_id is not None and end_page_id >= 0
            else pdf_page_num - 1
        )
        if end_page_id > pdf_page_num - 1:
            print('end_page_id is out of range, use images length')
            end_page_id = pdf_page_num - 1

        for index in range(0, doc.page_count):
            if start_page_id <= index <= end_page_id:
                page = doc[index]
                is_safe, reason = is_page_safe_to_render(page)
                if not is_safe:
                    print(f"pdf page {index} is not safe to render, skip")
                    return []
                img = fitz_doc_to_image(page, target_dpi=dpi)
                if img is None:
                    print(f"pdf page {index} is empty, skip")
                    continue
                images.append(img)
    return images