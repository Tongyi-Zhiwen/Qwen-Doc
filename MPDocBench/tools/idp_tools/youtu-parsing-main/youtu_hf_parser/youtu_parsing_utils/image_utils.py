import io
import os
import fitz
from PIL import Image
from pathlib import Path

# Optional import for URL loading
try:
    import requests
except ImportError:
    requests = None

def fitz_pdf_to_image(page, target_dpi=200, max_side=4500, fallback_dpi=72, bg_color=(255, 255, 255)):
    """
    Render a fitz.Page to a PIL.Image (RGB) reasonably safely.

    Args:
        page: fitz.Page instance (one page of an opened Document).
        target_dpi: desired DPI for rendering (default 200).
        max_side: maximum allowed width/height in pixels; if exceeded, fall back to fallback_dpi.
        fallback_dpi: dpi to use if target_dpi produces too-large images (default 72).
        bg_color: background color tuple for compositing if the pixmap has alpha.

    Returns:
        PIL.Image in RGB mode.
    """
    if not hasattr(page, "get_pixmap"):
        raise TypeError("page must be a fitz.Page")

    # build matrix for the desired DPI
    mat = fitz.Matrix(float(target_dpi) / 72.0, float(target_dpi) / 72.0)

    # render to pixmap (no alpha by default)
    pm = page.get_pixmap(matrix=mat, alpha=False)

    # if image too large, fall back to a smaller DPI and re-render once
    if (pm.width > max_side or pm.height > max_side) and target_dpi != fallback_dpi:
        mat = fitz.Matrix(float(fallback_dpi) / 72.0, float(fallback_dpi) / 72.0)
        pm = page.get_pixmap(matrix=mat, alpha=False)

    # determine PIL mode
    # Some PyMuPDF versions expose pm.alpha boolean; fallback check: n == 4 implies RGBA
    has_alpha = getattr(pm, "alpha", None)
    if has_alpha is None:
        # pm.n is number of components (3 = RGB, 4 = RGBA)
        has_alpha = getattr(pm, "n", 3) == 4

    mode = "RGBA" if has_alpha else "RGB"

    # pm.samples is the raw bytes (row-major)
    image = Image.frombytes(mode, (pm.width, pm.height), pm.samples)

    # If there's an alpha channel, composite onto background and return RGB
    if mode == "RGBA":
        bg = Image.new("RGB", image.size, bg_color)
        bg.paste(image, mask=image.split()[3])  # use alpha channel as mask
        return bg

    # already RGB
    return image



def load_images_from_pdf(pdf_path, dpi=200, start_page_idx=0, end_page_idx=-1):

    images = []
    with fitz.open(pdf_path) as p:
        if end_page_idx == -1 or end_page_idx > p.page_count - 1:
            end_page_idx = p.page_count - 1

        for page_idx in range(p.page_count):
            if start_page_idx <= page_idx <= end_page_idx:
                page = p[page_idx]
                img = fitz_pdf_to_image(page, target_dpi=dpi)
                images.append(img)
    return images


def load_image(input_path):
    """
    Load an image and return a PIL.Image in RGB mode.

    Supported input types:
      - PIL.Image.Image -> returned converted to RGB
      - str or pathlib.Path -> filesystem path
      - bytes or bytearray -> raw image bytes
      - file-like object with .read() -> read and load
      - URL string (http/https) if 'requests' is available

    Raises:
      - ValueError for unsupported types
      - IOError / OSError if PIL cannot open the image
      - RuntimeError if URL requested but 'requests' is not installed
    """
    # If already a PIL Image, just convert to RGB and return a copy
    if isinstance(input_path, Image.Image):
        return input_path.convert("RGB")

    # Path-like (str or Path)
    if isinstance(input_path, (str, Path)):
        s = str(input_path)
        # URL case
        if s.startswith("http://") or s.startswith("https://"):
            if requests is None:
                raise RuntimeError("requests is required to load images from URLs")
            resp = requests.get(s, timeout=30)
            resp.raise_for_status()
            data = resp.content
            bio = io.BytesIO(data)
            with Image.open(bio) as im:
                return im.convert("RGB")
        # filesystem path
        if os.path.exists(s):
            with Image.open(s) as im:
                return im.convert("RGB")
        # treat as raw bytes path-like string is not file -> error
        raise ValueError(f"File not found: {s}")

    # bytes / bytearray
    if isinstance(input_path, (bytes, bytearray)):
        bio = io.BytesIO(input_path)
        with Image.open(bio) as im:
            return im.convert("RGB")

    # file-like object (has read)
    if hasattr(input_path, "read"):
        # ensure we have bytes (some file-like return str; PIL wants bytes)
        data = input_path.read()
        # if it's text, encode? better to error
        if isinstance(data, str):
            raise ValueError("file-like.read() returned str, expected bytes")
        bio = io.BytesIO(data)
        with Image.open(bio) as im:
            return im.convert("RGB")

    raise ValueError("Unsupported input type for fetch_image")
