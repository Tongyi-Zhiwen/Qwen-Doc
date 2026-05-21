from PIL import Image, ImageDraw, ImageFont
import math
from io import BytesIO

# Color map for each layout type (RGBA)
dict_layout_type_to_color = {
    "Text":    (51, 160, 44, 255),    # Bright green, high saturation for distinction
    "Figure":  (214, 39, 40, 255),    # Vivid red, visually prominent
    "Caption": (255, 127, 14, 255),   # Bright orange – contrast with red/green
    "Header":  (31, 119, 180, 255),   # Deep blue – distinguish headers
    "Footer":  (148, 103, 189, 255),  # Deep purple, distinct from blue and others
    "Formula": (23, 190, 207, 255),   # Cyan blue – highlights formula regions
    "Table":   (247, 182, 210, 255),  # Light pink for tables
    "Title":   (255, 217, 47, 255),   # Bright yellow – draws attention to title
    "Code":    (127, 127, 127, 255),  # Neutral gray for code blocks
    "Unknown": (200, 200, 200, 128),  # Semi-transparent light gray for unknowns
    "Chart":   (102, 195, 165, 255),  # Light teal for charts
    "Seal":    (140, 86, 75, 255),   # Brown for seals
}

def draw_layout_on_image(image, cells, fill_bbox=True, draw_bbox=True):
    """
    Draw layout annotations on an image.

    Args:
        image (PIL.Image):        The base image to annotate.
        cells (list of dict):     List of layout elements, each with 'bbox' and 'type'.
                                  bbox: [x0, y0, x1, y1, x2, y2, x3, y3] (quadrilateral)
                                  type: String label for region type.
        fill_bbox (bool):         Fill the region polygon if True.
        draw_bbox (bool):         Draw the region outline if True.

    Returns:
        PIL.Image:                Annotated image in RGB mode.
    """
    # Ensure input image is in RGBA
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Create transparent overlay to draw layouts
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Load a font (fall back to default if unavailable)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()

    # Iterate over each cell layout
    for i, cell in enumerate(cells):
        # Get polygon points from bbox
        bbox = cell['bbox']  # [x0, y0, x1, y1, x2, y2, x3, y3]
        pts = [(bbox[j], bbox[j+1]) for j in range(0, 8, 2)]
        layout_type = cell['type']
        color = dict_layout_type_to_color.get(layout_type, (0,128,0,128))

        # Set fill and outline colors (half-transparent for fills)
        fill_color = tuple(color[:3]) + (128,) if fill_bbox else None
        # Fully opaque outline if drawing border
        outline_color = tuple(color[:3]) + (255,) if draw_bbox else None

        # Draw filled polygon with outline
        draw.polygon(pts, outline=outline_color, fill=fill_color)

        # Draw index and category label, rotated along top edge
        order_cate = f"{i}_{layout_type}"
        text_color = tuple(color[:3]) + (255,)  # Opaque text for visibility
        # Estimate text size (compatible with new Pillow versions)
        try:
            bbox_text = font.getbbox(order_cate)
            text_w, text_h = bbox_text[2] - bbox_text[0], bbox_text[3] - bbox_text[1]
        except AttributeError:
            text_w, text_h = font.getsize(order_cate)
        w, h = text_w + 6, text_h + 6

        # Text is drawn on a small transparent image
        text_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        text_draw = ImageDraw.Draw(text_img)
        text_draw.text((3, 3), order_cate, font=font, fill=text_color)

        # Compute orientation of the top edge (left-top to right-top)
        dx = pts[1][0] - pts[0][0]
        dy = pts[1][1] - pts[0][1]
        angle = -math.degrees(math.atan2(dy, dx))  # Negative for counter-clockwise

        # Rotate label to align with top edge
        rotated_text_img = text_img.rotate(angle, expand=1)

        # Position label centered at right-top vertex
        x_anchor, y_anchor = pts[1]
        new_w, new_h = rotated_text_img.size
        paste_pos = (int(x_anchor - new_w // 2), int(y_anchor - new_h // 2))
        overlay.alpha_composite(rotated_text_img, dest=paste_pos)

    # Composite overlay onto image and convert to RGB
    result = Image.alpha_composite(image, overlay)
    result = result.convert("RGB")
    return result
