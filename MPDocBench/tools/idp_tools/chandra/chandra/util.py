from PIL import Image
from PIL.ImageDraw import ImageDraw

from chandra.output import LayoutBlock


def draw_layout(image: Image.Image, layout_blocks: list[LayoutBlock]):
    draw_image = image.copy()
    draw = ImageDraw(draw_image)
    for block in layout_blocks:
        if block.bbox[2] <= block.bbox[0] or block.bbox[3] <= block.bbox[1]:
            continue

        draw.rectangle(block.bbox, outline="red", width=2)
        draw.text((block.bbox[0], block.bbox[1]), block.label, fill="blue")

    return draw_image
