import pytest
from PIL import Image, ImageDraw


@pytest.fixture(scope="session")
def simple_text_image() -> Image.Image:
    image = Image.new("RGB", (800, 600), "white")
    draw = ImageDraw.Draw(image)
    draw.text((50, 50), "Hello, World!", fill="black", font_size=32)
    return image
