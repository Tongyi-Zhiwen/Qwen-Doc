import os
from PIL import Image


def is_valid_image_path(image_path):
    """
    Checks if the image path is valid.

    Args:
        image_path: The path to the image.

    Returns:
        bool: True if the path is valid, False otherwise.
    """
    if not os.path.exists(image_path):
        return False

    # Check if the file extension is one of the common image formats.
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    _, extension = os.path.splitext(image_path)
    if extension.lower() in image_extensions:
        return True
    else:
        return False


def read_image(image_path, use_native=False):
    """
    Reads an image and resizes it while maintaining aspect ratio.

    Args:
        image_path: The path to the image.
        use_native: If True, the max dimension of the original image is used as the max size. 
                    If False, max size is set to 1024.

    Returns:
        tuple: (resized_image, original_width, original_height)
    """
    # Create a default 512x512 blue image as a fallback.
    image = Image.new('RGB', (512, 512), color=(0, 0, 255))

    if is_valid_image_path(image_path):
        image = Image.open(image_path)
    else:
        raise FileNotFoundError(f"{image_path}: Image path does not exist")

    w, h = image.size
    if use_native:
        max_size = max(w, h)
    else:
        max_size = 1024
        
    if w > h:
        new_w = max_size
        new_h = int(h * max_size / w)
    else:
        new_h = max_size
        new_w = int(w * max_size / h)
        
    image = image.resize((new_w, new_h))
    return image, w, h
