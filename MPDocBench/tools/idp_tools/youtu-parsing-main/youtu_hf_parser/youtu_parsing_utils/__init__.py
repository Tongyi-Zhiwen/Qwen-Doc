"""
Youtu Parsing Utils

Utility functions for Youtu OCR Parser - document parsing and layout analysis.
"""

__version__ = "0.1.0"
__author__ = "Youtu Team"

# Import all utility functions for easy access
from .image_utils import (
    fitz_pdf_to_image,
    load_images_from_pdf,
    load_image
)

from .layout_utils import (
    parse_layout_str
)

from .table_utils import (
    parse_ostl_row_str,
    parse_ostl_table_str,
    convert_table_ostl_to_html
)

from .visualize_utils import (
    draw_layout_on_image,
    dict_layout_type_to_color
)

from .parser_utils import (
    normalize_label,
    box_to_str,
    clean_repeated_substrings,
    compute_image_scaling,
    parse_batch_results,
    categorize_layout_items,
    scale_bounding_boxes,
    parse_hierarchy,
    build_hierarchy_json
)

from .figtext_utils import (
    parse_figtext_str
)

from .prompt import (
    PROMPT_DICT
)

from .consts import (
    MIN_PIXELS,
    MAX_PIXELS,
    PDF_EXT,
    IMAGE_EXT
)

__all__ = [
    # Image utilities
    'fitz_pdf_to_image',
    'load_images_from_pdf', 
    'load_image',
    
    # Layout utilities
    'parse_layout_str',
    
    # Table utilities
    'parse_ostl_row_str',
    'parse_ostl_table_str',
    'convert_table_ostl_to_html',
    
    # Visualization utilities
    'draw_layout_on_image',
    'dict_layout_type_to_color',
    
    # Parser utilities
    'normalize_label',
    'box_to_str',
    'clean_repeated_substrings',
    'compute_image_scaling',
    'parse_batch_results',
    'categorize_layout_items',
    'scale_bounding_boxes',
    'parse_hierarchy',
    'build_hierarchy_json',
    
    # Figure text utilities
    'parse_figtext_str',
    
    # Prompts and constants
    'PROMPT_DICT',
    'MIN_PIXELS',
    'MAX_PIXELS',
    'PDF_EXT',
    'IMAGE_EXT',
]
