import re
from typing import List, Dict
from .table_utils import convert_table_ostl_to_html

def parse_figtext_str(response_layout: str) -> List[Dict[str, str]]:
    """
    Parses a layout string and returns bounding box coordinates, layout type, and text content.
    Each entry has: <x_{x1}><y_{y1}><x_{x2}><y_{y2}><LAYOUT_TYPE>text_content
    Returns: [{'bbox': [x1, y1, x2, y2], 'type': layout_type, 'content': text_content}, ...]
    """
    # Define mapping from layout tags to human-readable type names
    type_map = {
        "<LAYOUT_TEXT>": "Text", "<LAYOUT_TITLE>": "Title",
        "<LAYOUT_HEADER>": "Header", "<LAYOUT_FOOTER>": "Footer",
        "<LAYOUT_FIGURE>": "Figure", "<LAYOUT_FORMULA>": "Formula",
        "<LAYOUT_TABLE>": "Table", "<LAYOUT_CODE>": "Code",
        "<LAYOUT_CAPTION>": "Caption", "<LAYOUT_CHART_DATA>": "Chart", "<LAYOUT_SEAL>": "Seal",
    }
    layout_types = "|".join(re.escape(t) for t in type_map)
    
    # Pattern to match the entire structure including text content with possible newlines
    pattern = re.compile(
        rf'<x_(\d+)><y_(\d+)><x_(\d+)><y_(\d+)>({layout_types})(.*?)(?=<x_\d+><y_\d+><x_\d+><y_\d+><LAYOUT_|$)',
        re.DOTALL
    )
    
    results = []
    
    # Find all matches in the input layout string
    matches = pattern.findall(response_layout)
    
    # Process each matched layout element
    for match in matches:
        # Unpack the captured groups: coordinates, layout type, and text content
        x1, y1, x2, y2, layout_type, text_content = match

        # otsl to html
        if layout_type == "<LAYOUT_TABLE>":
            text_content = convert_table_ostl_to_html(text_content)

        # Create result dictionary with bounding box, type, and content
        # Note: bbox format is [x1, y1, x2, y1, x2, y2, x1, y2] representing 4 corner points
        results.append({
            'bbox': [x1, y1, x2, y1, x2, y2, x1, y2],
            'type': type_map[layout_type],
            'content': text_content
        })
    
    return results
    