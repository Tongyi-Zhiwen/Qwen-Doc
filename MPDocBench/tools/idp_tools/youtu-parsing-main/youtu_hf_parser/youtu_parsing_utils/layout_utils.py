import re
from typing import List

def parse_layout_str(response_layout: str) -> List[List[str]]:
    """
    Parses a layout string and returns bounding box coordinates and layout type.
    Each line has: <x_{x1}><y_{y1}><x_{x2}><y_{y2}><LAYOUT_TYPE>
    Returns: [[x1, y1, x2, y2, layout_type], ...] with all strings.
    """
    type_map = {
        "<LAYOUT_TEXT>": "Text", "<LAYOUT_TITLE>": "Title",
        "<LAYOUT_HEADER>": "Header", "<LAYOUT_FOOTER>": "Footer",
        "<LAYOUT_FIGURE>": "Figure", "<LAYOUT_FORMULA>": "Formula",
        "<LAYOUT_TABLE>": "Table", "<LAYOUT_CODE>": "Code",
        "<LAYOUT_CAPTION>": "Caption", "<LAYOUT_CHART_DATA>": "Chart_Data", 
        "<LAYOUT_CHART_LOGIC>": "Chart_Logic", "<LAYOUT_SEAL>": "Seal",
        "<LAYOUT_FOOTER_FIGURE>": "Footer_Figure", "<LAYOUT_HEADER_FIGURE>": "Header_Figure",
        "<LAYOUT_SIGNATURE>": "Signature",
    }
    layout_types = "|".join(re.escape(t) for t in type_map)
    pattern = re.compile(
        rf'^<x_(\d+)><y_(\d+)><x_(\d+)><y_(\d+)>({layout_types})$'
    )
    return [
        [m.group(1), m.group(2), m.group(3), m.group(4), type_map[m.group(5)]]
        for line in response_layout.strip().split('\n')
        if (m := pattern.match(line))
    ]
