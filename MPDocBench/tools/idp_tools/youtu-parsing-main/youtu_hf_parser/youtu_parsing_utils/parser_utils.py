"""
Parser utility functions for OCR processing.

This module contains common utility functions used across different
OCR parser implementations.
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Any, Optional


def normalize_label(label: str) -> str:
    """
    Normalize layout labels to standard format.
    
    Args:
        label: Layout label to normalize
        
    Returns:
        Normalized label string
    """
    if not label.startswith("<"):
        label = "<" + label + ">"
    
    # Map various text types to LAYOUT_TEXT for unified processing
    text_types = {
        "<LAYOUT_TEXT>", "<LAYOUT_HEADER>", "<LAYOUT_FOOTER>", 
        "<LAYOUT_FORMULA>", "<LAYOUT_CODE>", "<LAYOUT_CAPTION>", 
    }
    
    if label in text_types:
        return "<LAYOUT_TEXT>"
    return label


def box_to_str(bbox: List[float], layout_type: str) -> str:
    """
    Convert bounding box and layout type to string format.
    
    Args:
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        layout_type: Layout type label
        
    Returns:
        Formatted string representation
    """
    label = normalize_label(layout_type)
    return f'<x_{int(bbox[0])}><y_{int(bbox[1])}><x_{int(bbox[2])}><y_{int(bbox[3])}>{label}'


def clean_repeated_substrings(text: str) -> str:
    """
    Clean repeated substrings from text output.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text with excessive repetitions removed
    """
    n = len(text)
    if n < 8000:
        return text
    
    # Check for repeated patterns at the end of text
    for length in range(2, n // 10 + 1):
        candidate = text[-length:] 
        count = 0
        i = n - length
        
        # Count consecutive repetitions
        while i >= 0 and text[i:i + length] == candidate:
            count += 1
            i -= length

        # Remove excessive repetitions (keep only one)
        if count >= 10:
            return text[:n - length * (count - 1)]  

    return text


def compute_image_scaling(image_size: Tuple[int, int], 
                         processed_size: Tuple[int, int]) -> Tuple[float, float]:
    """
    Compute scaling factors between original and processed images.
    
    Args:
        image_size: Original image size (width, height)
        processed_size: Processed image size (width, height)
        
    Returns:
        Tuple of (scale_w, scale_h) scaling factors
    """
    raw_w, raw_h = image_size
    proc_w, proc_h = processed_size
    
    scale_w = float(raw_w) / proc_w
    scale_h = float(raw_h) / proc_h
    
    return scale_w, scale_h


def parse_batch_results(output_text: str, expected_count: int) -> List[str]:
    """
    Parse batch inference results.
    
    Args:
        output_text: Raw output text from batch inference
        expected_count: Expected number of results
        
    Returns:
        List of parsed result strings
    """
    if output_text:
        results = output_text.split("<sep>")
        # Ensure result count matches input count
        while len(results) < expected_count:
            results.append("")
        return results[:expected_count]
    else:
        return [""] * expected_count


def categorize_layout_items(layout_items: List[List]) -> Dict[str, List[int]]:
    """
    Categorize layout items by processing type.
    
    Args:
        layout_items: List of layout items with format [x1, y1, x2, y2, type]
        
    Returns:
        Dictionary mapping category names to lists of indices
    """
    categories = {
        'figure_indices': [],    
        'chart_indices': [],     
        'table_indices': [],     
        'ocr_indices': [],      
        'seal_indices': [],    
        'skip_indices': []   
    }
    
    for i, layout_item in enumerate(layout_items):
        x1, y1, x2, y2, layout_type = layout_item
        
        if layout_type == "Figure":
            categories['figure_indices'].append(i)
        elif layout_type in ["Chart_Data", "Chart_Logic"]:
            categories['chart_indices'].append(i)
        elif layout_type == "Table":
            categories['table_indices'].append(i)
        elif layout_type in ["Header_Figure", "Footer_Figure"]:
            categories['skip_indices'].append(i)
        elif layout_type == "Seal":
            categories['seal_indices'].append(i)
        else:
            # TEXT, HEADER, FOOTER, CODE, CAPTION, TITLE, FORMULA
            categories['ocr_indices'].append(i)
    
    return categories


def scale_bounding_boxes(result: List[Dict], 
                        scale_w: float, 
                        scale_h: float, 
                        image_size: Tuple[int, int]) -> None:
    """
    Scale bounding boxes to match original image size (modifies in-place).
    
    Args:
        result: List of result dictionaries containing 'bbox' key
        scale_w: Width scaling factor
        scale_h: Height scaling factor
        image_size: Original image size (width, height)
    """
    image_w, image_h = image_size
    
    for item in result:
        # Original bounding box coordinates
        x1, y1, x2, y2 = item["bbox"]
        
        # Scale coordinates
        x1 = np.floor(x1 * scale_w)
        y1 = np.floor(y1 * scale_h)
        x2 = np.ceil(x2 * scale_w)
        y2 = np.ceil(y2 * scale_h)
        
        # Clip to image boundaries
        x1 = max(0, min(image_w, x1))
        x2 = max(0, min(image_w, x2))
        y1 = max(0, min(image_h, y1))
        y2 = max(0, min(image_h, y2))

        # Convert to 8-point format (4 corners)
        item["bbox"] = [
            x1, y1,   # top-left
            x2, y1,   # top-right
            x2, y2,   # bottom-right
            x1, y2    # bottom-left
        ]


def parse_hierarchy(hierarchy_str: str, 
                   new_to_orig_idx: Dict[str, int] = None) -> Dict[int, 'TreeNode']:
    """
    Parse hierarchical relationship string and build a tree structure.
    
    Hierarchy string format: "002<<001, 003<<002, 004++003"
    - "<<" indicates child-parent relationship (e.g., 002 is child of 001)
    - "++" indicates sibling relationship (e.g., 004 is sibling of 003)
    - Sibling nodes inherit parent from their siblings (e.g., if 003 is child of 002, 
      and 004 is sibling of 003, then 004 is also child of 002)
    - Nodes without any relationship are isolated root nodes
    
    Args:
        hierarchy_str: Hierarchical relationship string
        new_to_orig_idx: Mapping from new index (str) to original index (int)
        
    Returns:
        Dictionary mapping original index (int) to TreeNode object
    """
    # Tree node class definition
    class TreeNode:
        """Represents a node in the hierarchy tree."""
        def __init__(self, node_id: int):
            self.id = node_id
            self.parent = None
            self.children = []
            self.level = 1
            self.content = ""
    
    # Handle edge cases
    if not new_to_orig_idx:
        return {}
    
    # If no hierarchy relationships exist, all nodes are root nodes
    if not hierarchy_str or not isinstance(hierarchy_str, str) or not hierarchy_str.strip():
        return {}
    
    # Parse relationship string
    relations = []
    for relation in hierarchy_str.split(','):
        relation = relation.strip()
        if not relation:
            continue
            
        if '<<' in relation:
            # Parent-child relationship: child << parent
            parts = relation.split('<<')
            if len(parts) == 2:
                child_id = parts[0].strip()
                parent_id = parts[1].strip()
                relations.append(('parent_child', child_id, parent_id))
        elif '++' in relation:
            # Sibling relationship: sibling1 ++ sibling2
            parts = relation.split('++')
            if len(parts) == 2:
                sibling1_id = parts[0].strip()
                sibling2_id = parts[1].strip()
                relations.append(('sibling', sibling1_id, sibling2_id))
    
    # Collect all node IDs involved in relationships
    all_new_node_ids = set()
    for rel_type, node1_id, node2_id in relations:
        all_new_node_ids.add(node1_id)
        all_new_node_ids.add(node2_id)
    
    # Create node dictionary using original indices as keys
    nodes = {}
    new_to_orig_map = {}
    
    for new_node_id in all_new_node_ids:
        orig_idx = new_to_orig_idx.get(new_node_id)
        if orig_idx is not None:
            nodes[orig_idx] = TreeNode(orig_idx)
            new_to_orig_map[new_node_id] = orig_idx
    
    # Build parent-child relationships
    for rel_type, node1_id, node2_id in relations:
        if rel_type == 'parent_child':
            child_orig_idx = new_to_orig_map.get(node1_id)
            parent_orig_idx = new_to_orig_map.get(node2_id)
            
            if child_orig_idx in nodes and parent_orig_idx in nodes:
                child_node = nodes[child_orig_idx]
                parent_node = nodes[parent_orig_idx]
                
                # If child already has a parent, remove it from the old parent's children first
                if child_node.parent is not None:
                    old_parent = child_node.parent
                    if child_node in old_parent.children:
                        old_parent.children.remove(child_node)
                
                child_node.parent = parent_node
                # Avoid duplicate children
                if child_node not in parent_node.children:
                    parent_node.children.append(child_node)
    
    # Process sibling relationships - siblings should share the same parent
    sibling_pairs = []
    for rel_type, node1_id, node2_id in relations:
        if rel_type == 'sibling':
            sibling1_orig_idx = new_to_orig_map.get(node1_id)
            sibling2_orig_idx = new_to_orig_map.get(node2_id)
            if sibling1_orig_idx is not None and sibling2_orig_idx is not None:
                sibling_pairs.append((sibling1_orig_idx, sibling2_orig_idx))
    
    # Assign parent nodes to siblings (iterate until no changes to handle chained siblings)
    changed = True
    max_iterations = len(sibling_pairs) + 1  # prevent infinite loop
    iteration = 0
    while changed and iteration < max_iterations:
        changed = False
        iteration += 1
        for sibling1_idx, sibling2_idx in sibling_pairs:
            if sibling1_idx in nodes and sibling2_idx in nodes:
                node1 = nodes[sibling1_idx]
                node2 = nodes[sibling2_idx]
                
                # If one sibling has a parent, assign it to the other
                if node1.parent and not node2.parent:
                    node2.parent = node1.parent
                    if node2 not in node1.parent.children:
                        node1.parent.children.append(node2)
                    changed = True
                elif node2.parent and not node1.parent:
                    node1.parent = node2.parent
                    if node1 not in node2.parent.children:
                        node2.parent.children.append(node1)
                    changed = True
    
    # Recursively calculate node levels with cycle detection
    def calculate_level(node: TreeNode, visited: set = None) -> int:
        """Calculate hierarchy level for a node with cycle detection."""
        if visited is None:
            visited = set()
        
        # Check for cycles
        if node.id in visited:
            # Cycle detected, treat as root node
            return 1
        
        if node.parent is None:
            return 1
        
        # Add current node to visited set
        visited.add(node.id)
        level = calculate_level(node.parent, visited) + 1
        # Remove current node from visited set for other paths
        visited.remove(node.id)
        
        return level
    
    # Compute levels for all nodes
    for node in nodes.values():
        node.level = calculate_level(node)
    
    return nodes


def build_hierarchy_json(hierarchy_str: str, 
                         result_items: List[Dict] = None,
                         new_to_orig_idx: Dict[str, int] = None) -> Dict[str, Any]:
    """
    Build hierarchy JSON with complete information including structure, content, and coordinates.
    Uses the same tree-building logic as parse_hierarchy to ensure correctness.
    
    Args:
        hierarchy_str: Hierarchical relationship string from model
        result_items: List of result items, each containing:
                     {"type": str, "bbox": [x1, y1, x2, y2], "content": str (optional)}
                     These should already have transformed coordinates (scaled and angle-corrected)
        new_to_orig_idx: Optional mapping from new index (str) to original index (int)
        
    Returns:
        Dictionary containing hierarchical tree structure:
        {
            "total_nodes": int,
            "root_count": int,
            "max_level": int,
            "nodes": [
                {
                    "id": int,
                    "level": int,
                    "content": str (optional),
                    "bbox": [x1, y1, x2, y2] (optional),
                    "type": str (optional),
                    "children": [...]
                }
            ]
        }
    """
    # Parse hierarchy using the correct tree-building logic from parse_hierarchy
    # This ensures proper parent-child and sibling relationships
    nodes = parse_hierarchy(hierarchy_str, new_to_orig_idx)
    
    if not nodes:
        return {
            "total_nodes": 0,
            "root_count": 0,
            "max_level": 0,
            "nodes": []
        }
    
    # Track visited nodes to prevent infinite recursion in node_to_dict
    visited_in_dict = set()
    
    # Helper function to recursively convert TreeNode to nested dictionary
    def node_to_dict(node) -> Dict[str, Any]:
        """
        Convert TreeNode to nested dictionary structure with cycle detection.
        This preserves the correct tree relationships built by parse_hierarchy:
        - Parent-child relationships (child << parent)
        - Sibling relationships (sibling1 ++ sibling2)
        - Correct level calculation
        """
        node_data = {
            "id": node.id,
            "level": node.level,
        }
        
        # Get data from result_items (which already has transformed coordinates)
        if result_items and 0 <= node.id < len(result_items):
            item = result_items[node.id]
            
            # Add content if available
            if "content" in item and item["content"]:
                node_data["content"] = item["content"]
            
            # Add bbox (already transformed)
            if "bbox" in item:
                node_data["bbox"] = item["bbox"]
            
            # Add type
            if "type" in item:
                node_data["type"] = item["type"]
        
        # Recursively convert children with cycle detection
        if node.id not in visited_in_dict:
            visited_in_dict.add(node.id)
            node_data["children"] = [node_to_dict(child) for child in node.children]
            visited_in_dict.discard(node.id)
        else:
            node_data["children"] = []
        
        return node_data
    
    # Find root nodes (nodes without parent) - these form the top level of the tree
    root_nodes = [node for node in nodes.values() if node.parent is None]
    
    # Build complete hierarchy JSON with nested tree structure
    hierarchy_json = {
        "total_nodes": len(nodes),
        "root_count": len(root_nodes),
        "max_level": max((node.level for node in nodes.values()), default=0),
        "nodes": [node_to_dict(root) for root in root_nodes]
    }
    
    return hierarchy_json
