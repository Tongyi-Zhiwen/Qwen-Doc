"""
Youtu HF Parser

HuggingFace-based OCR parser with direct model inference and angle correction.
This module provides local inference capabilities using HuggingFace transformers.
"""

__version__ = "0.1.0"
__author__ = "Youtu Team"

from .youtu_ocr_parser_hf import YoutuOCRParserHF

__all__ = ['YoutuOCRParserHF']
