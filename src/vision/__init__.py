"""
Vision Understanding Module
Supports Moondream VLM (HuggingFace & GGUF), Tesseract OCR, and YOLO Ribbon Detection
"""

from .moondream import MoondreamProcessor
from .moondream_gguf import MoondreamGGUFProcessor
from .tesseract_ocr import TesseractProcessor
from .yolo_ribbon import YOLORibbonProcessor

__all__ = ["MoondreamProcessor", "MoondreamGGUFProcessor", "TesseractProcessor", "YOLORibbonProcessor"]
