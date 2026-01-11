"""
Tesseract OCR Processor

Fast text extraction from video frames using Tesseract OCR.
Primary use case: Extract ribbon/lower-third text from news livestreams.
"""

import io
import logging
from typing import Dict, Optional, Tuple
from PIL import Image, ImageEnhance
import pytesseract

logger = logging.getLogger(__name__)


class TesseractProcessor:
    """
    Fast OCR processor for ribbon text extraction.
    
    Uses Tesseract OCR engine to extract text from predefined regions.
    Significantly faster than VLM but requires manual ROI definition.
    """

    def __init__(self, config: dict):
        """
        Initialize Tesseract processor.
        
        Args:
            config: System configuration dictionary
        """
        ocr_config = config.get("ocr", {})
        
        # Region of Interest for ribbon (bottom banner)
        # Coordinates: (x1, y1, x2, y2) in pixels
        # Default assumes 1920x1080 video with ribbon at bottom
        self.ribbon_roi = ocr_config.get("ribbon_roi", {
            "x1": 0,
            "y1": 900,
            "x2": 1920,
            "y2": 1000
        })
        
        # OCR language (Indonesian)
        self.language = ocr_config.get("language", "ind+eng")
        
        # Confidence threshold for filtering low-quality results
        self.confidence_threshold = ocr_config.get("confidence_threshold", 0.5)
        
        # Image preprocessing options
        self.enable_preprocessing = ocr_config.get("preprocessing", True)
        self.contrast_factor = ocr_config.get("contrast_factor", 2.0)
        
        logger.info(f"Tesseract OCR initialized (lang: {self.language}, roi: {self.ribbon_roi})")

    def process_frame(self, frame_data: bytes) -> Optional[Dict]:
        """
        Process video frame and extract ribbon text.
        
        Args:
            frame_data: PNG image bytes
            
        Returns:
            Dictionary containing OCR results or None if failed
        """
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(frame_data))
            
            # Extract ribbon text
            ribbon_result = self._extract_ribbon_text(image)
            
            if not ribbon_result:
                return None
            
            return {
                "raw_text": ribbon_result["text"],
                "confidence": ribbon_result["confidence"],
                "method": "tesseract_ocr",
                "preprocessing_used": self.enable_preprocessing
            }
            
        except Exception as e:
            logger.error(f"ERROR: Tesseract processing failed: {e}")
            return None

    def _extract_ribbon_text(self, image: Image.Image) -> Optional[Dict]:
        """
        Extract text from ribbon region with preprocessing.
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary with text and confidence or None
        """
        try:
            # Crop to ribbon region
            ribbon_region = image.crop((
                self.ribbon_roi["x1"],
                self.ribbon_roi["y1"],
                self.ribbon_roi["x2"],
                self.ribbon_roi["y2"]
            ))
            
            # Preprocess image for better OCR accuracy
            if self.enable_preprocessing:
                ribbon_region = self._preprocess_image(ribbon_region)
            
            # Run Tesseract with detailed output
            ocr_data = pytesseract.image_to_data(
                ribbon_region,
                lang=self.language,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract text and calculate average confidence
            text_parts = []
            confidences = []
            
            for i, word in enumerate(ocr_data['text']):
                conf = float(ocr_data['conf'][i])
                if conf > 0 and word.strip():
                    text_parts.append(word)
                    confidences.append(conf)
            
            if not text_parts:
                return None
            
            full_text = " ".join(text_parts)
            avg_confidence = sum(confidences) / len(confidences) / 100.0
            
            # Filter by confidence threshold
            if avg_confidence < self.confidence_threshold:
                logger.debug(f"Low confidence result ({avg_confidence:.2f}): {full_text}")
                return None
            
            return {
                "text": full_text,
                "confidence": avg_confidence
            }
            
        except Exception as e:
            logger.error(f"ERROR: Ribbon extraction failed: {e}")
            return None

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image to improve OCR accuracy.
        
        Applies grayscale conversion and contrast enhancement.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Preprocessed PIL Image
        """
        # Convert to grayscale
        gray = image.convert('L')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(self.contrast_factor)
        
        return enhanced
