"""
OCR Processing Module
Extracts text from lower-third/ribbon using Tesseract
"""

import logging
from datetime import datetime
from io import BytesIO
from typing import List, Dict, Optional

import cv2
import numpy as np
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)


class OCRProcessor:
    """Processes video frames and extracts text from lower-third/ribbon"""

    def __init__(self, config: dict):
        self.config = config
        ocr_config = config.get("ocr", {})

        self.engine = ocr_config.get("engine", "tesseract")
        self.languages = "+".join(ocr_config.get("languages", ["ind", "eng"]))
        self.confidence_threshold = ocr_config.get("confidence_threshold", 0.7)
        self.preprocessing = ocr_config.get("preprocessing", {})

        # ROI configuration
        video_config = config.get("video", {})
        self.roi = video_config.get("ribbon_roi", {})

        logger.info(f"OCR Processor initialized with engine: {self.engine}")

    def extract_ribbon_roi(self, frame_data: bytes) -> Optional[np.ndarray]:
        """
        Extract Region of Interest (lower-third area) from frame

        Args:
            frame_data: PNG image bytes

        Returns:
            np.ndarray: Cropped image or None if failed
        """
        try:
            # Convert bytes to image
            image = Image.open(BytesIO(frame_data))
            frame = np.array(image)

            # Get frame dimensions
            height, width = frame.shape[:2]

            # Calculate ROI coordinates
            x = int(width * self.roi.get("x", 0))
            y = int(height * self.roi.get("y", 0.8))
            w = int(width * self.roi.get("width", 1.0))
            h = int(height * self.roi.get("height", 0.2))

            # Crop ROI
            roi_frame = frame[y : y + h, x : x + w]

            return roi_frame

        except Exception as e:
            logger.error(f"Failed to extract ROI: {e}")
            return None

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR accuracy

        Args:
            image: Input image

        Returns:
            np.ndarray: Preprocessed image
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Apply preprocessing steps
            if self.preprocessing.get("denoise", True):
                # Denoise
                gray = cv2.fastNlMeansDenoising(gray, h=10)

            if self.preprocessing.get("contrast_enhance", True):
                # Enhance contrast using CLAHE
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)

            # Thresholding (adaptive)
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

            return binary

        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return image

    def extract_text(self, image: np.ndarray) -> Optional[Dict]:
        """
        Extract text from image using OCR

        Args:
            image: Input image (ROI)

        Returns:
            dict: Extracted text with metadata
        """
        try:
            # Preprocess
            processed = self.preprocess_image(image)

            # OCR configuration
            custom_config = f"--oem 3 --psm 6 -l {self.languages}"

            # Extract text with confidence
            data = pytesseract.image_to_data(
                processed, config=custom_config, output_type=pytesseract.Output.DICT
            )

            # Filter by confidence and combine text
            text_parts = []
            total_conf = 0
            conf_count = 0

            for i in range(len(data["text"])):
                text = data["text"][i].strip()
                conf = int(data["conf"][i])

                if text and conf > 0:
                    if conf / 100.0 >= self.confidence_threshold:
                        text_parts.append(text)
                        total_conf += conf
                        conf_count += 1

            if not text_parts:
                return None

            # Combine text
            full_text = " ".join(text_parts)
            avg_confidence = total_conf / conf_count / 100.0 if conf_count > 0 else 0

            return {
                "text": full_text,
                "confidence": round(avg_confidence, 2),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return None

    def process_frame(self, frame_data: bytes) -> Optional[Dict]:
        """
        Complete pipeline: extract ROI -> OCR

        Args:
            frame_data: PNG frame bytes

        Returns:
            dict: OCR result with text and metadata
        """
        # Extract ribbon area
        roi = self.extract_ribbon_roi(frame_data)
        if roi is None:
            return None

        # Extract text
        result = self.extract_text(roi)
        return result

    def detect_ribbon_change(
        self, current_text: Optional[str], previous_text: Optional[str]
    ) -> bool:
        """
        Detect if ribbon text has changed significantly

        Args:
            current_text: Current ribbon text
            previous_text: Previous ribbon text

        Returns:
            bool: True if changed
        """
        if not current_text or not previous_text:
            return True

        # Simple change detection (can be improved)
        return current_text.strip().lower() != previous_text.strip().lower()
