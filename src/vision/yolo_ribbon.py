"""
YOLO Ribbon Detection + OCR Processor

Detects ribbon_main using fine-tuned YOLOv8n model and extracts text with EasyOCR.
Only performs OCR when ribbon changes (position or content).
"""

import io
import logging
import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import easyocr

logger = logging.getLogger(__name__)


class YOLORibbonProcessor:
    """
    Smart ribbon detection and OCR extraction.
    
    Features:
    - YOLOv8n detection for ribbon_main bounding box
    - EasyOCR for text extraction (Indonesian + English)
    - Change detection: only OCR when ribbon changes
    - Duplicate prevention: track unique ribbons by position + text
    """

    def __init__(self, config: dict):
        """
        Initialize YOLO + OCR processor.
        
        Args:
            config: System configuration dictionary
        """
        yolo_config = config.get("yolo_ribbon", {})
        
        # Model paths
        self.model_path = Path(yolo_config.get(
            "model_path",
            "./models/yolov8n_finetuned.pt"
        ))
        
        # Detection parameters
        self.conf_threshold = yolo_config.get("conf_threshold", 0.3)
        self.iou_threshold = yolo_config.get("iou_threshold", 0.4)
        
        # Change detection parameters
        self.position_change_threshold = yolo_config.get("position_change_threshold", 0.15)
        self.no_detection_frames = yolo_config.get("no_detection_frames", 3)
        self.stability_frames = yolo_config.get("stability_frames", 2)
        self.text_similarity_threshold = yolo_config.get("text_similarity_threshold", 0.85)
        self.animation_delay_frames = yolo_config.get("animation_delay_frames", 3)  # Wait for text animation
        
        # OCR parameters
        self.ocr_languages = yolo_config.get("ocr_languages", ['id', 'en'])
        
        # State tracking
        self.last_bbox = None
        self.last_text = None
        self.last_text_hash = None
        self.frames_without_detection = 0
        self.ribbon_present = False
        self.stable_bbox_count = 0
        self.pending_bbox = None
        self.animation_wait_count = 0  # Counter for animation delay
        
        # Initialize models
        logger.info("Loading YOLO ribbon detection model...")
        self._initialize_yolo()
        
        logger.info("Loading EasyOCR reader...")
        self._initialize_ocr()
        
        logger.info("YOLO Ribbon processor initialized")

    def _initialize_yolo(self):
        """Initialize YOLO model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {self.model_path}")
        
        try:
            self.yolo_model = YOLO(str(self.model_path))
            logger.info(f"YOLO model loaded: {self.model_path.name}")
            logger.info(f"Confidence threshold: {self.conf_threshold}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

    def _initialize_ocr(self):
        """Initialize EasyOCR reader."""
        try:
            self.ocr_reader = easyocr.Reader(
                self.ocr_languages,
                gpu=False,
                verbose=False
            )
            logger.info(f"EasyOCR loaded: {self.ocr_languages}")
        except Exception as e:
            logger.error(f"Failed to load EasyOCR: {e}")
            raise

    def process_frame(self, frame_data: bytes) -> Optional[Dict]:
        """
        Process video frame: detect ribbon and extract text if changed.
        
        Args:
            frame_data: PNG image bytes
            
        Returns:
            Dictionary with detection results or None if no change detected
        """
        try:
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                logger.error("Failed to decode frame")
                return None
            
            detection_result = self._detect_ribbon(frame)
            
            if detection_result is None:
                return self._handle_no_detection()
            
            bbox, confidence = detection_result
            
            # Check stability: is bbox consistent across frames?
            if self.pending_bbox is not None:
                iou_with_pending = self._calculate_iou(self.pending_bbox, bbox)
                if iou_with_pending >= self.iou_threshold:
                    self.stable_bbox_count += 1
                else:
                    self.stable_bbox_count = 0
                    self.pending_bbox = bbox
            else:
                self.pending_bbox = bbox
                self.stable_bbox_count = 0
            
            # Wait for stability before proceeding
            if self.stable_bbox_count < self.stability_frames:
                return None
            
            # Bbox is stable, now check if different from last saved
            is_new_ribbon = self.last_bbox is None
            is_position_changed = False
            
            if self.last_bbox is not None:
                iou = self._calculate_iou(self.last_bbox, bbox)
                
                if iou >= self.iou_threshold:
                    # Position unchanged, skip
                    return None
                else:
                    is_position_changed = True
            
            # Position changed significantly OR first detection
            # Wait for animation to complete before extracting text
            if is_new_ribbon or is_position_changed:
                self.animation_wait_count += 1
                
                if self.animation_wait_count < self.animation_delay_frames:
                    logger.debug(f"Waiting for animation ({self.animation_wait_count}/{self.animation_delay_frames})")
                    return None
            
            # Animation delay complete, now extract text
            ribbon_crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            text = self._extract_text(ribbon_crop)
            
            # Normalize text for comparison
            normalized_text = self._normalize_text(text)
            
            # Check text similarity with last saved
            if self.last_text is not None:
                similarity = self._text_similarity(normalized_text, self.last_text)
                
                if similarity >= self.text_similarity_threshold:
                    logger.debug(f"Text similarity {similarity:.2f}, skipping duplicate")
                    return None
            
            # Real change detected
            change_type = "new" if self.last_bbox is None else "changed"
            logger.info(f"Ribbon change detected: {change_type}")
            
            # Update state
            self.last_bbox = bbox
            self.last_text = normalized_text
            self.last_text_hash = self._hash_text(normalized_text)
            self.ribbon_present = True
            self.frames_without_detection = 0
            self.stable_bbox_count = 0
            self.pending_bbox = None
            self.animation_wait_count = 0  # Reset animation delay counter
            
            annotated_frame = self._annotate_frame(frame.copy(), bbox, text)
            
            return {
                "text": text,
                "bbox": bbox,
                "confidence": confidence,
                "change_type": change_type,
                "method": "yolo_ribbon_ocr",
                "annotated_frame": annotated_frame
            }
            
        except Exception as e:
            logger.error(f"Frame processing failed: {e}")
            return None

    def _detect_ribbon(self, frame: np.ndarray) -> Optional[Tuple[Tuple[int, int, int, int], float]]:
        """
        Detect ribbon_main bounding box using YOLO.
        
        Args:
            frame: OpenCV image (BGR)
            
        Returns:
            Tuple of (bbox, confidence) or None if not detected
        """
        try:
            results = self.yolo_model(frame, conf=self.conf_threshold, verbose=False)[0]
            
            if results.boxes is None or len(results.boxes) == 0:
                return None
            
            # Filter for ribbon_main class
            ribbon_main_boxes = []
            for box in results.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.yolo_model.names[cls_id]
                
                if cls_name == "ribbon_main":
                    ribbon_main_boxes.append(box)
            
            if len(ribbon_main_boxes) == 0:
                return None
            
            # Get highest confidence detection
            best_box = max(ribbon_main_boxes, key=lambda b: float(b.conf))
            
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            confidence = float(best_box.conf[0])
            
            return ((x1, y1, x2, y2), confidence)
            
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return None

    def _detect_change(self, bbox: Tuple[int, int, int, int]) -> Optional[str]:
        """
        Detect if ribbon changed significantly.
        
        Args:
            bbox: Current bounding box (x1, y1, x2, y2)
            
        Returns:
            Change type: "new", "position", or None if no change
        """
        if self.last_bbox is None:
            # First detection
            return "new"
        
        # Calculate IoU (Intersection over Union)
        iou = self._calculate_iou(self.last_bbox, bbox)
        
        if iou < self.iou_threshold:
            # Significant position/size change
            return "position"
        
        # Position similar, will check text later
        return "position"

    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """
        Calculate Intersection over Union between two bounding boxes.
        
        Args:
            bbox1: First bbox (x1, y1, x2, y2)
            bbox2: Second bbox (x1, y1, x2, y2)
            
        Returns:
            IoU value between 0 and 1
        """
        x1_inter = max(bbox1[0], bbox2[0])
        y1_inter = max(bbox1[1], bbox2[1])
        x2_inter = min(bbox1[2], bbox2[2])
        y2_inter = min(bbox1[3], bbox2[3])
        
        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        union_area = bbox1_area + bbox2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area

    def _extract_text(self, ribbon_crop: np.ndarray) -> str:
        """
        Extract text from ribbon crop using EasyOCR.
        
        Args:
            ribbon_crop: Cropped ribbon image (BGR)
            
        Returns:
            Extracted text
        """
        try:
            if ribbon_crop.size == 0:
                return ""
            
            # EasyOCR expects RGB
            ribbon_rgb = cv2.cvtColor(ribbon_crop, cv2.COLOR_BGR2RGB)
            
            # Run OCR
            ocr_results = self.ocr_reader.readtext(ribbon_rgb)
            
            # Combine all detected text
            texts = [res[1] for res in ocr_results]
            final_text = " ".join(texts).strip()
            
            return final_text
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return ""

    def _hash_text(self, text: str) -> str:
        """
        Create hash of text for comparison.
        
        Args:
            text: Input text
            
        Returns:
            MD5 hash of normalized text
        """
        # Normalize: lowercase, remove extra spaces
        normalized = " ".join(text.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison: remove noise, normalize whitespace.
        
        Args:
            text: Raw OCR text
            
        Returns:
            Normalized text for comparison
        """
        # Remove leading/trailing noise (single chars like 'r', 'i', etc)
        normalized = text.strip()
        
        # Remove leading single character if followed by uppercase
        if len(normalized) > 2 and normalized[0].islower() and normalized[1].isupper():
            normalized = normalized[1:]
        
        # Normalize whitespace: replace multiple spaces with single space
        normalized = ' '.join(normalized.split())
        
        # Normalize punctuation spacing: remove space before comma/period
        normalized = normalized.replace(' ,', ',').replace(' .', '.')
        
        # Convert to lowercase for comparison
        normalized = normalized.lower()
        
        return normalized
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity using Levenshtein distance.
        
        Args:
            text1: First text (normalized)
            text2: Second text (normalized)
            
        Returns:
            Similarity score from 0.0 to 1.0
        """
        if not text1 or not text2:
            return 0.0
        
        # Calculate Levenshtein distance
        len1, len2 = len(text1), len(text2)
        
        # Create matrix
        matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        
        # Initialize first column and row
        for i in range(len1 + 1):
            matrix[i][0] = i
        for j in range(len2 + 1):
            matrix[0][j] = j
        
        # Fill matrix
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if text1[i-1] == text2[j-1]:
                    cost = 0
                else:
                    cost = 1
                
                matrix[i][j] = min(
                    matrix[i-1][j] + 1,      # deletion
                    matrix[i][j-1] + 1,      # insertion
                    matrix[i-1][j-1] + cost  # substitution
                )
        
        distance = matrix[len1][len2]
        max_len = max(len1, len2)
        
        # Convert distance to similarity score
        similarity = 1.0 - (distance / max_len)
        
        return similarity

    def _handle_no_detection(self) -> Optional[Dict]:
        """
        Handle case when ribbon not detected.
        
        Returns:
            Result dict if ribbon disappeared, None otherwise
        """
        self.frames_without_detection += 1
        
        if self.ribbon_present and self.frames_without_detection >= self.no_detection_frames:
            # Ribbon disappeared (ad break or scene change)
            logger.info("Ribbon disappeared")
            
            self.ribbon_present = False
            self.last_bbox = None
            self.last_text = None
            self.last_text_hash = None
            self.animation_wait_count = 0  # Reset animation delay counter
            
            return {
                "text": "",
                "bbox": None,
                "confidence": 0.0,
                "change_type": "disappeared",
                "method": "yolo_ribbon_ocr",
                "annotated_frame": None
            }
        
        return None

    def _annotate_frame(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], text: str) -> np.ndarray:
        """
        Draw bounding box and text on frame for debugging.
        
        Args:
            frame: OpenCV image
            bbox: Bounding box (x1, y1, x2, y2)
            text: Detected text
            
        Returns:
            Annotated frame
        """
        x1, y1, x2, y2 = bbox
        
        # Draw green rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label background
        label = f"ribbon_main: {text[:30]}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 255, 0), -1)
        
        # Draw text
        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )
        
        return frame

    def reset(self):
        """Reset state tracking."""
        self.last_bbox = None
        self.last_text = None
        self.last_text_hash = None
        self.frames_without_detection = 0
        self.ribbon_present = False
        logger.info("State reset")
