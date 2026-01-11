"""
Moondream Vision-Language Model Processor

Intelligent visual understanding using Moondream 2 VLM from HuggingFace.
Official documentation: https://huggingface.co/vikhyatk/moondream2
"""

import logging
import re
from datetime import datetime
from io import BytesIO
from typing import Dict, Optional

from PIL import Image

logger = logging.getLogger(__name__)


class MoondreamProcessor:
    """
    Vision-language processor using Moondream 2 model.
    
    Capabilities:
    - Visual querying with natural language
    - Lower-third text extraction
    - Scene understanding and analysis
    - Object detection and counting
    """

    def __init__(self, config: dict):
        """
        Initialize Moondream processor.
        
        Args:
            config: System configuration dictionary
        """
        self.config = config
        vision_config = config.get("vision", {})

        self.model_id = vision_config.get("model_id", "vikhyatk/moondream2")
        self.revision = vision_config.get("revision", "2025-06-21")
        self.device = vision_config.get("device", "cpu")
        self.trust_remote_code = vision_config.get("trust_remote_code", True)
        self.confidence_threshold = vision_config.get("confidence_threshold", 0.7)

        self.model = None
        self.tokenizer = None
        self._load_model()
        
        logger.info(
            f"Moondream processor initialized (model: {self.model_id}, "
            f"revision: {self.revision}, device: {self.device})"
        )

    def _load_model(self):
        """Load Moondream model and tokenizer from HuggingFace."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            logger.info(f"Loading Moondream model from HuggingFace...")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                revision=self.revision,
                trust_remote_code=self.trust_remote_code,
                device_map={"": self.device}
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                revision=self.revision,
                trust_remote_code=self.trust_remote_code
            )
            
            logger.info("Moondream model loaded successfully")
                
        except ImportError:
            logger.error(
                "Transformers library not found. "
                "Install: pip install transformers torch pillow"
            )
            self.model = None
            self.tokenizer = None
        except Exception as e:
            logger.error(f"Failed to load Moondream model: {e}")
            self.model = None
            self.tokenizer = None

    def process_frame(self, frame_data: bytes) -> Optional[Dict]:
        """
        Process video frame with Moondream VLM.
        
        Args:
            frame_data: Image data in PNG format as bytes
            
        Returns:
            Dictionary containing ribbon info and scene analysis, or None if processing fails
        """
        if self.model is None or self.tokenizer is None:
            return self._create_fallback_response()
        
        try:
            image = Image.open(BytesIO(frame_data))
            
            results = {
                "ribbon_info": self._extract_ribbon_info(image),
                "scene_analysis": self._analyze_scene(image),
                "timestamp": datetime.now().isoformat()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Frame processing failed: {e}")
            return None

    def _extract_ribbon_info(self, image: Image.Image) -> Dict:
        """
        Extract lower-third ribbon information from news broadcast frame.
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary with ribbon text, person name, role, and confidence
        """
        try:
            query = (
                "Look at the bottom of this news broadcast image. "
                "If there is a lower-third graphic showing text (usually name and title), "
                "extract it exactly as shown. If none exists, respond with 'none'. "
                "Common format is 'NAME | TITLE' or just text."
            )
            
            answer = self.model.query(image, query)["answer"]
            
            # Parse response
            has_ribbon = answer.lower() != "none" and len(answer.strip()) > 0
            text = answer.strip()
            
            person_name = ""
            person_role = ""
            
            # Parse NAME | ROLE format if present
            if "|" in text:
                parts = text.split("|", 1)
                person_name = parts[0].strip()
                person_role = parts[1].strip() if len(parts) > 1 else ""
            
            return {
                "has_ribbon": has_ribbon,
                "text": text,
                "person_name": person_name,
                "person_role": person_role,
                "confidence": 0.9 if has_ribbon else 0.1
            }
            
        except Exception as e:
            logger.error(f"Ribbon extraction failed: {e}")
            return self._create_empty_ribbon_info()

    def _analyze_scene(self, image: Image.Image) -> Dict:
        """
        Analyze scene type and context of news broadcast.
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary with scene type, people count, and breaking news status
        """
        try:
            query = (
                "Describe this news broadcast scene in one sentence. "
                "Is it a studio newscast, field report, interview, or press conference? "
                "How many people are visible? "
                "Is there a 'BREAKING NEWS' banner displayed?"
            )
            
            answer = self.model.query(image, query)["answer"]
            answer_lower = answer.lower()
            
            # Determine scene type
            scene_type = "unknown"
            if "studio" in answer_lower:
                scene_type = "studio"
            elif "field" in answer_lower or "outdoor" in answer_lower:
                scene_type = "field"
            elif "interview" in answer_lower:
                scene_type = "interview"
            elif "press conference" in answer_lower or "conference" in answer_lower:
                scene_type = "press_conference"
            
            # Extract people count
            numbers = re.findall(r'\d+', answer)
            people_count = int(numbers[0]) if numbers else 0
            
            # Check for breaking news
            breaking_news = "breaking" in answer_lower
            
            return {
                "scene_type": scene_type,
                "people_count": people_count,
                "logo_detected": None,
                "breaking_news": breaking_news,
                "confidence": 0.8,
                "description": answer
            }
            
        except Exception as e:
            logger.error(f"Scene analysis failed: {e}")
            return self._create_empty_scene_analysis()

    def _create_fallback_response(self) -> Dict:
        """
        Create fallback response when model is unavailable.
        
        Returns:
            Dictionary with empty data structure and fallback flag
        """
        logger.warning("Using fallback mode - Moondream model not loaded")
        
        return {
            "ribbon_info": self._create_empty_ribbon_info(),
            "scene_analysis": self._create_empty_scene_analysis(),
            "timestamp": datetime.now().isoformat(),
            "fallback_mode": True
        }

    def _create_empty_ribbon_info(self) -> Dict:
        """Create empty ribbon info structure."""
        return {
            "has_ribbon": False,
            "text": "",
            "person_name": "",
            "person_role": "",
            "confidence": 0.0
        }

    def _create_empty_scene_analysis(self) -> Dict:
        """Create empty scene analysis structure."""
        return {
            "scene_type": "unknown",
            "people_count": 0,
            "logo_detected": None,
            "breaking_news": False,
            "confidence": 0.0,
            "description": ""
        }

    def detect_visual_change(
        self, 
        current_result: Optional[Dict], 
        previous_result: Optional[Dict]
    ) -> bool:
        """
        Detect significant visual content change between frames.
        
        Args:
            current_result: Current frame analysis result
            previous_result: Previous frame analysis result
            
        Returns:
            True if significant change detected, False otherwise
        """
        if not current_result or not previous_result:
            return True
        
        # Compare ribbon text
        current_ribbon = current_result.get("ribbon_info", {}).get("text", "")
        previous_ribbon = previous_result.get("ribbon_info", {}).get("text", "")
        
        if current_ribbon != previous_ribbon:
            logger.debug(
                f"Visual change: ribbon text changed "
                f"('{previous_ribbon}' -> '{current_ribbon}')"
            )
            return True
        
        # Compare scene type
        current_scene = current_result.get("scene_analysis", {}).get("scene_type", "")
        previous_scene = previous_result.get("scene_analysis", {}).get("scene_type", "")
        
        if current_scene != previous_scene:
            logger.debug(
                f"Visual change: scene type changed "
                f"('{previous_scene}' -> '{current_scene}')"
            )
            return True
        
        return False

    def extract_actor_info(self, frame_result: Dict) -> Optional[Dict]:
        """
        Extract actor/person information from frame analysis.
        
        Args:
            frame_result: Complete frame analysis result from process_frame()
            
        Returns:
            Standardized actor information dictionary, or None if no actor found
        """
        ribbon_info = frame_result.get("ribbon_info", {})
        
        if not ribbon_info.get("has_ribbon"):
            return None
        
        name = ribbon_info.get("person_name", "")
        role = ribbon_info.get("person_role", "")
        
        if not name:
            return None
        
        return {
            "name": name,
            "role": role,
            "source": ["vision"],
            "confidence": ribbon_info.get("confidence", 0.0),
            "timestamp": frame_result.get("timestamp")
        }
