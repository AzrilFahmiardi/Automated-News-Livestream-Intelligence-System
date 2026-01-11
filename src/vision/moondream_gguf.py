"""
Moondream GGUF Processor

Fast vision language model using GGUF quantized model via llama.cpp.
Significantly faster than HuggingFace transformers version.
"""

import io
import logging
from pathlib import Path
from typing import Dict, Optional
from PIL import Image
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

logger = logging.getLogger(__name__)


class MoondreamGGUFProcessor:
    """
    Fast Moondream VLM using GGUF quantized model.
    
    Uses llama.cpp for efficient CPU inference with multimodal support.
    Expected to be 5-10x faster than HuggingFace version.
    """

    def __init__(self, config: dict):
        """
        Initialize Moondream GGUF processor.
        
        Args:
            config: System configuration dictionary
        """
        vision_config = config.get("vision", {})
        
        # Model paths
        self.text_model_path = Path(vision_config.get(
            "gguf_text_model", 
            "./models/moondream2-text-model-f16_ct-vicuna.gguf"
        ))
        self.mmproj_path = Path(vision_config.get(
            "gguf_mmproj", 
            "./models/moondream2-mmproj-f16-20250414.gguf"
        ))
        
        # Inference parameters
        self.n_ctx = vision_config.get("n_ctx", 2048)
        self.n_threads = vision_config.get("n_threads", 4)
        self.n_gpu_layers = vision_config.get("n_gpu_layers", 0)  # 0 for CPU
        
        # Prompt for ribbon extraction
        # Focus on main ribbon/lower-third, ignore running text/news ticker
        self.ribbon_prompt = vision_config.get(
            "ribbon_prompt",
            """Look at the lower part of this news broadcast image. 
Find the MAIN ribbon banner (lower-third graphic) - this is usually a colored bar with important text.
IGNORE any running text or news ticker (text that scrolls horizontally, often shows time/date).
The main ribbon can be:
1. Breaking news headline (e.g., "BREAKING NEWS | [headline]")
2. Speaker identification (e.g., "[Name] | [Title/Role]")
3. News topic label (e.g., "[Topic] | [Description]")

Extract ONLY the text from this MAIN ribbon banner. If there are multiple lines in the ribbon, include all of them.
Do NOT include running text, time, date, or scrolling tickers."""
        )
        
        self.confidence_threshold = vision_config.get("confidence_threshold", 0.7)
        
        # Initialize model
        logger.info("Loading Moondream GGUF model...")
        self._initialize_model()
        logger.info("Moondream GGUF processor initialized")

    def _initialize_model(self):
        """Initialize llama.cpp model with vision support."""
        if not self.text_model_path.exists():
            raise FileNotFoundError(f"Text model not found: {self.text_model_path}")
        if not self.mmproj_path.exists():
            raise FileNotFoundError(f"MMProj not found: {self.mmproj_path}")
        
        try:
            # Create chat handler with vision projector
            self.chat_handler = Llava15ChatHandler(
                clip_model_path=str(self.mmproj_path)
            )
            
            # Load text model
            self.model = Llama(
                model_path=str(self.text_model_path),
                chat_handler=self.chat_handler,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                logits_all=True,
                verbose=False
            )
            
            logger.info(f"Model loaded: {self.text_model_path.name}")
            logger.info(f"MMProj loaded: {self.mmproj_path.name}")
            logger.info(f"Context: {self.n_ctx}, Threads: {self.n_threads}")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise

    def process_frame(self, frame_data: bytes) -> Optional[Dict]:
        """
        Process video frame and extract ribbon text.
        
        Focus on main ribbon/lower-third banner only.
        Ignores running text, news ticker, time/date overlays.
        
        Args:
            frame_data: PNG image bytes
            
        Returns:
            Dictionary containing ribbon text and confidence
        """
        try:
            # Save frame temporarily for llama.cpp
            # (llama.cpp requires file path for images)
            temp_path = Path("/tmp/moondream_frame.png")
            temp_path.write_bytes(frame_data)
            
            # Extract main ribbon text only (no scene description)
            ribbon_text = self._query_vision(
                str(temp_path), 
                self.ribbon_prompt
            )
            
            # Clean up
            temp_path.unlink()
            
            # Simple confidence based on response length
            # (longer, more detailed responses = higher confidence)
            confidence = min(0.9, 0.5 + len(ribbon_text.split()) * 0.05)
            
            return {
                "ribbon_text": ribbon_text.strip(),
                "confidence": confidence,
                "method": "moondream_gguf"
            }
            
        except Exception as e:
            logger.error(f"ERROR: Frame processing failed: {e}")
            return None

    def _query_vision(self, image_path: str, prompt: str) -> str:
        """
        Query the vision model with an image and prompt.
        
        Args:
            image_path: Path to image file
            prompt: Question/instruction prompt
            
        Returns:
            Model's text response
        """
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"file://{image_path}"}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            response = self.model.create_chat_completion(
                messages=messages,
                max_tokens=256,
                temperature=0.1,
                top_p=0.9
            )
            
            answer = response["choices"][0]["message"]["content"]
            return answer
            
        except Exception as e:
            logger.error(f"Vision query failed: {e}")
            return ""

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'chat_handler'):
            del self.chat_handler
