"""
LLM Reasoning Module
Uses llama.cpp to extract structured information from multimodal inputs
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class LlamaReasoning:
    """LLM-based reasoning and data extraction"""

    def __init__(self, config: dict):
        self.config = config
        llm_config = config.get("llm", {})

        model_path = Path(llm_config.get("model_path", "./models/qwen2.5-1.5b-instruct-q4_k_m.gguf"))

        if not model_path.exists():
            raise FileNotFoundError(
                f"LLM model not found: {model_path}\n"
                "Run: bash scripts/download_models.sh"
            )

        self.model_path = model_path
        self.context_size = llm_config.get("context_size", 4096)
        self.temperature = llm_config.get("temperature", 0.3)
        self.max_tokens = llm_config.get("max_tokens", 1024)
        self.n_threads = llm_config.get("n_threads", 4)

        # Initialize model
        self._load_model()

        logger.info(f"LLM Reasoning initialized with model: {model_path.name}")

    def _load_model(self):
        """Load llama.cpp model"""
        try:
            from llama_cpp import Llama

            self.llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.context_size,
                n_threads=self.n_threads,
                n_gpu_layers=0,  # CPU-only
                verbose=False,
            )

            logger.info("LLM model loaded successfully")

        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed.\n"
                "Install with: uv pip install llama-cpp-python"
            )
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            raise

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate text completion

        Args:
            prompt: User prompt
            system_prompt: System/instruction prompt

        Returns:
            str: Generated text
        """
        try:
            # Construct messages for chat format
            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({"role": "user", "content": prompt})

            # Generate
            response = self.llm.create_chat_completion(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # Extract text
            text = response["choices"][0]["message"]["content"]
            return text.strip()

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ""

    def _extract_json_from_response(self, response: str) -> str:
        """
        Extract and clean JSON from LLM response.
        Handles various formats: raw JSON, markdown code blocks, etc.
        
        Args:
            response: Raw LLM response
            
        Returns:
            str: Cleaned JSON string
        """
        import re
        
        text = response.strip()
        
        if "```json" in text:
            match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if match:
                text = match.group(1).strip()
        elif "```" in text:
            match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
            if match:
                text = match.group(1).strip()
        
        start = text.find('{')
        if start == -1:
            return text
            
        depth = 0
        end = start
        for i, char in enumerate(text[start:], start):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    end = i
                    break
        
        json_str = text[start:end+1]
        
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        return json_str

    def extract_news_segment(
        self, speech_text: str, ribbon_texts: List[Dict], channel: str
    ) -> Optional[Dict]:
        """
        Extract structured news segment from multimodal inputs

        Args:
            speech_text: Transcribed speech text
            ribbon_texts: List of ribbon OCR results with timestamps
            channel: Channel name

        Returns:
            dict: Structured news segment data
        """
        ribbon_context = "\n".join(
            [f"- [{r['timestamp']}] {r['text']}" for r in ribbon_texts if r.get("text")]
        )

        system_prompt = """Anda adalah AI yang menganalisis berita TV Indonesia. Output HARUS JSON valid.

FORMAT:
{"title":"JUDUL","actors":[],"summary":{"short":"1 kalimat","full":"2-3 kalimat"},"topics":["topik"]}

ATURAN:
- title: salin dari ribbon
- summary.short: ringkasan 1 kalimat dari transkrip
- summary.full: ringkasan 2-3 kalimat dari transkrip
- topics: 2-5 kata kunci"""

        has_transcript = bool(speech_text and speech_text.strip())
        
        transcript_text = speech_text[:1500] if has_transcript else ""
        
        first_ribbon = ""
        if ribbon_texts and len(ribbon_texts) > 0:
            first_ribbon = ribbon_texts[0].get("text", "")
        
        user_prompt = f"""Analisis berita:

JUDUL: {first_ribbon}

TRANSKRIP:
{transcript_text if has_transcript else "Tidak ada"}

Buat JSON dengan summary dari transkrip di atas."""

        logger.info(f"LLM input - has_transcript: {has_transcript}, transcript_len: {len(transcript_text)}")
        
        response = self.generate(user_prompt, system_prompt)
        
        logger.info(f"LLM raw response ({len(response)} chars): {response[:1000]}")

        try:
            json_text = self._extract_json_from_response(response)
            data = json.loads(json_text)
            
            logger.debug(f"Parsed JSON data: {json.dumps(data, ensure_ascii=False)[:500]}...")

            # Title 
            if "title" not in data or not data["title"]:
                if ribbon_texts and len(ribbon_texts) > 0:
                    data["title"] = ribbon_texts[0].get("text", "")
                else:
                    data["title"] = ""
            
            # Actors
            data["actors"] = self._clean_actors(data.get("actors"))
            
            summary = data.get("summary")
            if not isinstance(summary, dict):
                logger.warning(f"Summary is not a dict: {type(summary)} - {summary}")
                data["summary"] = {"short": None, "full": None}
            else:
                short_val = summary.get("short")
                full_val = summary.get("full")
                
                if isinstance(short_val, str) and short_val.strip():
                    short_val = short_val.strip()
                else:
                    short_val = None
                    
                if isinstance(full_val, str) and full_val.strip():
                    full_val = full_val.strip()
                else:
                    full_val = None
                
                data["summary"] = {"short": short_val, "full": full_val}
                
                if short_val is None and full_val is None and has_transcript:
                    logger.warning(f"LLM returned empty summary despite having transcript!")
                    logger.warning(f"Raw LLM response was: {response}")

            if not isinstance(data.get("topics"), list):
                data["topics"] = []

            return data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Full LLM response was:\n{response}")
            logger.error(f"Extracted JSON text was:\n{json_text}")

            title = ""
            if ribbon_texts and len(ribbon_texts) > 0:
                title = ribbon_texts[0].get("text", "")
            
            return {
                "title": title,
                "actors": None,
                "summary": {"short": None, "full": None},
                "topics": [],
                "raw_llm_response": response,
            }

    def _clean_actors(self, actors) -> Optional[List[Dict]]:
        """
        Clean and validate actors data.
        
        Returns None if no valid actors found, otherwise returns cleaned list.
        
        Args:
            actors: Raw actors data from LLM response
            
        Returns:
            List of valid actors or None
        """
        if not actors or not isinstance(actors, list):
            return None
        
        invalid_names = {"n/a", "na", "tidak ada", "unknown", "-", ""}
        
        valid_actors = []
        for actor in actors:
            if not isinstance(actor, dict):
                continue
            
            name = actor.get("name", "").strip()
            if not name or name.lower() in invalid_names:
                continue
            
            valid_actors.append({
                "name": name,
                "role": actor.get("role", "").strip() or None,
                "source": actor.get("source", "ribbon"),
                "confidence": actor.get("confidence", 0.5)
            })
        
        return valid_actors if valid_actors else None

    def normalize_actor_data(self, actors: List[Dict]) -> List[Dict]:
        """
        Normalize and deduplicate actor data

        Args:
            actors: List of actor dictionaries

        Returns:
            List[Dict]: Normalized actors
        """
        seen = {}

        for actor in actors:
            name = actor.get("name", "").strip().lower()
            if not name:
                continue

            if name in seen:
                seen[name]["source"] = list(
                    set(seen[name]["source"] + actor.get("source", []))
                )
                seen[name]["confidence"] = max(
                    seen[name]["confidence"], actor.get("confidence", 0.5)
                )
            else:
                seen[name] = {
                    "name": actor.get("name", "").strip(),
                    "role": actor.get("role", "").strip(),
                    "source": actor.get("source", []),
                    "confidence": actor.get("confidence", 0.5),
                }

        return list(seen.values())
