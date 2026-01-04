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
        # Build context from ribbon texts
        ribbon_context = "\n".join(
            [f"- [{r['timestamp']}] {r['text']}" for r in ribbon_texts if r.get("text")]
        )

        # System prompt
        system_prompt = """Anda adalah AI assistant yang bertugas menganalisis berita TV Indonesia dan mengekstrak informasi terstruktur.

Tugas Anda:
1. Identifikasi SEMUA aktor/tokoh yang disebutkan (nama orang, organisasi, dll)
2. Tentukan peran/jabatan masing-masing aktor
3. Buat ringkasan berita (pendek dan lengkap)
4. Tentukan topik/kategori berita

Output HARUS dalam format JSON yang valid dengan struktur berikut:
{
  "actors": [
    {
      "name": "Nama lengkap aktor",
      "role": "Jabatan/peran",
      "source": ["ribbon" atau "speech"],
      "confidence": 0.0-1.0
    }
  ],
  "summary": {
    "short": "Ringkasan 1 kalimat",
    "full": "Ringkasan lengkap 2-3 kalimat"
  },
  "topics": ["topik1", "topik2"]
}

Gunakan Bahasa Indonesia yang formal dan akurat."""

        # User prompt with data
        user_prompt = f"""Analisis segmen berita berikut dari channel {channel}:

TRANSKRIP AUDIO:
{speech_text[:2000]}  

TEKS RIBBON/LOWER-THIRD:
{ribbon_context[:1000]}

Ekstrak informasi terstruktur dalam format JSON."""

        # Generate
        response = self.generate(user_prompt, system_prompt)

        # Parse JSON
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_text = response
            if "```json" in response:
                json_text = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_text = response.split("```")[1].split("```")[0].strip()

            data = json.loads(json_text)

            # Validate structure
            if not isinstance(data.get("actors"), list):
                data["actors"] = []

            if not isinstance(data.get("summary"), dict):
                data["summary"] = {"short": "", "full": ""}

            if not isinstance(data.get("topics"), list):
                data["topics"] = []

            return data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response was: {response}")

            # Return minimal structure
            return {
                "actors": [],
                "summary": {"short": "", "full": speech_text[:200]},
                "topics": [],
                "raw_llm_response": response,
            }

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
                # Merge sources
                seen[name]["source"] = list(
                    set(seen[name]["source"] + actor.get("source", []))
                )
                # Use higher confidence
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
