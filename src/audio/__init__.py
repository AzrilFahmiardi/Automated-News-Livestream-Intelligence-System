"""
Audio Processing Module
Speech-to-text using Whisper.cpp
"""

import logging
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

import numpy as np

logger = logging.getLogger(__name__)


class WhisperProcessor:
    """Processes audio stream and converts speech to text"""

    def __init__(self, config: dict):
        self.config = config
        audio_config = config.get("audio", {})

        self.chunk_duration = audio_config.get("chunk_duration", 30)
        self.sample_rate = audio_config.get("sample_rate", 16000)
        self.model_name = audio_config.get("whisper_model", "base")
        self.language = audio_config.get("language", "id")

        # Model path
        model_path = Path("./models") / f"ggml-{self.model_name}.bin"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Whisper model not found: {model_path}\n"
                "Run: bash scripts/download_models.sh"
            )

        self.model_path = model_path

        # Check if whisper.cpp CLI is available
        # Note: pywhispercpp is used, but we can also use CLI
        self._check_whisper_installation()

        logger.info(f"Whisper Processor initialized with model: {self.model_name}")

    def _check_whisper_installation(self):
        """Check if whisper.cpp is properly installed"""
        try:
            import pywhispercpp.model as whisper_model

            logger.info("pywhispercpp is available")
        except ImportError:
            logger.warning("pywhispercpp not found, will use CLI if available")

    def transcribe_audio_file(self, audio_path: str) -> Optional[Dict]:
        """
        Transcribe audio file to text

        Args:
            audio_path: Path to audio file (WAV format recommended)

        Returns:
            dict: Transcription result with text and metadata
        """
        try:
            # Using pywhispercpp
            from pywhispercpp.model import Model

            # Load model
            model = Model(str(self.model_path), n_threads=4)

            # Transcribe
            segments = model.transcribe(audio_path, language=self.language)

            # Combine all segments
            full_text = " ".join([seg.text.strip() for seg in segments])

            result = {
                "text": full_text.strip(),
                "language": self.language,
                "timestamp": datetime.now().isoformat(),
                "duration": self.chunk_duration,
            }

            logger.info(f"Transcribed {len(full_text)} characters")
            return result

        except ImportError:
            # Fallback to CLI
            logger.info("Using whisper.cpp CLI fallback")
            return self._transcribe_cli(audio_path)

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return None

    def _transcribe_cli(self, audio_path: str) -> Optional[Dict]:
        """
        Fallback: Use whisper.cpp CLI

        Args:
            audio_path: Path to audio file

        Returns:
            dict: Transcription result
        """
        try:
            # Try to find whisper executable
            whisper_cmd = "whisper-cpp"  # or "main" depending on installation

            # Run CLI
            cmd = [
                whisper_cmd,
                "-m",
                str(self.model_path),
                "-f",
                audio_path,
                "-l",
                self.language,
                "-otxt",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                text = result.stdout.strip()
                return {
                    "text": text,
                    "language": self.language,
                    "timestamp": datetime.now().isoformat(),
                }

            logger.error(f"CLI transcription failed: {result.stderr}")
            return None

        except Exception as e:
            logger.error(f"CLI transcription error: {e}")
            return None

    def process_audio_chunk(self, audio_data: bytes, format: str = "wav") -> Optional[Dict]:
        """
        Process audio chunk and transcribe

        Args:
            audio_data: Audio data bytes
            format: Audio format (wav, mp3, etc)

        Returns:
            dict: Transcription result
        """
        try:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as tmp_file:
                tmp_file.write(audio_data)
                tmp_path = tmp_file.name

            # Transcribe
            result = self.transcribe_audio_file(tmp_path)

            # Cleanup
            Path(tmp_path).unlink(missing_ok=True)

            return result

        except Exception as e:
            logger.error(f"Audio chunk processing failed: {e}")
            return None

    def extract_audio_from_video(self, video_path: str, output_path: str) -> bool:
        """
        Extract audio from video file using FFmpeg

        Args:
            video_path: Path to video file
            output_path: Path to save audio (WAV format)

        Returns:
            bool: True if successful
        """
        try:
            cmd = [
                "ffmpeg",
                "-i",
                video_path,
                "-vn",  # No video
                "-acodec",
                "pcm_s16le",  # PCM 16-bit
                "-ar",
                str(self.sample_rate),  # Sample rate
                "-ac",
                "1",  # Mono
                "-y",  # Overwrite
                output_path,
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30, check=True
            )

            logger.info(f"Audio extracted to {output_path}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg extraction failed: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Audio extraction error: {e}")
            return False
