"""
Pipeline Orchestrator
Main pipeline that coordinates all components for production mode.

Production mode processes livestreams with:
- YOLO ribbon detection + OCR
- Segment-based audio recording and transcription
- LLM content extraction
- JSON output to output/segments/
"""

import asyncio
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, Optional
import time

from ..browser import StreamCapturer
from ..vision import YOLORibbonProcessor
from ..audio import WhisperProcessor
from ..llm import LlamaReasoning
from ..segment import SegmentDetector, SegmentData

logger = logging.getLogger(__name__)


class NewsOrchestrator:
    """
    Production mode orchestrator for news livestream intelligence pipeline.
    
    Features:
    - State machine based segment detection
    - Segment-based audio recording (starts when segment starts, stops when ends)
    - Whisper transcription
    - LLM content extraction
    - JSON output generation
    """

    def __init__(self, config: dict):
        self.config = config

        # Initialize components
        logger.info("Initializing pipeline components...")

        self.browser = StreamCapturer(config)
        self.vision = YOLORibbonProcessor(config)
        self.whisper = WhisperProcessor(config)
        self.llm = LlamaReasoning(config)
        self.detector = SegmentDetector(config)

        # Pipeline state
        self.is_running = False
        self.current_channel = None
        
        # Audio recording state
        self.current_segment_audio_path: Optional[Path] = None
        self.temp_audio_dir = Path(tempfile.gettempdir()) / "news_intel_audio"
        self.temp_audio_dir.mkdir(parents=True, exist_ok=True)

        # Output configuration
        output_config = config.get("output", {})
        self.output_dir = Path(output_config.get("directory", "./output/segments"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Pipeline components initialized")

    async def start(self, channel_config: Dict):
        """
        Start processing a channel.

        Args:
            channel_config: Channel configuration dict
        """
        self.current_channel = channel_config
        channel_name = channel_config.get("name", "Unknown")
        channel_url = channel_config.get("url")

        logger.info(f"Starting pipeline for channel: {channel_name}")

        # Initialize browser
        await self.browser.initialize()

        # Open livestream
        success = await self.browser.open_livestream(channel_url)
        if not success:
            logger.error("Failed to open livestream")
            return

        self.is_running = True
        
        # Setup segment detector with channel
        self.detector.set_channel(channel_name)

        # Start keep-alive task
        keep_alive_task = asyncio.create_task(self.browser.keep_alive())

        # Start main processing loop
        try:
            await self._processing_loop(channel_name)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
        finally:
            self.is_running = False
            keep_alive_task.cancel()
            
            await self._stop_segment_audio_recording()
            await self.browser.close()

        logger.info("Pipeline stopped")

    async def _processing_loop(self, channel: str):
        """
        Main processing loop.

        Args:
            channel: Channel name
        """
        video_config = self.config.get("video", {})
        frame_interval = 1.0 / video_config.get("fps_sample_rate", 0.5)

        last_frame_time = 0

        logger.info("Starting processing loop...")
        logger.info(f"Frame capture interval: {frame_interval:.1f}s")
        logger.info("Audio recording: segment-based (starts with segment, stops when segment ends)")

        while self.is_running:
            current_time = time.time()

            if current_time - last_frame_time >= frame_interval:
                await self._capture_and_process_frame()
                last_frame_time = current_time

            await asyncio.sleep(0.1)

    async def _capture_and_process_frame(self):
        """
        Capture frame and process with YOLO ribbon detection.
        Handles segment events (start, end, discard).
        """
        try:
            # Capture frame
            frame_data = await self.browser.capture_frame()
            if not frame_data:
                return

            # YOLO + OCR ribbon processing
            loop = asyncio.get_event_loop()
            vision_result = await loop.run_in_executor(
                None,
                self.vision.process_frame,
                frame_data
            )

            # Process segment detection
            segment_event = self.detector.process_vision_result(vision_result)
            await self._handle_segment_event(segment_event, vision_result)

        except Exception as e:
            logger.error(f"Frame processing error: {e}")

    async def _handle_segment_event(self, event: Dict, vision_result: Optional[Dict]):
        """
        Handle segment detection events and manage audio recording lifecycle.
        
        Args:
            event: Segment event dict from SegmentDetector
            vision_result: Vision result for logging
        """
        action = event.get("action")
        
        if action == "start_segment":
            segment = event.get("segment")
            if segment:
                logger.info(f"[SEGMENT START] {segment.segment_id}")
                await self._start_segment_audio_recording(segment)
            
        elif action == "end_segment":
            segment = event.get("segment")
            if segment:
                logger.info(f"[SEGMENT END] {segment.segment_id} "
                           f"(duration={segment.duration_sec}s, ribbons={len(segment.ribbon_texts)})")
                
                audio_file = await self._stop_segment_audio_recording()
                if audio_file:
                    segment.audio_file_path = str(audio_file)
                
                asyncio.create_task(self._finalize_segment(segment))
            
        elif action == "discard":
            segment = event.get("segment")
            reason = event.get("reason", "unknown")
            if segment:
                logger.info(f"[SEGMENT DISCARD] {segment.segment_id} - {reason}")
            else:
                logger.info(f"[SEGMENT DISCARD] {reason}")
            await self._discard_segment_audio_recording()
            
        elif action == "ready":
            logger.info("[SEGMENT] Ready for new segments (cold start complete)")
            
        elif action == "skip":
            logger.debug(f"[SEGMENT] Skipping - {event.get('reason', 'cold_start')}")
        
        elif vision_result and vision_result.get("change_type") in ("new", "changed"):
            text = vision_result.get("text", "")[:60]
            logger.info(f"[RIBBON {vision_result.get('change_type').upper()}] {text}")
        elif vision_result and vision_result.get("change_type") == "disappeared":
            logger.info("[RIBBON] Disappeared")

    async def _start_segment_audio_recording(self, segment: SegmentData):
        """
        Start audio recording for a new segment.
        Audio is saved to a temp file that will be deleted after transcription.
        
        Args:
            segment: The segment that just started
        """
        try:
            audio_filename = f"{segment.segment_id}.wav"
            audio_path = self.temp_audio_dir / audio_filename
            
            recording_started = await self.browser.start_segment_audio_recording(str(audio_path))
            
            if recording_started:
                self.current_segment_audio_path = audio_path
                logger.info(f"[AUDIO] Started recording for segment {segment.segment_id}")
            else:
                logger.warning(f"[AUDIO] Failed to start recording for segment {segment.segment_id}")
                self.current_segment_audio_path = None
                
        except Exception as e:
            logger.error(f"[AUDIO] Error starting segment recording: {e}")
            self.current_segment_audio_path = None

    async def _stop_segment_audio_recording(self) -> Optional[Path]:
        """
        Stop the current segment audio recording.
        
        Returns:
            Path to the recorded audio file, or None if no recording was active
        """
        if self.current_segment_audio_path is None:
            return None
        
        try:
            audio_file = await self.browser.stop_audio_recording()
            
            if audio_file and audio_file.exists():
                file_size = audio_file.stat().st_size
                if file_size > 1000:  
                    logger.info(f"[AUDIO] Recording stopped: {audio_file.name} ({file_size:,} bytes)")
                    self.current_segment_audio_path = None
                    return audio_file
                else:
                    logger.warning(f"[AUDIO] Recording too small ({file_size} bytes), discarding")
                    audio_file.unlink(missing_ok=True)
            
            self.current_segment_audio_path = None
            return None
            
        except Exception as e:
            logger.error(f"[AUDIO] Error stopping segment recording: {e}")
            self.current_segment_audio_path = None
            return None

    async def _discard_segment_audio_recording(self):
        """
        Discard the current segment audio recording without processing.
        """
        if self.current_segment_audio_path is None:
            return
        
        try:
            await self.browser.stop_audio_recording()
            
            if self.current_segment_audio_path.exists():
                self.current_segment_audio_path.unlink(missing_ok=True)
                logger.info(f"[AUDIO] Discarded recording: {self.current_segment_audio_path.name}")
            
            self.current_segment_audio_path = None
            
        except Exception as e:
            logger.error(f"[AUDIO] Error discarding segment recording: {e}")
            self.current_segment_audio_path = None

    async def _transcribe_segment_audio(self, segment: SegmentData) -> str:
        """
        Transcribe the audio file for a segment.
        The audio file is deleted after transcription.
        
        Args:
            segment: Segment data with audio_file_path set
            
        Returns:
            Transcribed text, or empty string if transcription failed
        """
        try:
            if not segment.audio_file_path:
                return ""
                
            audio_path = Path(segment.audio_file_path)
            
            if not audio_path.exists():
                logger.warning(f"[TRANSCRIPTION] Audio file not found: {audio_path}")
                return ""
            
            file_size = audio_path.stat().st_size
            logger.info(f"[TRANSCRIPTION] Starting for {segment.segment_id} ({file_size:,} bytes)")
            
            loop = asyncio.get_event_loop()
            transcription_result = await loop.run_in_executor(
                None,
                self.whisper.transcribe_audio_file,
                str(audio_path)
            )
            
            try:
                audio_path.unlink(missing_ok=True)
                logger.debug(f"[AUDIO] Deleted temp file: {audio_path.name}")
            except Exception as e:
                logger.warning(f"[AUDIO] Failed to delete temp file: {e}")
            
            if transcription_result and transcription_result.get("text"):
                text = transcription_result["text"].strip()
                text_preview = text[:100] if len(text) > 100 else text
                logger.info(f"[TRANSCRIPTION] {segment.segment_id} SUCCESS: {text_preview}...")
                return text
            else:
                logger.warning(f"[TRANSCRIPTION] {segment.segment_id} returned empty result")
                return ""
                
        except Exception as e:
            logger.error(f"[TRANSCRIPTION] {segment.segment_id} FAILED: {e}")
            return ""

    async def _finalize_segment(self, segment: SegmentData):
        """
        Finalize segment: transcribe audio, run LLM processing, and save JSON output.
        
        Args:
            segment: Completed segment data
        """
        try:
            logger.info(f"[PROCESSING] Segment {segment.segment_id}...")
            
            # Transcribe audio
            speech_text = ""
            if segment.audio_file_path and Path(segment.audio_file_path).exists():
                speech_text = await self._transcribe_segment_audio(segment)
            
            if not speech_text:
                speech_text = "(No audio transcription available)"
            
            # Run LLM extraction
            logger.info(f"[LLM] Running extraction for segment {segment.segment_id}...")
            
            loop = asyncio.get_event_loop()
            content_data = await loop.run_in_executor(
                None,
                self.llm.extract_news_segment,
                speech_text,
                segment.ribbon_texts,
                segment.channel
            )
            
            # Build and save output
            output = self._build_output_json(segment, content_data, speech_text)
            self._save_segment(output)
            
            logger.info(f"[COMPLETE] Segment {segment.segment_id} saved successfully")
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to finalize segment {segment.segment_id}: {e}", exc_info=True)

    def _build_output_json(self, segment: SegmentData, content_data: Dict, speech_text: str = "") -> Dict:
        """
        Build final JSON output structure.

        Args:
            segment: Segment data
            content_data: LLM extracted content
            speech_text: Transcribed speech text

        Returns:
            dict: Final output structure
        """
        return {
            "segment": {
                "channel": segment.channel,
                "segment_id": segment.segment_id,
                "start_time": segment.start_time.isoformat(),
                "end_time": segment.end_time.isoformat() if segment.end_time else None,
                "duration_sec": segment.duration_sec,
            },
            "content": {
                "title": content_data.get("title", ""),
                "actors": content_data.get("actors"),
                "summary": content_data.get("summary", {"short": None, "full": None}),
                "topics": content_data.get("topics", []),
            },
            "raw": {
                "speech_text": speech_text,
                "ribbon_texts": segment.ribbon_texts,
            },
        }

    def _save_segment(self, output: Dict):
        """
        Save segment to JSON file.

        Args:
            output: Output data dict
        """
        try:
            segment_id = output["segment"]["segment_id"]
            filename = f"{segment_id}.json"
            filepath = self.output_dir / filename

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

            logger.info(f"[SAVED] {filepath}")

        except Exception as e:
            logger.error(f"[ERROR] Failed to save segment: {e}")

    async def stop(self):
        """Stop the pipeline gracefully"""
        logger.info("Stopping pipeline...")
        self.is_running = False

        await self._stop_segment_audio_recording()
        
        if self.detector.current_segment and self.current_channel:
            await self._finalize_segment(self.detector.current_segment)
