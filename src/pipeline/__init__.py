"""
Pipeline Orchestrator
Main pipeline that coordinates all components
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import time

from ..browser import StreamCapturer
from ..vision import MoondreamProcessor  
from ..audio import WhisperProcessor
from ..llm import LlamaReasoning
from ..segment import SegmentDetector

logger = logging.getLogger(__name__)


class NewsOrchestrator:
    """Main orchestrator for news livestream intelligence pipeline"""

    def __init__(self, config: dict):
        self.config = config

        # Initialize components
        logger.info("Initializing pipeline components...")

        self.browser = StreamCapturer(config)
        self.vision = MoondreamProcessor(config)  
        self.whisper = WhisperProcessor(config)
        self.llm = LlamaReasoning(config)
        self.detector = SegmentDetector(config)

        # Pipeline state
        self.is_running = False
        self.current_channel = None

        # Output configuration
        output_config = config.get("output", {})
        self.output_dir = Path(output_config.get("directory", "./output/segments"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Pipeline components initialized")

    async def start(self, channel_config: Dict):
        """
        Start processing a channel

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
            await self.browser.close()

        logger.info("Pipeline stopped")

    async def _processing_loop(self, channel: str):
        """
        Main processing loop

        Args:
            channel: Channel name
        """
        video_config = self.config.get("video", {})
        audio_config = self.config.get("audio", {})

        frame_interval = 1.0 / video_config.get("fps_sample_rate", 0.5)  # seconds
        audio_interval = audio_config.get("chunk_duration", 30)  # seconds

        last_frame_time = 0
        last_audio_time = 0
        previous_ribbon_text = None

        # Start first segment
        self.detector.start_segment(channel)

        logger.info("Starting processing loop...")

        while self.is_running:
            current_time = time.time()

            # Process video frame
            if current_time - last_frame_time >= frame_interval:
                await self._process_frame(previous_ribbon_text)
                last_frame_time = current_time

                # Get latest ribbon text
                progress = self.detector.get_segment_progress()
                if progress and progress["ribbon_count"] > 0:
                    segment_data = self.detector.current_segment
                    if segment_data and segment_data["ribbon_texts"]:
                        previous_ribbon_text = segment_data["ribbon_texts"][-1].get("text")

            # Process audio (periodically)
            if current_time - last_audio_time >= audio_interval:
                await self._process_audio()
                last_audio_time = current_time

            # Check if segment should end
            if self.detector._should_end_segment():
                await self._finalize_segment(channel)

            # Small delay to prevent tight loop
            await asyncio.sleep(0.1)

    async def _process_frame(self, previous_ribbon_text: Optional[str]):
        """Process single video frame with Moondream VLM"""
        try:
            # Capture frame
            frame_data = await self.browser.capture_frame()
            if not frame_data:
                return

            # Moondream VLM processing 
            vision_result = self.vision.process_frame(frame_data)

            # Extract ribbon info for compatibility with segment detector
            if vision_result:
                ribbon_info = vision_result.get("ribbon_info", {})
                # Convert to format expected by segment detector
                ribbon_result = {
                    "text": ribbon_info.get("text", ""),
                    "confidence": ribbon_info.get("confidence", 0.0),
                    "timestamp": vision_result.get("timestamp"),
                    "person_name": ribbon_info.get("person_name", ""),
                    "person_role": ribbon_info.get("person_role", ""),
                    "scene_type": vision_result.get("scene_analysis", {}).get("scene_type", ""),
                }
            else:
                ribbon_result = None

            # Add to segment
            should_continue = self.detector.add_ribbon_detection(ribbon_result)

            if ribbon_result and ribbon_result.get("text"):
                logger.debug(
                    f"Vision: {ribbon_result.get('text', '')[:50]} "
                    f"| Scene: {ribbon_result.get('scene_type', 'unknown')}"
                )

        except Exception as e:
            logger.error(f"Frame processing error: {e}")

    async def _process_audio(self):
        """Process audio chunk (placeholder for now)"""
        try:
            # Note: Actual audio capture from browser stream would go here
            # For now, this is a placeholder

            # Whisper transcription would happen here
            # audio_result = self.whisper.transcribe_audio_file(audio_path)
            # self.detector.add_audio_chunk(audio_result)

            logger.debug("Audio processing placeholder (not implemented yet)")

        except Exception as e:
            logger.error(f"Audio processing error: {e}")

    async def _finalize_segment(self, channel: str):
        """
        Finalize current segment and generate output

        Args:
            channel: Channel name
        """
        try:
            # End segment
            segment_data = self.detector.end_segment()
            if not segment_data:
                return

            logger.info(f"Finalizing segment: {segment_data['segment_id']}")

            # Combine audio text
            speech_text = " ".join(
                [chunk.get("text", "") for chunk in segment_data.get("audio_chunks", [])]
            )

            # If no audio, use placeholder
            if not speech_text:
                speech_text = "(No audio transcription available)"

            # LLM reasoning
            content_data = self.llm.extract_news_segment(
                speech_text=speech_text,
                ribbon_texts=segment_data.get("ribbon_texts", []),
                channel=channel,
            )

            # Build final output
            output = self._build_output_json(segment_data, content_data)

            # Save to file
            self._save_segment(output)

            logger.info(f"Segment saved: {segment_data['segment_id']}")

            # Start new segment
            self.detector.start_segment(channel)

        except Exception as e:
            logger.error(f"Segment finalization error: {e}", exc_info=True)

    def _build_output_json(self, segment_data: Dict, content_data: Dict) -> Dict:
        """
        Build final JSON output according to schema

        Args:
            segment_data: Raw segment data
            content_data: LLM-extracted content

        Returns:
            dict: Final output structure
        """
        return {
            "segment": {
                "channel": segment_data["channel"],
                "segment_id": segment_data["segment_id"],
                "start_time": segment_data["start_time"].isoformat(),
                "end_time": segment_data["end_time"].isoformat(),
                "duration_sec": segment_data["duration_sec"],
            },
            "content": {
                "actors": content_data.get("actors", []),
                "summary": content_data.get("summary", {"short": "", "full": ""}),
                "topics": content_data.get("topics", []),
            },
            "raw": {
                "speech_text": " ".join(
                    [chunk.get("text", "") for chunk in segment_data.get("audio_chunks", [])]
                ),
                "ribbon_texts": segment_data.get("ribbon_texts", []),
            },
        }

    def _save_segment(self, output: Dict):
        """
        Save segment to JSON file

        Args:
            output: Output data dict
        """
        try:
            segment_id = output["segment"]["segment_id"]
            filename = f"{segment_id}.json"
            filepath = self.output_dir / filename

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved segment to: {filepath}")

        except Exception as e:
            logger.error(f"Failed to save segment: {e}")

    async def stop(self):
        """Stop the pipeline gracefully"""
        logger.info("Stopping pipeline...")
        self.is_running = False

        # Finalize current segment if any
        if self.detector.current_segment:
            await self._finalize_segment(self.current_channel["name"])
