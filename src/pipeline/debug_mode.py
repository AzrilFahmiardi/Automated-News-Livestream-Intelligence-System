"""
Debug Pipeline Mode

Continuous capture and monitoring for component testing.
Includes segment detection and LLM processing for end-to-end testing.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import time

from ..browser import StreamCapturer
from ..vision import YOLORibbonProcessor
from ..audio import WhisperProcessor
from ..llm import LlamaReasoning
from ..segment import SegmentDetector, SegmentData

logger = logging.getLogger(__name__)


class DebugOrchestrator:
    """
    Debug mode pipeline for component testing and segment processing.
    
    Features:
    - State machine based segment detection
    - Real-time console monitoring
    - LLM processing for segment finalization
    - JSON output generation
    """

    def __init__(self, config: dict):
        """
        Initialize debug orchestrator.
        
        Args:
            config: System configuration dictionary
        """
        self.config = config

        logger.info("Initializing debug mode components...")

        self.browser = StreamCapturer(config)
        self.ribbon_detector = YOLORibbonProcessor(config)
        self.whisper = WhisperProcessor(config)
        self.llm = LlamaReasoning(config)
        self.segment_detector = SegmentDetector(config)

        self.is_running = False
        self.current_channel = None
        self.start_time = None

        self.stats = {
            "frames_captured": 0,
            "ribbon_detections": 0,
            "ribbon_changes": 0,
            "ribbon_disappeared": 0,
            "ocr_extractions": 0,
            "segments_completed": 0,
            "segments_discarded": 0,
            "segments_audio_recorded": 0,
            "segments_transcribed": 0,
            "total_confidence": 0.0,
            "total_audio_duration": 0.0,
        }

        debug_config = config.get("debug", {})
        self.output_dir = Path(debug_config.get("output_dir", "./output/debug"))
        self.segments_dir = Path(config.get("output", {}).get("directory", "./output/segments"))
        self.save_frames = debug_config.get("save_frames", True)
        self.save_audio = debug_config.get("save_audio", True)
        
        self.vision_history = []
        self.last_vision_result = None
        
        self.vision_queue = asyncio.Queue()
        self.processing_tasks = []
        self.current_segment_audio_path = None

        logger.info("Debug mode initialized successfully")

    async def start(self, channel_config: Dict):
        """
        Start debug capture session.
        
        Args:
            channel_config: Channel configuration dictionary
        """
        self.current_channel = channel_config
        channel_name = channel_config.get("name", "Unknown")
        channel_url = channel_config.get("url")

        self.start_time = datetime.now()
        
        # Setup segment detector with channel
        self.segment_detector.set_channel(channel_name)
        
        # Create session directory
        session_id = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_dir / f"{channel_name.lower()}_{session_id}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure segments output directory exists
        self.segments_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for outputs
        (self.session_dir / "audio").mkdir(exist_ok=True)
        (self.session_dir / "ocr").mkdir(exist_ok=True)
        (self.session_dir / "transcripts").mkdir(exist_ok=True)

        logger.info("=" * 80)
        logger.info("DEBUG MODE STARTED")
        logger.info(f"Channel: {channel_name}")
        logger.info(f"URL: {channel_url}")
        logger.info(f"Debug Output: {self.session_dir}")
        logger.info(f"Segments Output: {self.segments_dir}")
        logger.info(f"Segment Detection: ENABLED (state={self.segment_detector.get_state()})")
        logger.info("=" * 80)

        await self.browser.initialize()

        success = await self.browser.open_livestream(channel_url)
        if not success:
            logger.error("Failed to open livestream")
            return

        self.is_running = True

        keep_alive_task = asyncio.create_task(self.browser.keep_alive())
        monitor_task = asyncio.create_task(self._display_monitor())

        try:
            await self._debug_loop(channel_name)
        except KeyboardInterrupt:
            logger.info("\nInterrupted by user")
        except Exception as e:
            logger.error(f"Debug pipeline error: {e}", exc_info=True)
        finally:
            self.is_running = False
            keep_alive_task.cancel()
            monitor_task.cancel()
            
            await self._stop_segment_audio_recording()
            await self.browser.close()
            
            self._save_session_report()

        logger.info("Debug session stopped")

    async def _debug_loop(self, channel: str):
        """
        Main debug loop - continuous capture
        
        Args:
            channel: Channel name
        """
        video_config = self.config.get("video", {})

        frame_interval = 1.0 / video_config.get("fps_sample_rate", 0.5)

        last_frame_time = 0

        logger.info("Starting continuous capture loop...")
        logger.info(f"Frame capture: every {frame_interval:.1f}s")
        logger.info("Audio recording: segment-based (starts with segment, stops when segment ends)")

        while self.is_running:
            current_time = time.time()

            if current_time - last_frame_time >= frame_interval:
                logger.debug(f"[LOOP] Capturing frame at {current_time:.2f}")
                
                await self._capture_and_process_frame()
                
                logger.debug(f"[LOOP] Frame capture completed")
                last_frame_time = current_time

            await asyncio.sleep(0.1)

    async def _capture_and_process_frame(self):
        """
        Capture frame and process with YOLO ribbon detection
        """
        try:
            logger.debug(f"[CAPTURE] Starting frame capture")
            
            frame_data = await self.browser.capture_frame()
            self.stats["frames_captured"] += 1
            
            logger.debug(f"[CAPTURE] Frame {self.stats['frames_captured']} captured, size={len(frame_data) if frame_data else 0} bytes")
            
            if not frame_data:
                logger.debug("[CAPTURE] No frame data, returning")
                return

            frame_number = self.stats["frames_captured"]
            
            # Run YOLO + OCR for ribbon detection
            await self._process_with_yolo_ribbon(frame_number, frame_data)

        except Exception as e:
            logger.error(f"Frame capture error: {e}", exc_info=True)
    
    async def _process_with_yolo_ribbon(self, frame_number: int, frame_data: bytes):
        """
        Process frame with YOLO ribbon detection + OCR and segment detection.
        
        Args:
            frame_number: Sequential frame number
            frame_data: PNG image bytes
        """
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.ribbon_detector.process_frame,
                frame_data
            )
            
            self.stats["ribbon_detections"] += 1
            
            # Process segment detection
            segment_event = self.segment_detector.process_vision_result(result)
            await self._handle_segment_event(segment_event, frame_number)
            
            if result is None:
                return
            
            change_type = result.get("change_type")
            
            if change_type == "disappeared":
                self.stats["ribbon_disappeared"] += 1
                logger.info(f"Frame {frame_number}: Ribbon disappeared")
                return
            
            # Only save OCR result if we have valid ribbon data with text
            if "text" not in result:
                logger.debug(f"Frame {frame_number}: Result has no text field, skipping save")
                return
            
            self.stats["ribbon_changes"] += 1
            self.stats["ocr_extractions"] += 1
            self.stats["total_confidence"] += result.get("confidence", 0.0)
            
            # Save OCR result
            ocr_filename = f"ribbon_{frame_number:06d}.json"
            ocr_path = self.session_dir / "ocr" / ocr_filename
            
            ocr_path.write_text(json.dumps({
                "text": result.get("text", ""),
                "bbox": result.get("bbox"),
                "confidence": result.get("confidence", 0.0),
                "change_type": change_type,
                "method": result.get("method", "yolo_ribbon_ocr"),
                "frame_number": frame_number
            }, indent=2))
            
            if result.get("annotated_frame") is not None:
                import cv2
                frame_filename = f"ribbon_{frame_number:06d}.png"
                frame_path = self.session_dir / "ocr" / frame_filename
                cv2.imwrite(str(frame_path), result["annotated_frame"])
            
            text = result.get("text", "")
            if change_type == "new":
                logger.info(f"[RIBBON NEW] Frame {frame_number}: {text[:80]}")
            elif change_type == "changed":
                logger.info(f"[RIBBON CHANGED] Frame {frame_number}: {text[:80]}")
                
        except Exception as e:
            logger.error(f"YOLO ribbon processing failed for frame {frame_number}: {e}")

    async def _handle_segment_event(self, event: Dict, frame_number: int):
        """
        Handle segment detection events and manage segment audio recording.
        
        Args:
            event: Segment event dict from SegmentDetector
            frame_number: Current frame number
        """
        action = event.get("action")
        
        if action == "start_segment":
            segment = event.get("segment")
            logger.info(f"[SEGMENT START] {segment.segment_id} at frame {frame_number}")
            
            await self._start_segment_audio_recording(segment)
            
        elif action == "end_segment":
            segment = event.get("segment")
            logger.info(f"[SEGMENT END] {segment.segment_id} at frame {frame_number} "
                       f"(duration={segment.duration_sec}s, ribbons={len(segment.ribbon_texts)})")
            
            audio_file = await self._stop_segment_audio_recording()
            if audio_file:
                segment.audio_file_path = str(audio_file)
            
            asyncio.create_task(self._finalize_segment(segment))
            
        elif action == "discard":
            segment = event.get("segment")
            reason = event.get("reason", "unknown")
            logger.info(f"[SEGMENT DISCARD] {segment.segment_id} - {reason}")
            self.stats["segments_discarded"] += 1
            
            await self._discard_segment_audio_recording()
            
        elif action == "ready":
            logger.info(f"[SEGMENT] Ready for new segments (cold start complete)")
            
        elif action == "skip":
            logger.debug(f"[SEGMENT] Skipping - {event.get('reason', 'cold_start')}")

    async def _start_segment_audio_recording(self, segment: SegmentData):
        """
        Start audio recording for a new segment.
        
        Args:
            segment: The segment that just started
        """
        if not self.save_audio:
            return
        
        try:
            audio_filename = f"{segment.segment_id}.wav"
            audio_path = self.session_dir / "audio" / audio_filename
            
            recording_started = await self.browser.start_segment_audio_recording(str(audio_path))
            
            if recording_started:
                self.current_segment_audio_path = audio_path
                logger.info(f"[AUDIO] Started recording for segment {segment.segment_id}")
            else:
                logger.error(f"[AUDIO] Failed to start recording for segment {segment.segment_id}")
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
                    self.stats["segments_audio_recorded"] += 1
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
        Discard the current segment audio recording without saving.
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
        
        Args:
            segment: Segment data with audio_file_path set
            
        Returns:
            Transcribed text, or empty string if transcription failed
        """
        try:
            audio_path = Path(segment.audio_file_path)
            
            if not audio_path.exists():
                logger.warning(f"[TRANSCRIPTION] Audio file not found: {audio_path}")
                return ""
            
            file_size = audio_path.stat().st_size
            logger.info(f"[TRANSCRIPTION] Starting transcription for {segment.segment_id} ({file_size:,} bytes)")
            
            loop = asyncio.get_event_loop()
            transcription_result = await loop.run_in_executor(
                None,
                self.whisper.transcribe_audio_file,
                str(audio_path)
            )
            
            if transcription_result and transcription_result.get("text"):
                text = transcription_result["text"].strip()
                
                transcript_data = {
                    "segment_id": segment.segment_id,
                    "audio_file": audio_path.name,
                    "text": text,
                    "language": transcription_result.get("language", "id"),
                    "duration_sec": segment.duration_sec,
                    "timestamp": transcription_result.get("timestamp"),
                    "file_size_bytes": file_size
                }
                
                transcript_path = self.session_dir / "transcripts" / f"{segment.segment_id}.json"
                transcript_path.write_text(json.dumps(transcript_data, indent=2, ensure_ascii=False))
                
                segment.audio_chunks.append(transcript_data)
                
                self.stats["segments_transcribed"] += 1
                self.stats["total_audio_duration"] += segment.duration_sec
                
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
            logger.info(f"Processing segment {segment.segment_id}...")
            
            speech_text = ""
            
            if segment.audio_file_path and Path(segment.audio_file_path).exists():
                speech_text = await self._transcribe_segment_audio(segment)
            
            logger.info(f"Running LLM extraction for segment {segment.segment_id}...")
            
            loop = asyncio.get_event_loop()
            content_data = await loop.run_in_executor(
                None,
                self.llm.extract_news_segment,
                speech_text,
                segment.ribbon_texts,
                segment.channel
            )
            
            output = self._build_output_json(segment, content_data, speech_text)
            
            self._save_segment(output)
            
            self.stats["segments_completed"] += 1
            logger.info(f"Segment {segment.segment_id} saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to finalize segment {segment.segment_id}: {e}", exc_info=True)

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
            filepath = self.segments_dir / filename

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved segment to: {filepath}")

        except Exception as e:
            logger.error(f"Failed to save segment: {e}")
    
    async def _display_monitor(self):
        """Display real-time monitoring dashboard"""
        while self.is_running:
            await asyncio.sleep(5)
            
            runtime = (datetime.now() - self.start_time).total_seconds()
            runtime_str = self._format_duration(runtime)
            
            fps_actual = self.stats["frames_captured"] / runtime if runtime > 0 else 0
            
            avg_confidence = (
                self.stats["total_confidence"] / self.stats["ocr_extractions"]
                if self.stats["ocr_extractions"] > 0 else 0
            )
            
            transcription_rate = (
                self.stats["segments_transcribed"] / self.stats["segments_audio_recorded"] * 100
                if self.stats["segments_audio_recorded"] > 0 else 0
            )
            
            segment_state = self.segment_detector.get_state()
            segment_progress = self.segment_detector.get_segment_progress()
            
            is_recording_audio = self.current_segment_audio_path is not None
            
            print("\n" + "=" * 80)
            print("  LIVESTREAM INTELLIGENCE - DEBUG MODE")
            print(f"  Channel: {self.current_channel.get('name', 'Unknown')} | Runtime: {runtime_str}")
            print("=" * 80)
            print()
            print("SEGMENT DETECTION")
            print(f"   State: {segment_state.upper()}")
            if segment_progress:
                print(f"   Current Segment: {segment_progress['segment_id']}")
                print(f"   Duration: {segment_progress['duration']}s")
                print(f"   Ribbons: {segment_progress['ribbon_count']}")
                print(f"   Audio Recording: {'ACTIVE' if is_recording_audio else 'INACTIVE'}")
            print(f"   Segments Completed: {self.stats['segments_completed']}")
            print(f"   Segments Discarded: {self.stats['segments_discarded']}")
            print()
            print("RIBBON DETECTION (YOLO + EasyOCR)")
            print(f"   Frames Scanned: {self.stats['ribbon_detections']}")
            print(f"   Ribbon Changes: {self.stats['ribbon_changes']}")
            print(f"   Ribbon Disappeared: {self.stats['ribbon_disappeared']}")
            print(f"   OCR Extractions: {self.stats['ocr_extractions']}")
            print(f"   Avg Confidence: {avg_confidence:.2f}")
            print()
            print("CAPTURE")
            print(f"   Total Frames: {self.stats['frames_captured']}")
            print(f"   FPS (actual): {fps_actual:.2f}")
            print()
            print("SEGMENT AUDIO (Whisper)")
            print(f"   Segments Recorded: {self.stats['segments_audio_recorded']}")
            print(f"   Segments Transcribed: {self.stats['segments_transcribed']} ({transcription_rate:.1f}%)")
            print(f"   Total Audio Duration: {self._format_duration(self.stats['total_audio_duration'])}")
            print()
            print("OUTPUT")
            print(f"   Debug: {self.session_dir}")
            print(f"   Segments: {self.segments_dir}")
            print(f"   Ribbons Saved: {len(list((self.session_dir / 'ocr').glob('ribbon_*.json')))}")
            print(f"   Segment Files: {len(list(self.segments_dir.glob('*.json')))}")
            print()
            print("[Press Ctrl+C to stop]")
            print("=" * 80)

    def _format_duration(self, seconds: float) -> str:
        """Format duration in HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _save_session_report(self):
        """Save final session report"""
        try:
            runtime = (datetime.now() - self.start_time).total_seconds()
            
            avg_confidence = (
                self.stats["total_confidence"] / self.stats["ocr_extractions"]
                if self.stats["ocr_extractions"] > 0 else 0
            )
            
            transcription_rate = (
                self.stats["segments_transcribed"] / self.stats["segments_audio_recorded"] * 100
                if self.stats["segments_audio_recorded"] > 0 else 0
            )
            
            report = {
                "session": {
                    "channel": self.current_channel.get("name"),
                    "start_time": self.start_time.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "duration_sec": int(runtime),
                },
                "statistics": self.stats,
                "metrics": {
                    "fps_actual": self.stats["frames_captured"] / runtime if runtime > 0 else 0,
                    "ribbon_detection_rate": (
                        self.stats["ribbon_changes"] / self.stats["ribbon_detections"] * 100
                        if self.stats["ribbon_detections"] > 0 else 0
                    ),
                    "avg_ribbon_confidence": avg_confidence,
                    "segments_audio_recorded": self.stats["segments_audio_recorded"],
                    "transcription_rate": transcription_rate,
                    "total_audio_duration_sec": self.stats["total_audio_duration"],
                },
                "segment_detection": {
                    "segments_completed": self.stats["segments_completed"],
                    "segments_discarded": self.stats["segments_discarded"],
                    "final_state": self.segment_detector.get_state(),
                },
                "output_files": {
                    "ribbons": len(list((self.session_dir / "ocr").glob("ribbon_*.json"))),
                    "audio_files": len(list((self.session_dir / "audio").glob("*.wav"))),
                    "transcripts": len(list((self.session_dir / "transcripts").glob("*.json"))),
                    "segments": len(list(self.segments_dir.glob("*.json"))),
                },
                "output_directory": str(self.session_dir),
                "segments_directory": str(self.segments_dir),
            }
            
            report_path = self.session_dir / "session_report.json"
            report_path.write_text(json.dumps(report, indent=2))
            
            logger.info(f"Session report saved: {report_path}")
            
            print("\n" + "=" * 80)
            print("SESSION SUMMARY")
            print("=" * 80)
            print(f"Runtime: {self._format_duration(runtime)}")
            print()
            print("SEGMENTS:")
            print(f"  Completed: {self.stats['segments_completed']}")
            print(f"  Discarded: {self.stats['segments_discarded']}")
            print()
            print("VISION:")
            print(f"  Frames Captured: {self.stats['frames_captured']}")
            print(f"  Ribbon Changes: {self.stats['ribbon_changes']}")
            print(f"  OCR Extractions: {self.stats['ocr_extractions']}")
            print(f"  Avg Confidence: {avg_confidence:.2f}")
            print()
            print("SEGMENT AUDIO:")
            print(f"  Segments Recorded: {self.stats['segments_audio_recorded']}")
            print(f"  Segments Transcribed: {self.stats['segments_transcribed']} ({transcription_rate:.1f}%)")
            print(f"  Total Audio Duration: {self._format_duration(self.stats['total_audio_duration'])}")
            print()
            print("OUTPUT:")
            print(f"  Debug: {self.session_dir}")
            print(f"  Segments: {self.segments_dir}")
            print(f"  Segment Files: {report['output_files']['segments']}")
            print("=" * 80)
            
        except Exception as e:
            logger.error(f"Failed to save session report: {e}")

