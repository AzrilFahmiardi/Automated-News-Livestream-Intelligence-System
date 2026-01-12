"""
Debug Pipeline Mode

Continuous capture and monitoring for component testing.
Focuses on YOLO ribbon detection quality and data collection.
Auto-segmentation is disabled to focus on component quality.
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

logger = logging.getLogger(__name__)


class DebugOrchestrator:
    """
    Debug mode pipeline for component testing and data collection.
    
    Features:
    - Continuous capture without auto-segmentation
    - Real-time console monitoring
    - Raw data export for frames, audio, and vision results
    - Quality metrics tracking
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

        self.is_running = False
        self.current_channel = None
        self.start_time = None

        self.stats = {
            "frames_captured": 0,
            "ribbon_detections": 0,
            "ribbon_changes": 0,
            "ribbon_disappeared": 0,
            "ocr_extractions": 0,
            "audio_chunks": 0,
            "audio_success": 0,
            "total_confidence": 0.0,
        }

        debug_config = config.get("debug", {})
        self.output_dir = Path(debug_config.get("output_dir", "./output/debug"))
        self.save_frames = debug_config.get("save_frames", True)
        self.save_audio = debug_config.get("save_audio", True)
        
        self.vision_history = []
        self.audio_history = []
        self.last_vision_result = None
        
        # Background processing queue
        self.vision_queue = asyncio.Queue()
        self.processing_tasks = []

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
        
        # Create session directory
        session_id = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_dir / f"{channel_name.lower()}_{session_id}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for outputs
        (self.session_dir / "audio").mkdir(exist_ok=True)
        (self.session_dir / "ocr").mkdir(exist_ok=True)
        (self.session_dir / "transcripts").mkdir(exist_ok=True)

        logger.info("=" * 80)
        logger.info("DEBUG MODE STARTED")
        logger.info(f"Channel: {channel_name}")
        logger.info(f"URL: {channel_url}")
        logger.info(f"Output: {self.session_dir}")
        logger.info("Auto-segmentation: DISABLED")
        logger.info("=" * 80)

        # Initialize browser
        await self.browser.initialize()

        # Open livestream
        success = await self.browser.open_livestream(channel_url)
        if not success:
            logger.error("ERROR: Failed to open livestream")
            return

        self.is_running = True

        # Start keep-alive task
        keep_alive_task = asyncio.create_task(self.browser.keep_alive())
        
        # Start monitoring display task
        monitor_task = asyncio.create_task(self._display_monitor())
        
        # YOLO ribbon processing
        worker_tasks = []

        # Start main debug loop
        try:
            await self._debug_loop(channel_name)
        except KeyboardInterrupt:
            logger.info("\nInterrupted by user")
        except Exception as e:
            logger.error(f"ERROR: Debug pipeline error: {e}", exc_info=True)
        finally:
            self.is_running = False
            keep_alive_task.cancel()
            monitor_task.cancel()
            for task in worker_tasks:
                task.cancel()
            await self.browser.close()
            
            # Save final report
            self._save_session_report()

        logger.info("Debug session stopped")

    async def _debug_loop(self, channel: str):
        """
        Main debug loop - continuous capture
        
        Args:
            channel: Channel name
        """
        video_config = self.config.get("video", {})
        audio_config = self.config.get("audio", {})

        frame_interval = 1.0 / video_config.get("fps_sample_rate", 0.5)  # seconds
        audio_interval = audio_config.get("chunk_duration", 30)  # seconds

        last_frame_time = 0
        last_audio_time = 0

        logger.info("Starting continuous capture loop...")
        logger.info(f"Frame capture: every {frame_interval:.1f}s")
        logger.info(f"Audio capture: every {audio_interval}s")

        while self.is_running:
            current_time = time.time()

            # Capture video frame 
            if current_time - last_frame_time >= frame_interval:
                logger.debug(f"[LOOP] Capturing frame at {current_time:.2f}")
                
                await self._capture_and_process_frame()
                
                logger.debug(f"[LOOP] Frame capture completed")
                last_frame_time = current_time

            # Process audio (placeholder)
            if current_time - last_audio_time >= audio_interval:
                await self._capture_and_process_audio()
                last_audio_time = current_time

            # Small delay
            await asyncio.sleep(0.1)

    async def _capture_and_process_frame(self):
        """
        Capture frame and process with YOLO ribbon detection
        """
        try:
            logger.debug(f"[CAPTURE] Starting frame capture")
            
            # Capture frame 
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
            logger.error(f"ERROR: Frame capture error: {e}", exc_info=True)
    
    async def _process_with_yolo_ribbon(self, frame_number: int, frame_data: bytes):
        """
        Process frame with YOLO ribbon detection + OCR.
        
        Only performs OCR when ribbon changes (new, moved, or disappeared).
        Runs in thread executor to avoid blocking.
        
        Args:
            frame_number: Sequential frame number
            frame_data: PNG image bytes
        """
        try:
            # Run YOLO + OCR in thread executor (blocking call)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.ribbon_detector.process_frame,
                frame_data
            )
            
            # Update detection stats
            self.stats["ribbon_detections"] += 1
            
            if result is None:
                # No change detected, skip saving
                return
            
            change_type = result.get("change_type")
            
            if change_type == "disappeared":
                self.stats["ribbon_disappeared"] += 1
                logger.info(f"Frame {frame_number}: Ribbon disappeared")
                return
            
            # Ribbon changed (new or position/text change)
            self.stats["ribbon_changes"] += 1
            self.stats["ocr_extractions"] += 1
            self.stats["total_confidence"] += result.get("confidence", 0.0)
            
            # Save OCR result
            ocr_filename = f"ribbon_{frame_number:06d}.json"
            ocr_path = self.session_dir / "ocr" / ocr_filename
            
            ocr_path.write_text(json.dumps({
                "text": result["text"],
                "bbox": result["bbox"],
                "confidence": result["confidence"],
                "change_type": change_type,
                "method": result["method"],
                "frame_number": frame_number
            }, indent=2))
            
            # Save annotated frame (with bounding box)
            if result.get("annotated_frame") is not None:
                import cv2
                frame_filename = f"ribbon_{frame_number:06d}.png"
                frame_path = self.session_dir / "ocr" / frame_filename
                cv2.imwrite(str(frame_path), result["annotated_frame"])
            
            # Display ribbon detection events
            text = result.get("text", "")
            if change_type == "new":
                logger.info(f"[RIBBON DETECTED] Frame {frame_number}: {text[:80]}")
            elif change_type == "changed":
                logger.info(f"[RIBBON CHANGED] Frame {frame_number}: {text[:80]}")
            elif change_type == "disappeared":
                logger.info(f"[RIBBON DISAPPEARED] Frame {frame_number}")
            else:
                logger.info(f"[RIBBON] Frame {frame_number} [{change_type}]: {text[:60]}")
                
        except Exception as e:
            logger.error(f"ERROR: YOLO ribbon processing failed for frame {frame_number}: {e}")
    
    async def _capture_and_process_audio(self):
        """Capture and process audio chunk (placeholder)"""
        try:
            # Note: Actual audio capture implementation would go here
            # For now, this is a placeholder
            
            self.stats["audio_chunks"] += 1
            
            logger.debug(f"DEBUG: Audio chunk {self.stats['audio_chunks']} (placeholder)")
            
        except Exception as e:
            logger.error(f"ERROR: Audio processing error: {e}")

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
            
            # Display
            print("\n" + "=" * 80)
            print("  LIVESTREAM INTELLIGENCE - DEBUG MODE (YOLO Ribbon)")
            print(f"  Channel: {self.current_channel.get('name', 'Unknown')} | Runtime: {runtime_str}")
            print("=" * 80)
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
            print("AUDIO (Whisper)")
            print(f"   Chunks Processed: {self.stats['audio_chunks']}")
            print(f"   Status: {'OK' if self.stats['audio_chunks'] > 0 else 'Waiting'}")
            print()
            print("DATA SAVED")
            print(f"   Output: {self.session_dir}")
            print(f"   Frames: {len(list((self.session_dir / 'frames').glob('*.png')))}")
            print(f"   Ribbons Saved: {len(list((self.session_dir / 'ocr').glob('ribbon_*.json')))}")
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
                    "success_rate": (
                        self.stats["vision_success"] / self.stats["frames_processed"] * 100
                        if self.stats["frames_processed"] > 0 else 0
                    ),
                    "avg_confidence": (
                        self.stats["total_confidence"] / self.stats["vision_success"]
                        if self.stats["vision_success"] > 0 else 0
                    ),
                },
                "output_directory": str(self.session_dir),
            }
            
            # Save report
            report_path = self.session_dir / "session_report.json"
            report_path.write_text(json.dumps(report, indent=2))
            
            logger.info(f"Session report saved: {report_path}")
            
            # Print summary
            print("\n" + "=" * 80)
            print("SESSION SUMMARY")
            print("=" * 80)
            print(f"Runtime: {self._format_duration(runtime)}")
            print(f"Frames Captured: {self.stats['frames_captured']}")
            print(f"Vision Success: {self.stats['vision_success']}")
            print(f"Success Rate: {report['metrics']['success_rate']:.1f}%")
            print(f"Avg Confidence: {report['metrics']['avg_confidence']:.2f}")
            print(f"Report: {report_path}")
            print("=" * 80)
            
        except Exception as e:
            logger.error(f"Failed to save session report: {e}")
