"""
Segment Detection Module
Heuristic-based detection of news segment boundaries
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from collections import deque

logger = logging.getLogger(__name__)


class SegmentDetector:
    """Detects news segment boundaries using heuristic rules"""

    def __init__(self, config: dict):
        self.config = config
        segment_config = config.get("segment", {})

        self.min_duration = segment_config.get("min_duration", 30)  # seconds
        self.max_duration = segment_config.get("max_duration", 600)  # seconds
        self.idle_threshold = segment_config.get("idle_threshold", 10)  # seconds

        # State tracking
        self.current_segment = None
        self.ribbon_history = deque(maxlen=10)  # Last 10 ribbon detections
        self.last_change_time = None

        logger.info("Segment Detector initialized")

    def start_segment(self, channel: str) -> Dict:
        """
        Start a new segment

        Args:
            channel: Channel name

        Returns:
            dict: Segment metadata
        """
        segment_id = self._generate_segment_id(channel)

        self.current_segment = {
            "segment_id": segment_id,
            "channel": channel,
            "start_time": datetime.now(),
            "end_time": None,
            "ribbon_texts": [],
            "audio_chunks": [],
            "status": "active",
        }

        self.last_change_time = datetime.now()
        self.ribbon_history.clear()

        logger.info(f"Started new segment: {segment_id}")
        return self.current_segment

    def add_ribbon_detection(self, ribbon_data: Optional[Dict]) -> bool:
        """
        Add ribbon OCR detection to current segment

        Args:
            ribbon_data: OCR result dict

        Returns:
            bool: True if segment should continue, False if should end
        """
        if not self.current_segment:
            return False

        # Add to segment
        if ribbon_data and ribbon_data.get("text"):
            self.current_segment["ribbon_texts"].append(ribbon_data)
            self.ribbon_history.append(ribbon_data["text"])
            self.last_change_time = datetime.now()

        # Check if segment should end
        return not self._should_end_segment()

    def add_audio_chunk(self, audio_data: Dict):
        """
        Add audio transcription to current segment

        Args:
            audio_data: Whisper transcription result
        """
        if self.current_segment and audio_data:
            self.current_segment["audio_chunks"].append(audio_data)

    def _should_end_segment(self) -> bool:
        """
        Determine if current segment should end

        Returns:
            bool: True if segment should end
        """
        if not self.current_segment:
            return False

        now = datetime.now()
        duration = (now - self.current_segment["start_time"]).total_seconds()

        # Rule 1: Maximum duration exceeded
        if duration >= self.max_duration:
            logger.info(f"Segment ending: max duration reached ({duration}s)")
            return True

        # Rule 2: Minimum duration not reached
        if duration < self.min_duration:
            return False

        # Rule 3: Ribbon text changed significantly
        if self._detect_topic_change():
            logger.info("Segment ending: topic change detected")
            return True

        # Rule 4: Idle threshold (no changes for X seconds)
        if self.last_change_time:
            idle_time = (now - self.last_change_time).total_seconds()
            if idle_time >= self.idle_threshold:
                logger.info(f"Segment ending: idle for {idle_time}s")
                return True

        return False

    def _detect_topic_change(self) -> bool:
        """
        Detect if topic changed based on ribbon text history

        Returns:
            bool: True if topic likely changed
        """
        if len(self.ribbon_history) < 2:
            return False

        # Get last two unique ribbon texts
        unique_ribbons = list(dict.fromkeys(self.ribbon_history))

        if len(unique_ribbons) < 2:
            return False

        # Simple heuristic: if last ribbon is completely different
        last = unique_ribbons[-1].lower().strip()
        previous = unique_ribbons[-2].lower().strip()

        # Check if names/topics are different
        # This is a simple approach, can be improved with NLP
        common_words = set(last.split()) & set(previous.split())
        similarity = len(common_words) / max(len(last.split()), len(previous.split()))

        # If similarity < 30%, likely different topic
        return similarity < 0.3

    def end_segment(self) -> Optional[Dict]:
        """
        End current segment and return complete data

        Returns:
            dict: Complete segment data or None
        """
        if not self.current_segment:
            return None

        # Set end time
        self.current_segment["end_time"] = datetime.now()
        self.current_segment["status"] = "completed"

        # Calculate duration
        duration = (
            self.current_segment["end_time"] - self.current_segment["start_time"]
        ).total_seconds()
        self.current_segment["duration_sec"] = int(duration)

        logger.info(
            f"Ended segment {self.current_segment['segment_id']} "
            f"(duration: {duration}s, ribbons: {len(self.current_segment['ribbon_texts'])})"
        )

        # Return and clear
        segment = self.current_segment
        self.current_segment = None
        self.ribbon_history.clear()

        return segment

    def _generate_segment_id(self, channel: str) -> str:
        """
        Generate unique segment ID

        Args:
            channel: Channel name

        Returns:
            str: Segment ID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        channel_clean = channel.lower().replace(" ", "")
        return f"{channel_clean}_{timestamp}"

    def get_segment_progress(self) -> Optional[Dict]:
        """
        Get current segment progress info

        Returns:
            dict: Progress info or None
        """
        if not self.current_segment:
            return None

        duration = (datetime.now() - self.current_segment["start_time"]).total_seconds()

        return {
            "segment_id": self.current_segment["segment_id"],
            "duration": int(duration),
            "ribbon_count": len(self.current_segment["ribbon_texts"]),
            "audio_chunks": len(self.current_segment["audio_chunks"]),
            "status": self.current_segment["status"],
        }
