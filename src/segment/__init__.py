"""
Segment Detection Module

State machine based detection of news segment boundaries.
Handles cold start, segment recording, and segment finalization.
"""

import logging
from enum import Enum
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass, field

from ..utils import now_wib

logger = logging.getLogger(__name__)


class SegmentState(Enum):
    """Segment detector states"""
    COLD_START = "cold_start"
    IDLE = "idle"
    RECORDING = "recording"


@dataclass
class SegmentData:
    """Data container for a single news segment"""
    segment_id: str
    channel: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_sec: int = 0
    ribbon_texts: List[Dict] = field(default_factory=list)
    audio_chunks: List[Dict] = field(default_factory=list)
    audio_file_path: Optional[str] = None
    status: str = "active"


class SegmentDetector:
    """
    State machine based news segment detector.
    
    States:
        COLD_START: Program just started, skip first partial segment
        IDLE: No active segment, waiting for new ribbon
        RECORDING: Active segment, collecting data
    
    Transitions:
        COLD_START + ribbon_detected -> COLD_START (wait for disappear)
        COLD_START + ribbon_disappeared -> IDLE (ready for new segment)
        IDLE + ribbon_detected -> RECORDING (start new segment)
        IDLE + no_ribbon -> IDLE (stay idle)
        RECORDING + ribbon_detected -> RECORDING (add data)
        RECORDING + ribbon_disappeared -> IDLE (end segment)
    """

    def __init__(self, config: dict):
        """
        Initialize segment detector.
        
        Args:
            config: System configuration dictionary
        """
        self.config = config
        segment_config = config.get("segment", {})

        self.min_duration = segment_config.get("min_duration", 30)
        self.max_duration = segment_config.get("max_duration", 600)
        self.ribbon_disappear_threshold = segment_config.get("ribbon_disappear_threshold", 3)
        
        # Compilation detection thresholds
        self.compilation_min_ribbons = segment_config.get("compilation_min_ribbons", 3)
        self.compilation_max_interval = segment_config.get("compilation_max_interval", 25)  # seconds

        self.state = SegmentState.COLD_START
        self.current_segment: Optional[SegmentData] = None
        self.frames_without_ribbon = 0
        self.frames_without_ribbon_at_start = 0  
        self.cold_start_ribbon_seen = False

        logger.info(f"SegmentDetector initialized (state={self.state.value})")
        logger.info(f"Config: min_duration={self.min_duration}s, "
                   f"max_duration={self.max_duration}s, "
                   f"disappear_threshold={self.ribbon_disappear_threshold} frames")

    def process_vision_result(self, vision_result: Optional[Dict]) -> Dict:
        """
        Process vision result and determine segment action.
        
        Args:
            vision_result: YOLO+OCR result dict or None if no change/no ribbon
            
        Returns:
            dict: Action result with keys:
                - action: "none", "skip", "start_segment", "add_data", "end_segment"
                - segment: SegmentData if applicable
                - reason: Optional explanation
                
        Note:
            vision_result can be:
            - None: No change detected (ribbon same as before OR no ribbon during IDLE)
            - Dict with change_type="disappeared": Ribbon explicitly disappeared
            - Dict with change_type="new"/"changed": New/changed ribbon with text
        """
        if vision_result and vision_result.get("change_type") == "disappeared":
            return self._on_ribbon_disappeared()
        
        # Handle ribbon detected 
        if vision_result and vision_result.get("change_type") in ("new", "changed"):
            ribbon_data = {
                "text": vision_result.get("text", ""),
                "confidence": vision_result.get("confidence", 0.0),
                "timestamp": now_wib().isoformat(),
                "bbox": vision_result.get("bbox"),
                "change_type": vision_result.get("change_type", "unknown")
            }
            return self._on_ribbon_detected(ribbon_data)
        
        if self.state == SegmentState.RECORDING:
            self.frames_without_ribbon = 0
            return {"action": "none", "reason": "ribbon_unchanged"}
        
        return self._on_no_ribbon()

    def _on_ribbon_detected(self, ribbon_data: Dict) -> Dict:
        """Handle ribbon detection event (new or changed ribbon)."""
        self.frames_without_ribbon = 0

        if self.state == SegmentState.COLD_START:
            if not self.cold_start_ribbon_seen:
                self.cold_start_ribbon_seen = True
                
                if self.frames_without_ribbon_at_start >= self.ribbon_disappear_threshold:
                    logger.info("COLD_START (from ad break) -> RECORDING: Starting fresh segment")
                    self._start_segment(ribbon_data)
                    self.state = SegmentState.RECORDING
                    return {
                        "action": "start_segment",
                        "segment": self.current_segment,
                        "reason": "fresh_start_from_ad"
                    }
                else:
                    logger.info("COLD_START: Ribbon detected mid-segment, waiting for disappear")
                    return {"action": "skip", "reason": "cold_start_wait"}
            else:
                return {"action": "skip", "reason": "cold_start_wait"}

        elif self.state == SegmentState.IDLE:
            self._start_segment(ribbon_data)
            logger.info(f"IDLE -> RECORDING: Segment started ({self.current_segment.segment_id})")
            return {
                "action": "start_segment",
                "segment": self.current_segment,
                "reason": "new_ribbon"
            }

        elif self.state == SegmentState.RECORDING:
            self.current_segment.ribbon_texts.append(ribbon_data)
            
            if self._is_max_duration_exceeded():
                logger.info("Max duration exceeded, forcing segment end")
                return self._end_segment_and_return()
            
            return {"action": "add_data", "reason": "ribbon_added"}

        return {"action": "none"}

    def _on_no_ribbon(self) -> Dict:
        """Handle no ribbon detection event."""
        self.frames_without_ribbon += 1
        
        if self.state == SegmentState.COLD_START and not self.cold_start_ribbon_seen:
            self.frames_without_ribbon_at_start = self.frames_without_ribbon

        if self.frames_without_ribbon >= self.ribbon_disappear_threshold:
            return self._on_ribbon_disappeared()

        return {"action": "none", "reason": "waiting_for_threshold"}

    def _on_ribbon_disappeared(self) -> Dict:
        """Handle ribbon disappeared event (threshold reached)."""
        if self.state == SegmentState.COLD_START:
            if self.cold_start_ribbon_seen:
                self.state = SegmentState.IDLE
                logger.info("COLD_START -> IDLE: Ready for new segments")
                return {"action": "ready", "reason": "cold_start_complete"}
            else:
                logger.debug("COLD_START: No ribbon seen yet (possibly ad break)")
                return {"action": "none", "reason": "cold_start_no_ribbon"}

        elif self.state == SegmentState.RECORDING:
            return self._end_segment_and_return()

        elif self.state == SegmentState.IDLE:
            return {"action": "none", "reason": "already_idle"}

        return {"action": "none"}

    def _start_segment(self, ribbon_data: Dict):
        """Start a new segment."""
        channel = self.config.get("_current_channel", "unknown")
        segment_id = self._generate_segment_id(channel)

        self.current_segment = SegmentData(
            segment_id=segment_id,
            channel=channel,
            start_time=now_wib(),
            ribbon_texts=[ribbon_data],
            audio_chunks=[],
            status="active"
        )

        self.state = SegmentState.RECORDING
        self.frames_without_ribbon = 0

    def _end_segment_and_return(self) -> Dict:
        """End current segment and return result."""
        if not self.current_segment:
            self.state = SegmentState.IDLE
            return {"action": "none", "reason": "no_segment_to_end"}

        self.current_segment.end_time = now_wib()
        self.current_segment.duration_sec = int(
            (self.current_segment.end_time - self.current_segment.start_time).total_seconds()
        )
        self.current_segment.status = "completed"

        if self.current_segment.duration_sec < self.min_duration:
            logger.info(f"Segment too short ({self.current_segment.duration_sec}s), discarding")
            segment = self.current_segment
            self.current_segment = None
            self.state = SegmentState.IDLE
            return {"action": "discard", "segment": segment, "reason": "too_short"}

        if not self.current_segment.ribbon_texts:
            logger.info("Segment has no ribbon data, discarding")
            segment = self.current_segment
            self.current_segment = None
            self.state = SegmentState.IDLE
            return {"action": "discard", "segment": segment, "reason": "no_ribbons"}

        if self._is_compilation_segment():
            logger.info(f"Segment is a news compilation ({len(self.current_segment.ribbon_texts)} ribbons "
                       f"in {self.current_segment.duration_sec}s), discarding")
            segment = self.current_segment
            self.current_segment = None
            self.state = SegmentState.IDLE
            return {"action": "discard", "segment": segment, "reason": "compilation"}

        logger.info(f"RECORDING -> IDLE: Segment ended ({self.current_segment.segment_id}, "
                   f"duration={self.current_segment.duration_sec}s, "
                   f"ribbons={len(self.current_segment.ribbon_texts)})")

        segment = self.current_segment
        self.current_segment = None
        self.state = SegmentState.IDLE

        return {
            "action": "end_segment",
            "segment": segment,
            "reason": "ribbon_disappeared"
        }

    def _is_max_duration_exceeded(self) -> bool:
        """Check if current segment exceeded max duration."""
        if not self.current_segment:
            return False
        duration = (now_wib() - self.current_segment.start_time).total_seconds()
        return duration >= self.max_duration

    def _is_compilation_segment(self) -> bool:
        """
        Check if current segment is a news compilation/rundown.
        
        A compilation is detected when multiple different ribbons appear
        within a short time period (e.g., headline rundown at start of news).
        
        Returns:
            bool: True if segment appears to be a compilation
        """
        if not self.current_segment:
            return False
        
        ribbons = self.current_segment.ribbon_texts
        num_ribbons = len(ribbons)
        
        if num_ribbons < self.compilation_min_ribbons:
            return False
        
        if num_ribbons < 2:
            return False
        
        try:
            from datetime import datetime as dt
            
            timestamps = []
            for r in ribbons:
                ts_str = r.get("timestamp", "")
                if ts_str:
                    ts = dt.fromisoformat(ts_str)
                    timestamps.append(ts)
            
            if len(timestamps) < 2:
                return False
            
            intervals = []
            for i in range(1, len(timestamps)):
                interval = (timestamps[i] - timestamps[i-1]).total_seconds()
                intervals.append(interval)
            
            avg_interval = sum(intervals) / len(intervals)
            
            if avg_interval < self.compilation_max_interval:
                logger.debug(f"Compilation detected: {num_ribbons} ribbons, "
                           f"avg interval={avg_interval:.1f}s < {self.compilation_max_interval}s")
                return True
                
        except Exception as e:
            logger.warning(f"Error checking compilation: {e}")
        
        return False

    def _generate_segment_id(self, channel: str) -> str:
        """Generate unique segment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        channel_clean = channel.lower().replace(" ", "").replace("-", "")
        return f"{channel_clean}_{timestamp}"

    def add_audio_chunk(self, audio_data: Dict):
        """Add audio transcription to current segment."""
        if self.current_segment and audio_data:
            self.current_segment.audio_chunks.append(audio_data)
            logger.debug(f"Audio chunk added (total: {len(self.current_segment.audio_chunks)})")

    def get_state(self) -> str:
        """Get current state as string."""
        return self.state.value

    def get_segment_progress(self) -> Optional[Dict]:
        """Get current segment progress info."""
        if not self.current_segment:
            return None

        duration = (now_wib() - self.current_segment.start_time).total_seconds()

        return {
            "segment_id": self.current_segment.segment_id,
            "duration": int(duration),
            "ribbon_count": len(self.current_segment.ribbon_texts),
            "audio_chunks": len(self.current_segment.audio_chunks),
            "status": self.current_segment.status,
        }

    def set_channel(self, channel_name: str):
        """Set current channel name for segment ID generation."""
        self.config["_current_channel"] = channel_name

    def reset(self):
        """Reset detector to initial state."""
        self.state = SegmentState.COLD_START
        self.current_segment = None
        self.frames_without_ribbon = 0
        self.frames_without_ribbon_at_start = 0
        self.cold_start_ribbon_seen = False
        logger.info("SegmentDetector reset to COLD_START")
