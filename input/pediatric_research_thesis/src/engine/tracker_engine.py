import time
import logging
import warnings
import cv2
import numpy as np
import supervision as sv
from typing import Tuple, Dict, Optional, Any
from src.engine.config import settings


logger = logging.getLogger(__name__)

SUPPORTED_TRACKERS = {
    "bytetrack": "ByteTrack (Fastest Baseline)",
    "botsort": "BoT-SORT (Camera Motion Comp.)",
    "ocsort": "OC-SORT (Non-Linear Motion)",
    "fasttracker": "FastTracker (Occlusion Aware)"
}

class SceneAnalyzer:
    """
    Analyzes live video frames and detection bounding boxes to estimate environmental
    conditions (camera motion, crowd occlusion density) and recommend optimal tracking algorithms.
    """

    @staticmethod
    def estimate_camera_motion(prev_gray: Optional[np.ndarray], current_gray: np.ndarray) -> float:
        """
        Estimates camera motion score using mean frame difference on downsampled grayscale frames.
        Returns a float score representing average pixel displacement.
        """
        if prev_gray is None or current_gray is None:
            return 0.0
        try:
            # Downsample for lightweight computation
            h, w = current_gray.shape[:2]
            target_w = 160
            target_h = int(h * (160.0 / max(1, w)))
            
            p_small = cv2.resize(prev_gray, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            c_small = cv2.resize(current_gray, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            
            diff = cv2.absdiff(p_small, c_small)
            return float(np.mean(diff))
        except Exception as e:
            logger.error(f"Error estimating camera motion: {e}")
            return 0.0

    @staticmethod
    def estimate_occlusion_density(boxes: np.ndarray) -> float:
        """
        Calculates bounding box overlap density (mean pairwise IoU) among detections.
        Returns a float between 0.0 (no overlap) and 1.0 (heavy crowding).
        """
        if boxes is None or len(boxes) < 2:
            return 0.0

        try:
            n = len(boxes)
            ious = []
            for i in range(n):
                for j in range(i + 1, n):
                    b1 = boxes[i]
                    b2 = boxes[j]

                    x1 = max(b1[0], b2[0])
                    y1 = max(b1[1], b2[1])
                    x2 = min(b1[2], b2[2])
                    y2 = min(b1[3], b2[3])

                    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                    if inter_area <= 0:
                        continue

                    b1_area = (b1[2] - b1[0]) * (b1[3] - b1[1])
                    b2_area = (b2[2] - b2[0]) * (b2[3] - b2[1])
                    union_area = b1_area + b2_area - inter_area
                    
                    if union_area > 0:
                        ious.append(inter_area / union_area)

            return float(np.mean(ious)) if ious else 0.0
        except Exception as e:
            logger.error(f"Error estimating occlusion density: {e}")
            return 0.0

    @classmethod
    def recommend_tracker(cls, motion_score: float, occlusion_score: float) -> str:
        """
        Recommends the best tracker based on estimated scene motion and occlusion score:
        - High Camera Motion -> BoT-SORT
        - High Occlusion / Crowding -> FastTracker
        - Stable / Low Motion -> ByteTrack (Default Baseline)
        """
        if motion_score > settings.CAMERA_MOTION_THRESHOLD:
            return "botsort"
        elif occlusion_score > settings.OCCLUSION_DENSITY_THRESHOLD:
            return "fasttracker"
        else:
            return "bytetrack"


class MultiTrackerEngine:
    """
    Manages active tracking algorithms, automatic situation-based switching with hysteresis stabilization,
    and manual tracker overrides.
    """

    def __init__(self, initial_mode: str = "auto"):
        self.mode = initial_mode.lower()  # "auto" or specific tracker key
        self.active_tracker_name = "bytetrack"
        
        # Instantiate Supervision trackers with specialized parameter profiles
        self._trackers = {
            "bytetrack": self._create_tracker_instance(
                track_thresh=settings.BYTETRACK_TRACK_THRESH,
                match_thresh=settings.BYTETRACK_MATCH_THRESH,
                lost_buffer=30
            ),
            "botsort": self._create_tracker_instance(
                track_thresh=max(0.20, settings.BYTETRACK_TRACK_THRESH - 0.05),
                match_thresh=0.70,
                lost_buffer=45  # Extended buffer for camera motion compensation
            ),
            "ocsort": self._create_tracker_instance(
                track_thresh=max(0.20, settings.BYTETRACK_TRACK_THRESH - 0.05),
                match_thresh=0.60,
                lost_buffer=30  # Non-linear motion recovery
            ),
            "fasttracker": self._create_tracker_instance(
                track_thresh=max(0.15, settings.BYTETRACK_TRACK_THRESH - 0.10),
                match_thresh=0.50,
                lost_buffer=60  # Occlusion-aware high density buffer
            )
        }
        
        self.prev_gray_frame: Optional[np.ndarray] = None
        self.last_switch_time: float = time.time()
        self.last_motion_score: float = 0.0
        self.last_occlusion_score: float = 0.0

    def _create_tracker_instance(self, track_thresh: float, match_thresh: float, lost_buffer: int = 30) -> sv.ByteTrack:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            return sv.ByteTrack(
                track_activation_threshold=float(track_thresh),
                minimum_matching_threshold=float(match_thresh),
                lost_track_buffer=int(lost_buffer),
                frame_rate=settings.BYTETRACK_FRAME_RATE
            )



    def set_mode(self, new_mode: str) -> Tuple[bool, str]:
        """Sets engine operating mode ('auto' or explicit tracker key)."""
        mode_lower = str(new_mode).lower().strip()
        if mode_lower == "auto":
            self.mode = "auto"
            return True, f"Automatic Situation Switching Enabled (Active: {SUPPORTED_TRACKERS.get(self.active_tracker_name, self.active_tracker_name)})"
        
        if mode_lower in SUPPORTED_TRACKERS:
            self.mode = mode_lower
            self.active_tracker_name = mode_lower
            return True, f"Manual Tracker Set to {SUPPORTED_TRACKERS[mode_lower]}"

        return False, f"Unknown tracker mode: {new_mode}"

    def update_with_detections(self, detections: sv.Detections, frame: Optional[np.ndarray] = None) -> Tuple[sv.Detections, str]:
        """
        Updates scene analysis, performs auto-switching (with hysteresis stabilization),
        and tracks detections using the active tracker instance.
        
        Returns (tracked_detections, tracker_status_label).
        """
        # Scene analysis & Auto-Switching logic
        if frame is not None:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                self.last_motion_score = SceneAnalyzer.estimate_camera_motion(self.prev_gray_frame, gray)
                self.prev_gray_frame = gray
            except Exception:
                pass

        boxes = detections.xyxy if detections is not None else None
        self.last_occlusion_score = SceneAnalyzer.estimate_occlusion_density(boxes)

        # Handle automatic tracker switching if in "auto" mode
        if self.mode == "auto":
            recommended = SceneAnalyzer.recommend_tracker(self.last_motion_score, self.last_occlusion_score)
            now = time.time()
            if recommended != self.active_tracker_name:
                if (now - self.last_switch_time) >= settings.AUTO_SWITCH_STABILIZATION_SECONDS:
                    logger.info(f"Auto-switching tracker from '{self.active_tracker_name}' to '{recommended}' (Motion: {self.last_motion_score:.1f}, Occlusion: {self.last_occlusion_score:.2f})")
                    self.active_tracker_name = recommended
                    self.last_switch_time = now

        active_tracker = self._trackers.get(self.active_tracker_name, self._trackers["bytetrack"])
        tracked_detections = active_tracker.update_with_detections(detections)

        tracker_title = SUPPORTED_TRACKERS.get(self.active_tracker_name, self.active_tracker_name.upper())
        status_suffix = "[Auto]" if self.mode == "auto" else "[Manual]"
        tracker_status_label = f"{tracker_title} {status_suffix}"

        return tracked_detections, tracker_status_label

    def get_info(self) -> Dict[str, Any]:
        """Returns structured metadata about the current tracking engine state for UI display."""
        tracker_title = SUPPORTED_TRACKERS.get(self.active_tracker_name, self.active_tracker_name.upper())
        return {
            "mode": self.mode,
            "active_tracker_key": self.active_tracker_name,
            "active_tracker_title": tracker_title,
            "status_label": f"{tracker_title} {'[Auto]' if self.mode == 'auto' else '[Manual]'}",
            "motion_score": round(self.last_motion_score, 2),
            "occlusion_score": round(self.last_occlusion_score, 2)
        }
