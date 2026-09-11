import os
import logging
from typing import Tuple, Optional, Dict, List
import cv2

logger = logging.getLogger(__name__)

LOCAL_HOSPITAL_VIDEO_PATH = os.path.abspath("src/assets/Hospital_OTMC_GF_OPD_Hospital_Hospital_20260618104602_20260618123051.mp4")

PRESET_TEST_STREAMS: List[Dict[str, str]] = [
    {
        "name": "Local Hospital OPD Camera Feed (MP4 Video)",
        "url": LOCAL_HOSPITAL_VIDEO_PATH if os.path.exists(LOCAL_HOSPITAL_VIDEO_PATH) else "src/assets/Hospital_OTMC_GF_OPD_Hospital_Hospital_20260618104602_20260618123051.mp4",
        "type": "local_file",
        "description": "Real recorded OPD Hospital Triage Camera footage"
    },
    {
        "name": "Webcam 0 (Default)",
        "url": "0",
        "type": "webcam",
        "description": "Default local camera feed"
    },
    {
        "name": "Pediatric Clinic Evaluation Video (YouTube)",
        "url": "https://www.youtube.com/watch?v=LXb3EKWsInQ",
        "type": "youtube",
        "description": "Real public pediatric clinic and child care test stream"
    },
    {
        "name": "Hospital Waiting Area Video (YouTube)",
        "url": "https://www.youtube.com/watch?v=jds9xHgKSaY",
        "type": "youtube",
        "description": "Real public hospital waiting room and patient triage test stream"
    }
]

class StreamResolver:
    """
    Utility for resolving user video input strings (YouTube URLs, direct video URLs,
    webcam indices, or local video files) into cv2.VideoCapture objects or source targets.
    """

    @staticmethod
    def is_youtube_url(url: str) -> bool:
        url_lower = url.lower().strip()
        return "youtube.com" in url_lower or "youtu.be" in url_lower

    @classmethod
    def resolve_stream_source(cls, source_input: str) -> Tuple[Optional[cv2.VideoCapture], str, Optional[str]]:
        """
        Resolves input string into (cv2_capture_object, display_label, error_message).
        
        - cv2_capture_object: cv2.VideoCapture instance ready for frame reading, or None if failed
        - display_label: Human readable label for UI
        - error_message: None if successful, string describing error if resolution failed
        """
        source_str = str(source_input).strip()

        if not source_str or source_str == "0":
            try:
                cap = cv2.VideoCapture(0)
                if cap and cap.isOpened():
                    return cap, "Webcam 0", None
            except Exception as e:
                logger.warning(f"Webcam 0 open failed: {e}")

            # Fallback to local hospital video file if webcam 0 is unavailable
            if os.path.exists(LOCAL_HOSPITAL_VIDEO_PATH):
                fallback_cap = cv2.VideoCapture(LOCAL_HOSPITAL_VIDEO_PATH)
                if fallback_cap and fallback_cap.isOpened():
                    filename = os.path.basename(LOCAL_HOSPITAL_VIDEO_PATH)
                    logger.info(f"Webcam 0 unavailable. Falling back to local hospital video feed: {filename}")
                    return fallback_cap, f"Hospital OPD Video ({filename})", None

            return None, "Webcam 0", "Could not open default camera (Webcam 0) or local fallback video."


        # Check numeric camera index (e.g. "1", "2")
        if source_str.isdigit():
            idx = int(source_str)
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                return cap, f"Camera {idx}", None
            return None, f"Camera {idx}", f"Could not open camera device {idx}."

        # Check YouTube URL
        if cls.is_youtube_url(source_str):
            try:
                from cap_from_youtube import cap_from_youtube
            except ImportError:
                logger.error("cap_from_youtube package is not installed.")
                return None, "YouTube Stream (Missing Dependency)", "cap_from_youtube package is required for YouTube streaming."

            try:
                cap = cap_from_youtube(source_str, resolution='360p')
                if cap and cap.isOpened():
                    display_title = "YouTube Video Stream"
                    for p in PRESET_TEST_STREAMS:
                        if p["url"] == source_str:
                            display_title = p["name"]
                            break
                    return cap, display_title, None
                else:
                    return None, "YouTube (Failed)", "Could not open YouTube stream into OpenCV."
            except Exception as e:
                err_msg = str(e)
                logger.error(f"Failed to extract YouTube stream for {source_str}: {err_msg}")
                return None, "YouTube (Error)", f"YouTube resolution error: {err_msg}"

        # Check local file path
        if os.path.exists(source_str):
            cap = cv2.VideoCapture(source_str)
            filename = os.path.basename(source_str)
            if cap.isOpened():
                return cap, f"Local File ({filename})", None
            return None, f"Local File ({filename})", f"Failed to open video file: {filename}"

        # Generic URL or video path attempt
        cap = cv2.VideoCapture(source_str)
        if cap.isOpened():
            display_name = source_str.split("/")[-1] or source_str[:20]
            return cap, f"Video Stream ({display_name})", None

        return None, "Invalid Source", f"Could not open video source: {source_str}"
