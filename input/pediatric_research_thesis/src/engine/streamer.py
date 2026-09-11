from typing import Generator
from src.engine.detector import Detector

def mjpeg_generator(detector: Detector) -> Generator[bytes, None, None]:
    """
    Wraps the raw frame bytes from the detector into an MJPEG multipart response.
    """
    for frame_bytes in detector.get_frame_generator():
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )
