# `src/engine/detector.py`

## Purpose
`detector.py` provides the `Detector` class, which manages camera capture, YOLOv26 object detection, ByteTrack tracking, tracking trail visualization, spatial centroid fallback, and MJPEG frame generation.

---

## Key Architecture

### 1. Decoupled Dual-Thread Loop
* **Capture Thread (`_capture_loop`)**: Drains the OpenCV video stream constantly using `CAP_PROP_BUFFERSIZE = 1`, storing only the latest frame in `self.latest_frame`.
* **Inference Thread (`_inference_loop`)**: Pulls `self.latest_frame`, runs YOLOv26 inference (`track()`), draws bounding boxes and tracking trails, and computes JPEG bytes.

### 2. Untracked Spatial Centroid Resolution
Method `_resolve_track_id(box, track_id, class_id, claimed_ids)` handles untracked boxes (`track_id == -1`):
* Calculates box centroid $(cx, cy)$.
* Finds closest active centroid of matching `class_id` within `UNTRACKED_SPATIAL_MATCH_RADIUS`.
* Prevents stolen or duplicate IDs by maintaining a `claimed_ids` set per frame.
* Assigns a synthetic tracking ID if no match is found.

### 3. Dynamic Source Switching
Method `change_source(new_source: str) -> Tuple[bool, str]`:
* Thread-safe via `self.source_lock`.
* Uses `StreamResolver.resolve_stream_source(new_source)` to open new webcam/file/YouTube streams.
* Releases old `VideoCapture` and seamlessly redirects the capture thread without restarting the server.

### 4. Frame Streaming
Method `get_latest_jpeg_bytes() -> Optional[bytes]`:
* Returns the most recent annotated JPEG frame bytes for browser MJPEG streaming at `/camera/stream`.
