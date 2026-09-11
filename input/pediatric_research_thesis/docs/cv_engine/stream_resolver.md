# `src/engine/stream_resolver.py`

## Purpose
`stream_resolver.py` provides the `StreamResolver` utility class and preset test stream configurations (`PRESET_TEST_STREAMS`) to resolve user-specified video source inputs into valid `cv2.VideoCapture` instances.

---

## Supported Source Inputs

1. **Webcams**: Integer strings (e.g., `"0"`, `"1"`) mapped to local USB camera devices.
2. **Local Video Files**: Paths to local MP4/AVI files (e.g., `src/assets/Hospital_OTMC_GF_OPD_Hospital_Hospital_20260618104602_20260618123051.mp4`).
3. **YouTube Streams**: YouTube URLs (e.g., `https://www.youtube.com/watch?v=LXb3EKWsInQ`) resolved via `cap_from_youtube` at 360p resolution.
4. **RTSP / HTTP Video Streams**: Direct network stream URLs.

---

## Static Methods

### `is_youtube_url(url: str) -> bool`
Returns `True` if input URL string contains `youtube.com` or `youtu.be`.

### `resolve_stream_source(source_input: str) -> Tuple[Optional[cv2.VideoCapture], str, Optional[str]]`
Resolves source input string into:
- `cv2_capture_object`: OpenCV `VideoCapture` object ready for reading, or `None` if failed.
- `display_label`: Human-readable label for UI badges.
- `error_message`: Error description if resolution failed, `None` on success.
