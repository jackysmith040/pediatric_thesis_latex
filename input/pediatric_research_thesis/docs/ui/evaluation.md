# `src/ui/evaluation.py`

## Purpose
`evaluation.py` implements the **Video Testing & Evaluation Lab** page (`/video-test`). It allows clinical operators and developers to evaluate detector accuracy across preset streams and custom video sources.

---

## Key Features

1. **Preset Stream Selector**: Preset buttons for preset streams defined in `StreamResolver.PRESET_TEST_STREAMS`:
   - Local OPD Triage MP4 video (`src/assets/...mp4`).
   - Webcam 0 (Default USB camera).
   - Pediatric Clinic evaluation stream (YouTube).
   - Hospital Waiting Room stream (YouTube).
2. **Custom URL Input**: Text input supporting arbitrary YouTube URLs, local video paths, or RTSP camera streams.
3. **Live Stream Preview**: Embedded preview displaying real-time detection boxes, active tracks, and telemetry counts.
4. **Source Activation**: Calls `detector.change_source(target_url)` asynchronously using `run.io_bound()` to update the active camera feed.
