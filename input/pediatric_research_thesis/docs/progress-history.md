# 📈 Progress History & Project Changelog

Welcome to the historical log of **The Invisible Child**. This document tracks the technical evolution of the project from initial design to senior-stable delivery.

---

## Phase 1: Dual-Stack Prototype (Laravel + FastAPI)
* **Goal**: Build a pediatric monitoring system counting adults and children in hospital waiting areas.
* **Architecture**: Decoupled Python (FastAPI + YOLO) CV Engine sending telemetry to a Laravel 13 Command Center via HTTP and WebSockets (Laravel Reverb).
* **UI**: "Impeccable" clinical editorial aesthetic with Space Gray and Electric Blue styling.

---

## Phase 2: OpenCV Threading Optimization
* **Issue**: OpenCV's `VideoCapture` buffer caused severe stream lag (3-5 frame latency) when YOLO inference ran on the main loop.
* **Fix**: Implemented a decoupled dual-thread pattern:
  * **Capture Thread**: Drains OpenCV buffer (`CAP_PROP_BUFFERSIZE = 1`) continuously.
  * **Inference Thread**: Fetches latest frame asynchronously for YOLO inference and tracking overlay.

---

## Phase 3: The Monolith Pivot (NiceGUI)
* **Issue**: Dual-stack network overhead across localhost and complex multi-server deployment (Laravel + Reverb + Uvicorn).
* **Pivot**: Consolidated UI, state, and vision backend into a single **NiceGUI Python Monolith**.
* **Result**: Zero-network telemetry binding via `TelemetryState` Pydantic model, native browser MJPEG stream at `/camera/stream`, and simplified single-command execution.

---

## Phase 4: Spatial Centroid Tracking & Dynamic Source Switching
* **Issue**: Untracked detections (`track_id == -1`) caused temporary ID flickers when ByteTrack lost target continuity.
* **Fix**: Added spatial centroid fallback matching (`UNTRACKED_SPATIAL_MATCH_RADIUS = 40.0`) in `Detector._resolve_track_id()` to match untracked boxes to active centroids.
* **Feature**: Added thread-safe dynamic camera source switching (`Detector.change_source()`) allowing seamless camera stream changes without server restarts.

---

## Phase 5: Stream Resolver, Evaluation Suite & Unit Tests
* **Stream Resolver (`stream_resolver.py`)**: Support for multi-source inputs (Webcam 0, YouTube clinical streams via `cap_from_youtube`, local MP4 files, RTSP streams).
* **Evaluation Lab (`/video-test`)**: NiceGUI page enabling live testing across preset hospital triage feeds and custom URLs.
* **PyWebView Native Download Support**: Detected native desktop mode and routed CSV/PDF report downloads directly to local `Downloads/` directory.
* **Automated Unit Testing (`tests/unit/`)**: Comprehensive Pytest suite covering `Counter`, `Detector` (synthetic IDs, spatial fallback, source switching), and `StreamResolver`.

---

## Phase 6: Roboflow Supervision & CLAHE Preprocessing Upgrade
* **Roboflow Supervision (`supervision` v0.30.0)**: Standardized object detection parsing around `sv.Detections.from_ultralytics(results)` and `sv.ByteTrack` multi-object tracking.
* **CLAHE Preprocessing**: Added LAB color space Contrast Limited Adaptive Histogram Equalization (`Detector.apply_clahe`) to enhance low-light triage contrast prior to YOLO inference.
* **Configurability**: Added `ENABLE_CLAHE`, `CLAHE_CLIP_LIMIT`, `CLAHE_TILE_GRID_SIZE`, `BYTETRACK_TRACK_THRESH`, `BYTETRACK_MATCH_THRESH`, and `BYTETRACK_FRAME_RATE` to `src/engine/config.py`.

---

## Phase 8: Model Suite & Dynamic Model Switching
* **Fine-Tuned & Base Model Integration**: Organized and integrated models into `models/fine_tuned/` (`pediatric-model.pt`, `pediatric-kids-only.pt`, `pediatric-smaller-dataset-trained.pt`) and `models/base_model/` (`yolo26s.pt`).
* **Dynamic Class Resolution**: Automatically resolves child/adult class indices (`child_class_id` and `adult_class_id`) dynamically from YOLO model class names.
* **Runtime Model Switching**: Added thread-safe `Detector.change_model(model_path)` enabling live runtime model switching without restarting the video stream.
* **Interactive UI Selectors**: Integrated AI model selector dropdown menus in both `/dashboard` and `/video-test`.
* **27/27 Unit Tests Passing**: Expanded test suite to cover dynamic model loading and class resolution.



