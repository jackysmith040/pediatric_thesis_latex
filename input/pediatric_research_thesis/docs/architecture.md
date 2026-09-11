# 🏗️ Architecture Decisions & System Design

This document details the software architecture, technical decisions, and operational design of **The Invisible Child - Pediatric Monitor**.

---

## 1. Executive Summary & Monolith Architecture

The Pediatric Monitor is an edge-deployed computer vision and telemetry system designed to detect, track, and differentiate pediatric patients (children) from adults in hospital triage and outpatient waiting areas.

### The NiceGUI Monolith Architecture

The application operates as a **Single Python Monolith** powered by [NiceGUI](https://nicegui.io/) (built on FastAPI and Uvicorn).

```
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                       Pediatric Monitor Monolith                        │
 │                                                                         │
 │  ┌───────────────────────┐             ┌─────────────────────────────┐  │
 │  │ NiceGUI Web UI        │             │ Computer Vision Engine      │  │
 │  │ - / (Landing)         │             │ - YOLOv26 Object Detector   │  │
 │  │ - /dashboard          │             │ - Roboflow Supervision      │  │
 │  │ - /video-test         │ ◄─────────► │ - ByteTrack Multi-Tracker   │  │
 │  │ - /camera/stream      │   In-Memory │ - LAB Space CLAHE Preproc   │  │
 │  │                       │   State     │ - Centroid Spatial Fallback │  │
 │  └───────────────────────┘   Binding   └─────────────────────────────┘  │
 │              │               Binding                  │                 │
 └──────────────┼────────────────────────────────────────┼─────────────────┘
                ▼                                        ▼
   Browser / Native PyWebView                  Camera Feed / Stream Source
```

---

## 2. Core Subsystems

### A. Computer Vision & Inference Engine (`src/engine/`)
* **Detector (`detector.py`)**: Manages the dual-thread execution model, YOLO inference, CLAHE preprocessing, multi-tracker management, and dynamic camera source switching.
* **Multi-Tracker Suite (`tracker_engine.py`)**: Supports `ByteTrack` (default baseline), `BoT-SORT` (motion compensation), `OC-SORT` (non-linear motion), and `FastTracker` (occlusion aware).
* **Scene Analyzer (`tracker_engine.py`)**: Analyzes live camera motion (frame difference optical flow) and crowd occlusion density (pairwise IoU overlap) to auto-switch trackers with a 3.0-second hysteresis stabilization buffer.
* **Dual-Thread Capture Pattern**: 
  * **Capture Thread**: Continuously drains the OpenCV video buffer (`CAP_PROP_BUFFERSIZE = 1`) to guarantee zero frame delay on live feeds.
  * **Inference Thread**: Executes CLAHE preprocessing, object detection, multi-tracker update, tracking ID resolution, and centroid spatial fallback without blocking frame capture.

* **Tracker Manager (`tracker_manager.py`)**: Tracks active object IDs and purges expired IDs after a configurable timeout (`ID_EXPIRY_SECONDS = 30`).
* **Counter (`counter.py`)**: Computes real-time pediatric vs. adult counts, maintains daily processed totals, updates active centroids, and checks overcrowding alert logic.
* **Stream Resolver (`stream_resolver.py`)**: Dynamically resolves video inputs into OpenCV streams (Webcams, RTSP streams, YouTube videos via `cap_from_youtube`, local MP4 files).
* **Reporter (`reporter.py`)**: Generates downloadable clinical CSV telemetry logs and formatted PDF capacity reports.

### B. Shared In-Memory State (`src/state/telemetry.py`)
* **TelemetryState**: A Pydantic data model holding live occupancy statistics (`current_children`, `current_adults`, `total_daily_children`, `total_daily_adults`, `overcrowding_alert`).
* **Direct UI Binding**: NiceGUI elements natively bind to `TelemetryState` attributes via `bind_text_from`, eliminating REST/WebSocket network serialization overhead.

### C. Clinical UI & Design System (`src/ui/`)
* **Landing Page (`landing.py`)**: Operational front page presenting clinical metrics and action triggers.
* **Command Center Dashboard (`dashboard.py`)**: Live video stream monitoring, real-time telemetry sidebar, webcam toggle controls, and report downloads.
* **Evaluation Lab (`evaluation.py`)**: Stream testing suite supporting preset YouTube/local video feeds and custom URL inputs.
* **Design Tokens (`components.py`)**: Dark mode aesthetic inspired by clinical interfaces (`slate-950` background, `slate-900` cards, `indigo-600` accents).

---

## 3. High-Performance Multithreading Architecture

```
                      ┌────────────────────────┐
                      │    Video Source Feed   │
                      └───────────┬────────────┘
                                  │
                                  ▼
                     ┌──────────────────────────┐
                     │  Capture Thread (Frame)  │
                     │  CAP_PROP_BUFFERSIZE = 1 │
                     └────────────┬─────────────┘
                                  │ Latest Frame
                                  ▼
                     ┌──────────────────────────┐
                     │ Inference Thread (Async) │
                     │ - YOLOv26 Inference      │
                     │ - ByteTrack ID Assign    │
                     │ - Centroid Fallback      │
                     └────────────┬─────────────┘
                                  │ Annotate Frame & Update State
                     ┌────────────┴─────────────┐
                     ▼                          ▼
           ┌──────────────────┐       ┌──────────────────┐
           │ TelemetryState   │       │ /camera/stream   │
           │ In-Memory Model  │       │ MJPEG Response   │
           └──────────────────┘       └──────────────────┘
```

---

## 4. Operational Modes

1. **Browser Mode**: Served via standard HTTP on port 8080 (supported on desktop, mobile, and wall-mounted clinical monitors).
2. **Native PyWebView Mode**: Desktop app mode where downloadable CSV/PDF reports are written directly to the user's local `Downloads/` directory.
