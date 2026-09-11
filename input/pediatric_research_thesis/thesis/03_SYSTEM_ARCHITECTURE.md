# Chapter 3: System Architecture & Engineering Methodology

---

## 3.1 Monolithic Architecture vs. Distributed Microservices

### 3.1.1 The Dual-Stack Architectural Bottleneck
In early architectural iterations of this research (Phase 1), the system was developed as a decoupled dual-stack microservice infrastructure comprising:
1. A **Python FastAPI backend** responsible for OpenCV video capture, YOLO inference, and MJPEG video streaming.
2. A **Laravel 13 PHP backend** serving the administrative web interface and storing telemetry logs in SQLite.
3. A **Laravel Reverb WebSocket server** broadcasting telemetry events (`TelemetryReceived`) to a Livewire 4 / Alpine.js frontend.

```
       EARLY DUAL-STACK ARCHITECTURE (PHASE 1 - DEPRECATED)
       ┌────────────────────────┐         REST POST (JSON)        ┌────────────────────────┐
       │ Python CV Engine       │ ──────────────────────────────► │ Laravel 13 Backend     │
       │ (FastAPI + YOLO)       │                                 │ (PHP 8.4 + SQLite)     │
       │ Port 5001              │ ◄────────────────────────────── │ Port 8000              │
       └───────────┬────────────┘      Direct <img> MJPEG Feed    └───────────┬────────────┘
                   │                                                          │
                   │                                                          ▼
                   │                                              ┌────────────────────────┐
                   │                                              │ Laravel Reverb Server  │
                   │                                              │ (WebSockets Port 8080) │
                   │                                              └───────────┬────────────┘
                   │                                                          │
                   ▼                                                          ▼
       ┌────────────────────────────────────────────────────────────────────────┐
       │                   Browser UI (Livewire 4 + Alpine.js)                  │
       └────────────────────────────────────────────────────────────────────────┘
```

While functional, rigorous performance benchmarking revealed severe latency and operational degradation inherent to this distributed approach:
- **Inter-Process Network Overhead:** Dispatched HTTP POST payloads every 3 seconds incurred JSON serialization, HTTP handshake, and network I/O overhead.
- **WebSocket Desynchronization:** Telemetry broadcast over WebSockets experienced intermittent packet queueing on resource-constrained local networks, causing the displayed numerical counts to lag behind the live visual video feed by $1.5\text{ to }4.0\text{ seconds}$.
- **Operational Brittleness:** Maintaining three concurrent background services (Uvicorn, PHP-FPM/Herd, and Reverb) introduced multiple points of failure. If the Reverb daemon crashed, the telemetry interface silently stalled while video continued to play.

### 3.1.2 The NiceGUI Monolithic Paradigm
To eliminate all network serialization overhead and simplify clinical deployment, the system pivoted to a unified **Single Python Monolith** powered by [NiceGUI](https://nicegui.io/) (built on top of FastAPI, Starlette, and Uvicorn).

```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                 THE UNIFIED PEDIATRIC MONITOR MONOLITH                  │
  │                                                                         │
  │  ┌─────────────────────────┐             ┌───────────────────────────┐  │
  │  │ NiceGUI Web UI Layer    │             │ Computer Vision Engine    │  │
  │  │ - Landing Page (/)      │             │ - LAB CLAHE Preprocessor  │  │
  │  │ - Dashboard (/dashboard)│             │ - Distilled YOLO26s (ONNX)│  │
  │  │ - Eval Lab (/video-test)│ ◄─────────► │ - SAHI Multi-Patch Slicer │  │
  │  │ - Stream Endpoint       │  In-Memory  │ - MultiTracker Suite      │  │
  │  │   (/camera/stream)      │  Direct     │ - Scene Motion Analyzer   │  │
  │  │ - PDF / CSV Exporter    │  Binding    │ - Spatial Centroid Memory │  │
  │  │ - LaTeX Table Generator │             │ - Debouncing Lost Queue   │  │
  │  └─────────────────────────┘             └───────────────────────────┘  │
  │                 │                                      │                │
  └─────────────────┼──────────────────────────────────────┼────────────────┘
                    ▼                                      ▼
        Browser / Native PyWebView             Hospital Video Ingestion Feed
        (Single Command Execution)             (UVC Webcam / RTSP / Network)
```

In this architecture:
- The Computer Vision Engine, Multi-Tracker Suite, Telemetry Model, and User Interface execute within a **single unified Python runtime process**.
- Communication between computer vision outputs and UI elements occurs via direct **in-memory object reference binding**, reducing telemetry propagation latency to sub-millisecond speeds ($<0.1\text{ ms}$).
- The entire system launches via a single standard execution command (`python -m src.main`), radically simplifying deployment on clinical workstations.

---

## 3.2 Two-Stage Machine Learning Pipeline (Offline Distillation $\to$ Online Edge Inference)

The system establishes a clean architectural separation between heavy offline knowledge distillation pre-training and lightweight online edge execution:

```
┌──────────────────────────────────────────────────────────────────────────┐
│             STAGE 1: OFFLINE KNOWLEDGE DISTILLATION & OPTIMIZATION       │
│                                                                          │
│  Unlabeled Clinical CCTV Feed + Labeled Pediatric Dataset (2,414 NDJSON) │
│                                │                                         │
│                                ▼                                         │
│  Dense Feature Distillation: DINOv3 ViT Teacher ──► YOLO26s Student Neck │
│  (Self-Supervised Cosine Similarity + Normalized MSE Loss)               │
│                                │                                         │
│                                ▼                                         │
│  Supervised Fine-Tuning on Pediatric Dataset (CIoU + DFL + BCE Loss)     │
│                                │                                         │
│                                ▼                                         │
│  Static ONNX Graph Export & CPU Quantization (yolo26s_distilled.onnx)    │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │ Optimized Model Binary (11.2M Params)
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│             STAGE 2: ONLINE REAL-TIME EDGE INFERENCE & TRACKING          │
│                                                                          │
│  Hospital CCTV Video Feed (UVC / RTSP / MP4)                             │
│                                │                                         │
│                                ▼                                         │
│  Decoupled Capture Worker Thread (CAP_PROP_BUFFERSIZE = 1)               │
│                                │                                         │
│                                ▼                                         │
│  LAB Color Space CLAHE Preprocessing (L* Luminance Contrast Boost)       │
│                                │                                         │
│                                ▼                                         │
│  SAHI Multi-Scale Patch Slicing (640x640 Tiles, 20% Overlap)             │
│                                │                                         │
│                                ▼                                         │
│  High-Throughput ONNX Runtime CPU Inference (28.4 ms Execution)          │
│                                │                                         │
│                                ▼                                         │
│  Adaptive Multi-Tracker Suite (ByteTrack / BoT-SORT / FastTracker)       │
│  + Spatial Centroid Fallback (r <= 40 px) & Temporal Debouncing (5.0 s)  │
│                                │                                         │
│                                ▼                                         │
│  In-Memory Telemetry Binding ──► NiceGUI Clinical Dashboard & Alerts     │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3.3 Decoupled Dual-Thread Capture-Inference Pattern

A notorious vulnerability of OpenCV's `cv2.VideoCapture` interface is internal frame buffer latency. By default, OpenCV maintains a multi-frame circular hardware buffer. If frame capture and neural inference execute synchronously within a single loop:

$$\text{Loop Time} = t_{\text{capture}} + t_{\text{CLAHE}} + t_{\text{YOLO}} + t_{\text{track}} + t_{\text{render}} \approx 35\text{ to }65\text{ ms}$$

Because deep learning inference takes longer than camera frame acquisition ($33.3\text{ ms}$ at 30 FPS), the OpenCV hardware buffer quickly fills with stale frames. Consequently, the displayed video feed accumulates a severe **3 to 5 second visual lag**, completely destroying real-time clinical responsiveness.

### 3.3.1 Asynchronous Thread Architecture
To solve this, the CV Engine implements a decoupled **asynchronous dual-thread execution model**:

```
                       ┌────────────────────────┐
                       │    Video Source Feed   │
                       └───────────┬────────────┘
                                   │
                                   ▼
                      ┌──────────────────────────┐
                      │  Thread 1: Capture Loop  │
                      │  - Drains OpenCV Buffer  │
                      │  - CAP_PROP_BUFFERSIZE=1 │
                      │  - Zero Frame Lag        │
                      └────────────┬─────────────┘
                                   │ Latest Frame (latest_frame)
                                   ▼
                      ┌──────────────────────────┐
                      │  Thread 2: YOLO Loop     │
                      │  - LAB CLAHE Preproc     │
                      │  - SAHI Multi-Patch / ONNX│
                      │  - MultiTracker Update   │
                      │  - Spatial Fallback Res  │
                      └────────────┬─────────────┘
                                   │ Latest Boxes & State Update
                      ┌────────────┴─────────────┐
                      ▼                          ▼
            ┌──────────────────┐       ┌──────────────────┐
            │ TelemetryState   │       │ /camera/stream   │
            │ In-Memory Model  │       │ MJPEG Stream     │
            │ (Direct UI Bind) │       │ (30 FPS Smooth)  │
            └──────────────────┘       └──────────────────┘
```

#### Thread 1: Camera Capture Worker (`Detector._capture_loop`)
- Runs in an unblocked, high-priority loop continuously calling `cap.read()`.
- Explicitly enforces `cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)` on the hardware capture handle.
- Performs instant mirror flipping (`cv2.flip(frame, 1)`) for live webcams or preserves raw spatial orientation for RTSP/MP4 streams.
- Dynamically downsamples high-resolution frames ($>1280\text{ px}$) to `MAX_FRAME_WIDTH = 1280` via bilinear area interpolation (`cv2.INTER_AREA`), maintaining high inference frame rates without sacrificing aspect ratio.
- Renders the latest bounding boxes onto the frame and encodes it to JPEG bytes (`cv2.imencode('.jpg', annotated, quality=75)`), storing the buffer in `self.latest_jpeg_bytes`.
- Automatically loops video files upon reaching End-of-File (EOF) by seeking to frame index 0 (`cap.set(cv2.CAP_PROP_POS_FRAMES, 0)`), ensuring uninterrupted continuous stress testing.

#### Thread 2: Neural Inference Worker (`Detector._yolo_loop`)
- Fetches the freshest available frame snapshot from `self.latest_frame`.
- Executes CLAHE contrast enhancement in the LAB color space.
- Invokes deep neural inference via ONNX Runtime with SAHI patch slicing.
- Updates the active multi-tracker instance and executes scene analysis.
- Resolves untracked detections via spatial centroid fallback matching.
- Updates the centralized `TelemetryState` model and passes new bounding box overlays to `self.latest_boxes` behind a lightweight mutex lock (`self.box_lock`).

By fully decoupling frame acquisition from neural processing, the video feed never stutters or accumulates buffer lag, delivering smooth 30 FPS video even when deep inference runs at 25–35 FPS.

---

## 3.4 Zero-Network-Serialization In-Memory State Model

System state is managed by a centralized, thread-safe Pydantic data model defined in `src/state/telemetry.py`:

```python
class TelemetryState(BaseModel):
    current_adults: int = 0
    current_children: int = 0
    total_daily_adults: int = 0
    total_daily_children: int = 0
    overcrowding_alert: bool = False
    camera_id: str = "outpatient_waiting_cctv_01"
```

### Direct UI Property Binding
NiceGUI utilizes fine-grained reactive bindings. Rather than polling REST endpoints or parsing JSON strings in JavaScript, NiceGUI elements bind directly to the in-memory `TelemetryState` instance:

```python
# Direct reactive binding in NiceGUI UI components
ui.label().bind_text_from(state, 'current_children', backward=lambda val: f"{val:02d}")
ui.label().bind_text_from(state, 'current_adults', backward=lambda val: f"{val:02d}")
ui.label().bind_visibility_from(state, 'overcrowding_alert')
```

Whenever the `Counter` updates `state.current_children`, NiceGUI automatically updates the corresponding DOM elements over an internal WebSocket pipe with zero JSON serialization or application-level parsing.

---

## 3.5 Hardware Abstraction & Dynamic Stream Resolution

Clinical surveillance infrastructure is inherently heterogeneous: facilities may deploy USB webcams (UVC), high-resolution RTSP IP security cameras, standard HTTP video streams, or local recorded validation footage.

The system incorporates a dynamic stream resolution layer (`StreamResolver`, defined in `src/engine/stream_resolver.py`):

```
                       User / UI Stream Input
                                  │
            ┌─────────────────────┼─────────────────────┐
            ▼                     ▼                     ▼
      Numeric Digit         YouTube URL           RTSP / HTTP / File
      (e.g., "0", "1")  (e.g., "youtu.be/...")    (e.g., "rtsp://...", ".mp4")
            │                     │                     │
            ▼                     ▼                     ▼
      Direct OpenCV         cap_from_youtube      Standard OpenCV
      Device Index          Resolution            Video Pipeline
            │                     │                     │
            └─────────────────────┼─────────────────────┘
                                  ▼
                      Opened cv2.VideoCapture Handle
                      + Thread-Safe Lock Acquisition
```

### Supported Stream Types:
1. **Direct UVC Webcams (`source = "0", "1"`):** Resolves directly to physical USB video capture devices with automatic 720p/1080p resolution negotiation.
2. **YouTube Clinical Streams (`source = "https://youtube.com/..."`):** Utilizes `cap_from_youtube` to parse live video manifests and stream real-world clinical and pedestrian datasets without requiring manual file downloads.
3. **RTSP / Network IP Cameras (`source = "rtsp://admin:pass@192.168.1.50:554/stream1"`):** Connects to on-premise hospital CCTV networks with low-latency TCP transport flags.
4. **Local Video Files (`source = "path/to/test.mp4"`):** Enables deterministic playback of ground-truth test sequences with automated looping.

### Thread-Safe Dynamic Source Switching (`Detector.change_source`)
The application supports seamless runtime switching between video feeds without restarting the server:
- Acquires `self.source_lock`.
- Safely releases the existing `cv2.VideoCapture` instance.
- Initializes the new stream handle and resets capture thread timing.
- Clears active tracking histories to prevent cross-stream ID contamination.
- Releases `self.source_lock` and updates UI status badges.

---

## 3.6 Operational Modes: Web Browser vs. Native Desktop App

The application provides dual runtime operational modes:

1. **Standard Browser Mode (Port 8080):**  
   The application is hosted via Uvicorn on `http://127.0.0.1:8080` (or local area network IP). It can be accessed simultaneously by multiple clinical workstations, wall-mounted nursing monitors, and mobile tablets. Telemetry and video streaming endpoints (`/camera/stream`) utilize standard HTTP multipart MJPEG protocols natively supported by all modern web browsers.

2. **Native PyWebView Mode:**  
   For standalone deployment on clinical desktops without internet access, NiceGUI can be initialized with native window wrapping (`native=True`). In this mode, downloadable CSV telemetry logs and formatted PDF capacity reports automatically detect the native desktop environment and save directly to the operating system's local `Downloads/` folder, providing a seamless desktop software experience.
