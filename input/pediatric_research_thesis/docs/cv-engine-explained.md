# 👁️ CV Engine Code Explanation & ELI5

The Computer Vision (CV) Engine is responsible for ingesting camera streams, running object detection and tracking, categorizing pediatric patients versus adults, and updating the shared telemetry state.

---

## 1. File Breakdown

### `config.py`
**Purpose:** Configuration parameters and settings management.
- Uses `pydantic_settings` to read environment variables from `.env`.
- Configures model parameters (`MODEL_PATH`, `CONFIDENCE_THRESHOLD`, `IOU_THRESHOLD`), tracking rules, waiting room capacity limits, and class mapping (`ADULT_CLASS_ID = 0`, `CHILD_CLASS_ID = 1`).

### `tracker_manager.py`
**Purpose:** Lifecycle management for object tracking IDs.
- Tracks active object IDs assigned by ByteTrack along with their last seen timestamp.
- Purges expired IDs when objects leave the camera view for longer than `ID_EXPIRY_SECONDS` (default 30 seconds).

### `counter.py`
**Purpose:** Metrics computation and spatial centroid tracking.
- Maintains real-time occupancy counts (`current_adults`, `current_children`) and daily cumulative totals (`total_daily_adults`, `total_daily_children`).
- Performs spatial centroid fallback matching to resolve untracked detections (`track_id == -1`) against nearby active tracks.
- Triggers `overcrowding_alert` when pediatric counts exceed capacity thresholds.

### `detector.py`
**Purpose:** The core vision processing engine.
- Loads YOLO weights (`expbetter.pt`) and initializes Roboflow `supervision` (`sv.Detections`) and `sv.ByteTrack`.
- Applies Contrast Limited Adaptive Histogram Equalization (**CLAHE**) in the LAB color space to enhance low-light contrast without altering color balance.
- Runs a decoupled dual-thread loop:
  - **Capture Thread:** Continuously drains the OpenCV video buffer to prevent lag.
  - **Inference Thread:** Performs CLAHE preprocessing, YOLO detection, ByteTrack tracking, bounding box overlay, tracking trail drawing, and state updates.
- Provides `change_source(new_source)` to dynamically switch video inputs (Webcam, local MP4, YouTube, RTSP streams).
- Generates JPEG bytes for the native browser `/camera/stream` MJPEG endpoint.

### `stream_resolver.py`
**Purpose:** Input stream resolution and validation.
- Parses video source input strings (webcam index `0`, YouTube URLs, RTSP/HTTP endpoints, or local MP4 files).
- Uses `cap_from_youtube` for YouTube resolution.
- Maintains preset clinical test video feeds for evaluation.

### `reporter.py`
**Purpose:** Report generation.
- Generates downloadable CSV logs of system telemetry metrics.
- Uses `FPDF` to construct formatted clinical PDF capacity reports.

---

## 2. ELI5 (Explain Like I'm 5)

Imagine a specialized medical security team operating a triage desk:

1. **The Rulebook (`config.py`)**  
   The official guide telling the team where the camera is, how to identify adults vs. children, and when to trigger an overcrowding alarm.

2. **The Bouncer (`tracker_manager.py`)**  
   Stands by the door and gives every person a sticky name badge. If someone leaves the room for more than 30 seconds, the bouncer removes their badge.

3. **The Accountant (`counter.py`)**  
   Constantly tallies the active badges in the room, updates the whiteboard with total adults and children, and sounds an alarm if the waiting room gets too crowded.

4. **The Detective (`detector.py`)**  
   Watches the camera screen with two eyes: one eye constantly watches the live feed so it never stutters, while the other eye draws colored boxes around people.

5. **The Stream Finder (`stream_resolver.py`)**  
   Connects the Detective to any camera feed: a local webcam, a recorded hospital video, or a live YouTube feed.

6. **The Report Clerk (`reporter.py`)**  
   Takes the numbers from the Accountant's whiteboard and creates formatted CSV spreadsheet files or PDF reports for hospital administrators.
