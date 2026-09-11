# `src/engine/tracker_engine.py`

## Purpose
`tracker_engine.py` provides the **MultiTrackerEngine** and **SceneAnalyzer** classes for situation-aware multi-tracker selection and automatic tracker switching.

---

## Supported Tracking Algorithms

| Key | Title | Ideal Use Case |
|---|---|---|
| `bytetrack` | ByteTrack (Fastest Baseline) | Static triage camera feeds, fast zero-ReID baseline. |
| `botsort` | BoT-SORT (Camera Motion Comp.) | Handheld, moving camera, or PTZ camera feeds with camera motion compensation. |
| `ocsort` | OC-SORT (Non-Linear Motion) | Non-linear patient movement and abrupt direction changes without appearance overhead. |
| `fasttracker` | FastTracker (Occlusion Aware) | Crowded waiting room triage with high bounding box overlap and partial occlusions. |

---

## Core Classes

### `SceneAnalyzer`
* **`estimate_camera_motion(prev_gray, current_gray) -> float`**: Downsamples grayscale frames and calculates average pixel displacement.
* **`estimate_occlusion_density(boxes: np.ndarray) -> float`**: Computes mean pairwise Intersection-over-Union (IoU) overlap among detection bounding boxes.
* **`recommend_tracker(motion_score, occlusion_score) -> str`**: Returns optimal tracker key based on environmental conditions.

### `MultiTrackerEngine`
* **`set_mode(new_mode: str) -> Tuple[bool, str]`**: Switches operating mode (`"auto"` or explicit tracker key `"bytetrack"`, `"botsort"`, `"ocsort"`, `"fasttracker"`).
* **`update_with_detections(detections, frame) -> Tuple[sv.Detections, str]`**: Evaluates scene conditions, applies a 3.0-second hysteresis stabilization buffer to prevent rapid switching flickers, and tracks detections.
* **`get_info() -> dict`**: Returns structured telemetry for UI badge display.
