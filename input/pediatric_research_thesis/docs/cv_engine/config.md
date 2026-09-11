# `src/engine/config.py`

## Purpose
`config.py` manages application configuration, model paths, detection thresholds, and clinical thresholds using `pydantic_settings`.

---

## Configuration Variables

| Property | Default Value | Description |
|---|---|---|
| `MODEL_PATH` | `"models/fine_tuned/pediatric-model.pt"` | Default path to trained YOLO model weights (`.onnx` or `.pt`) |

| `CONFIDENCE_THRESHOLD` | `0.45` | Minimum confidence score to accept object detections |
| `IOU_THRESHOLD` | `0.40` | Intersection-over-Union threshold for Non-Maximum Suppression (NMS) |
| `TRACKER_CONFIG` | `"bytetrack.yaml"` | ByteTrack tracker configuration file |
| `ID_EXPIRY_SECONDS` | `30` | Time in seconds before unseen tracking IDs are purged from memory |
| `UNTRACKED_SPATIAL_MATCH_RADIUS` | `40.0` | Radius (pixels) for matching untracked bounding boxes (`track_id == -1`) to existing centroids |
| `VIDEO_SOURCE` | `"0"` | Video source input (Webcam index `"0"`, local file path, YouTube URL, or RTSP stream) |
| `WAITING_ROOM_CAPACITY` | `50` | Total patient capacity of monitored waiting area |
| `PEDIATRIC_ALERT_THRESHOLD_PERCENT` | `30.0` | Pediatric percentage threshold that triggers an overcrowding alert |
| `ADULT_CLASS_ID` | `0` | Model class index mapped to Adult |
| `CHILD_CLASS_ID` | `1` | Model class index mapped to Child |

---

## Supported Preset Models

| Model Name | Path | Runtime | Speed (CPU) | Description |
|---|---|---|---|---|
| **Fine-Tuned Pediatric Model (ONNX)** | `models/onnx_versions_fine_tuned/pediatric-model.onnx` | ONNX Runtime | ~150 ms (~6.7 FPS) | High-speed 2-class pediatric vs adult detection (2.1x speedup) |
| **Kids-Only Model (ONNX)** | `models/onnx_versions_fine_tuned/pediatric-kids-only.onnx` | ONNX Runtime | ~150 ms (~6.7 FPS) | High-speed dedicated pediatric patient detector |
| **Smaller Dataset Model (ONNX)** | `models/onnx_versions_fine_tuned/pediatric-smaller-dataset-trained.onnx` | ONNX Runtime | ~150 ms (~6.7 FPS) | High-speed variant fine-tuned pediatric detector |
| **Fine-Tuned Pediatric Model (PyTorch)** | `models/fine_tuned/pediatric-model.pt` | PyTorch | ~318 ms (~3.1 FPS) | Full fine-tuned PyTorch model |
| **Kids-Only Model (PyTorch)** | `models/fine_tuned/pediatric-kids-only.pt` | PyTorch | ~318 ms (~3.1 FPS) | Dedicated PyTorch pediatric detector |
| **Smaller Dataset Model (PyTorch)** | `models/fine_tuned/pediatric-smaller-dataset-trained.pt` | PyTorch | ~318 ms (~3.1 FPS) | Variant PyTorch fine-tuned detector |
| **Base YOLO26 Small Model** | `models/base_model/yolo26s.pt` | PyTorch | ~240 ms (~4.2 FPS) | COCO pretrained model with perspective normalization |

---

## Environment Override (.env)

Configuration values can be overridden via `.env` file or environment variables:

```ini
MODEL_PATH=models/onnx_versions_fine_tuned/pediatric-model.onnx
CONFIDENCE_THRESHOLD=0.45
WAITING_ROOM_CAPACITY=50
PEDIATRIC_ALERT_THRESHOLD_PERCENT=30.0
VIDEO_SOURCE=0
```
