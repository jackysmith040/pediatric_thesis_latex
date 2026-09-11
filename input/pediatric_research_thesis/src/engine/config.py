import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Model Configuration

    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/fine_tuned/pediatric-model.pt")
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.45"))

    IOU_THRESHOLD: float = float(os.getenv("IOU_THRESHOLD", "0.4"))
    
    # Tracking Configuration
    TRACKER_CONFIG: str = os.getenv("TRACKER_CONFIG", "bytetrack.yaml")
    ID_EXPIRY_SECONDS: int = int(os.getenv("ID_EXPIRY_SECONDS", "30"))
    UNTRACKED_SPATIAL_MATCH_RADIUS: float = float(os.getenv("UNTRACKED_SPATIAL_MATCH_RADIUS", "40.0"))

    # Video Feed Configuration
    # Defaults to local webcam (0)
    VIDEO_SOURCE: str = os.getenv("VIDEO_SOURCE", "0")

    CAMERA_ID: str = os.getenv("CAMERA_ID", "outpatient_waiting_cctv_01")
    # Overcrowding Logic
    WAITING_ROOM_CAPACITY: int = int(os.getenv("WAITING_ROOM_CAPACITY", "50"))
    PEDIATRIC_ALERT_THRESHOLD_PERCENT: float = float(os.getenv("PEDIATRIC_ALERT_THRESHOLD_PERCENT", "30.0"))

    # Detection & Performance Configuration
    PEDIATRIC_CONF_THRESHOLD: float = float(os.getenv("PEDIATRIC_CONF_THRESHOLD", "0.30"))
    MAX_FRAME_WIDTH: int = int(os.getenv("MAX_FRAME_WIDTH", "1280"))

    # CLAHE Preprocessing Configuration
    ENABLE_CLAHE: bool = os.getenv("ENABLE_CLAHE", "true").lower() in ("true", "1", "yes")
    CLAHE_CLIP_LIMIT: float = float(os.getenv("CLAHE_CLIP_LIMIT", "2.0"))
    CLAHE_TILE_GRID_SIZE: int = int(os.getenv("CLAHE_TILE_GRID_SIZE", "8"))

    # Roboflow Supervision & ByteTrack Configuration
    BYTETRACK_TRACK_THRESH: float = float(os.getenv("BYTETRACK_TRACK_THRESH", "0.30"))
    BYTETRACK_MATCH_THRESH: float = float(os.getenv("BYTETRACK_MATCH_THRESH", "0.8"))
    BYTETRACK_FRAME_RATE: int = int(os.getenv("BYTETRACK_FRAME_RATE", "30"))

    # Perspective Normalization & Temporal Class Voting Configuration
    ENABLE_PERSPECTIVE_CORRECTION: bool = os.getenv("ENABLE_PERSPECTIVE_CORRECTION", "true").lower() in ("true", "1", "yes")
    PERSPECTIVE_HORIZON_Y: float = float(os.getenv("PERSPECTIVE_HORIZON_Y", "0.20"))
    PERSPECTIVE_FAR_HEIGHT_RATIO: float = float(os.getenv("PERSPECTIVE_FAR_HEIGHT_RATIO", "0.22"))
    PERSPECTIVE_NEAR_HEIGHT_RATIO: float = float(os.getenv("PERSPECTIVE_NEAR_HEIGHT_RATIO", "0.55"))
    TEMPORAL_VOTING_WINDOW: int = int(os.getenv("TEMPORAL_VOTING_WINDOW", "15"))

    # Multi-Tracker & Auto-Switching Configuration
    DEFAULT_TRACKER_MODE: str = os.getenv("DEFAULT_TRACKER_MODE", "auto")
    AUTO_SWITCH_STABILIZATION_SECONDS: float = float(os.getenv("AUTO_SWITCH_STABILIZATION_SECONDS", "3.0"))
    CAMERA_MOTION_THRESHOLD: float = float(os.getenv("CAMERA_MOTION_THRESHOLD", "12.0"))
    OCCLUSION_DENSITY_THRESHOLD: float = float(os.getenv("OCCLUSION_DENSITY_THRESHOLD", "0.25"))



    # Class mappings (expbetter.pt trained weights: 0=Adult, 1=Child)
    ADULT_CLASS_ID: int = int(os.getenv("ADULT_CLASS_ID", "0"))
    CHILD_CLASS_ID: int = int(os.getenv("CHILD_CLASS_ID", "1"))

settings = Settings()


PRESET_MODELS = [
    {
        "name": "Fine-Tuned Pediatric Model (ONNX - 2x Fast)",
        "path": "models/onnx_versions_fine_tuned/pediatric-model.onnx",
        "type": "onnx_fine_tuned",
        "description": "High-speed ONNX runtime model for pediatric vs adult classification (2.1x CPU speedup)"
    },
    {
        "name": "Kids-Only Model (ONNX - 2x Fast)",
        "path": "models/onnx_versions_fine_tuned/pediatric-kids-only.onnx",
        "type": "onnx_fine_tuned",
        "description": "High-speed ONNX runtime model focused exclusively on pediatric patient detection"
    },
    {
        "name": "Smaller Dataset Trained Model (ONNX - 2x Fast)",
        "path": "models/onnx_versions_fine_tuned/pediatric-smaller-dataset-trained.onnx",
        "type": "onnx_fine_tuned",
        "description": "High-speed ONNX runtime variant fine-tuned pediatric detection model"
    },
    {
        "name": "Fine-Tuned Pediatric Model (PyTorch)",
        "path": "models/fine_tuned/pediatric-model.pt",
        "type": "fine_tuned",
        "description": "Full fine-tuned PyTorch YOLO model for pediatric vs adult classification"
    },
    {
        "name": "Kids-Only Pediatric Model (PyTorch)",
        "path": "models/fine_tuned/pediatric-kids-only.pt",
        "type": "fine_tuned",
        "description": "Fine-tuned PyTorch model focused exclusively on pediatric patient detection"
    },
    {
        "name": "Smaller Dataset Trained Model (PyTorch)",
        "path": "models/fine_tuned/pediatric-smaller-dataset-trained.pt",
        "type": "fine_tuned",
        "description": "Variant fine-tuned PyTorch pediatric detection model"
    },
    {
        "name": "DINOv3 Distilled Pediatric Model (ONNX - SOTA Edge)",
        "path": "models/onnx_versions_fine_tuned/yolo26s_distilled.onnx",
        "type": "onnx_distilled",
        "description": "DINOv3 ViT representation-distilled YOLO26s model optimized for severe occlusion via ONNX runtime"
    },
    {
        "name": "DINOv3 Distilled Pediatric Model (PyTorch - SOTA)",
        "path": "models/fine_tuned/yolo26s_distilled.pt",
        "type": "distilled",
        "description": "DINOv3 ViT representation-distilled YOLO26s model with dense semantic feature transfer"
    },
    {
        "name": "Base YOLO26 Small Model",
        "path": "models/base_model/yolo26s.pt",
        "type": "base_model",
        "description": "COCO pretrained YOLO26 Small model with height heuristics"
    }
]

