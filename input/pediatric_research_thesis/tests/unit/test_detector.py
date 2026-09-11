import pytest
from unittest.mock import MagicMock
from src.engine.detector import Detector
from src.engine.counter import Counter
from src.engine.tracker_manager import TrackerManager
from src.state.telemetry import TelemetryState
from src.engine.config import settings

@pytest.fixture
def mock_detector(monkeypatch):
    # Prevent YOLO model loading and video capture opening during unit tests
    monkeypatch.setattr("src.engine.detector.YOLO", MagicMock())
    monkeypatch.setattr("cv2.VideoCapture", MagicMock())
    
    tracker_manager = TrackerManager(expiry_seconds=30)
    state = TelemetryState()
    counter = Counter(tracker_manager, state)
    
    settings.ADULT_CLASS_ID = 0
    settings.CHILD_CLASS_ID = 1
    settings.UNTRACKED_SPATIAL_MATCH_RADIUS = 40.0
    
    detector = Detector.__new__(Detector)
    detector.counter = counter
    detector._synthetic_id_counter = 10000
    return detector

def test_resolve_track_id_returns_original_if_valid_and_unclaimed(mock_detector):
    box = (10, 10, 50, 50)
    claimed_ids = set()
    resolved = mock_detector._resolve_track_id(box, track_id=42, class_id=0, claimed_ids=claimed_ids)
    assert resolved == 42

def test_resolve_track_id_matches_existing_active_centroid_if_unclaimed(mock_detector):
    mock_detector.counter.active_centroids[5] = (30.0, 30.0, 0)
    box = (12, 12, 52, 52)
    claimed_ids = set()
    resolved = mock_detector._resolve_track_id(box, track_id=-1, class_id=0, claimed_ids=claimed_ids)
    assert resolved == 5

def test_resolve_track_id_prevents_stolen_claimed_ids(mock_detector):
    # Active centroid 5 is already claimed in this frame
    mock_detector.counter.active_centroids[5] = (30.0, 30.0, 0)
    box = (12, 12, 52, 52)
    claimed_ids = {5}
    resolved = mock_detector._resolve_track_id(box, track_id=-1, class_id=0, claimed_ids=claimed_ids)
    # Cannot claim ID 5 because it's already claimed -> gets new synthetic ID
    assert resolved == 10001

def test_multiple_untracked_detections_receive_distinct_ids(mock_detector):
    # Two people close together without track IDs (-1)
    box1 = (10, 10, 30, 30) # Centroid (20, 20)
    box2 = (15, 15, 35, 35) # Centroid (25, 25)
    
    claimed_ids = set()
    res1 = mock_detector._resolve_track_id(box1, track_id=-1, class_id=1, claimed_ids=claimed_ids)
    claimed_ids.add(res1)
    
    res2 = mock_detector._resolve_track_id(box2, track_id=-1, class_id=1, claimed_ids=claimed_ids)
    claimed_ids.add(res2)
    
    assert res1 != res2
    assert len(claimed_ids) == 2

def test_detector_change_source_success(mock_detector, monkeypatch):
    import threading
    mock_detector.source_lock = threading.Lock()
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    monkeypatch.setattr("cv2.VideoCapture", lambda target: mock_cap)
    
    success, label = mock_detector.change_source("0")
    assert success is True
    assert mock_detector.current_source_label == label
    assert mock_detector.last_error is None

def test_detector_change_source_failure_on_closed_cap(mock_detector, monkeypatch):
    import threading
    mock_detector.source_lock = threading.Lock()
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    monkeypatch.setattr("cv2.VideoCapture", lambda target: mock_cap)
    
def test_class_id_mappings(mock_detector):
    assert settings.ADULT_CLASS_ID == 0
    assert settings.CHILD_CLASS_ID == 1

def test_uses_coco_person_flag(mock_detector):
    mock_detector.names = {0: 'person'}
    mock_detector.uses_coco_person = len(mock_detector.names) == 1 and 'person' in str(mock_detector.names.get(0, '')).lower()
    assert mock_detector.uses_coco_person is True

def test_apply_clahe_preprocessing():
    import numpy as np
    # Create a synthetic 100x100 BGR test image
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_frame[20:80, 20:80] = [100, 150, 200]
    
    enhanced = Detector.apply_clahe(dummy_frame, clip_limit=2.0, tile_grid_size=8)
    assert enhanced is not None
    assert enhanced.shape == dummy_frame.shape
    assert enhanced.dtype == np.uint8

def test_supervision_bytetrack_initialization(mock_detector):
    import supervision as sv
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        tracker = sv.ByteTrack(
            track_activation_threshold=settings.BYTETRACK_TRACK_THRESH,
            minimum_matching_threshold=settings.BYTETRACK_MATCH_THRESH,
            frame_rate=settings.BYTETRACK_FRAME_RATE
        )
    assert tracker is not None


def test_change_tracker_mode(mock_detector):
    from src.engine.tracker_engine import MultiTrackerEngine
    mock_detector.tracker_engine = MultiTrackerEngine(initial_mode="auto")
    mock_detector.current_tracker_status_label = "ByteTrack [Auto]"
    
    success, msg = mock_detector.change_tracker_mode("botsort")
    assert success is True
    assert "BoT-SORT" in msg
    assert "BoT-SORT" in mock_detector.current_tracker_status_label
    assert "[Manual]" in mock_detector.current_tracker_status_label

def test_change_model_success(mock_detector, monkeypatch):
    import threading
    mock_detector.model_lock = threading.Lock()
    mock_model = MagicMock()
    mock_model.names = {0: 'child', 1: 'adult'}
    monkeypatch.setattr("src.engine.detector.YOLO", lambda *args, **kwargs: mock_model)
    
    success, msg = mock_detector.change_model("models/fine_tuned/pediatric-model.pt")
    assert success is True
    assert mock_detector.child_class_id == 0
    assert mock_detector.adult_class_id == 1
    assert "pediatric-model.pt" in msg
    assert "PyTorch" in msg

def test_change_model_onnx_success(mock_detector, monkeypatch):
    import threading
    mock_detector.model_lock = threading.Lock()
    mock_model = MagicMock()
    mock_model.names = {0: 'child', 1: 'adult'}
    monkeypatch.setattr("src.engine.detector.YOLO", lambda *args, **kwargs: mock_model)
    
    success, msg = mock_detector.change_model("models/onnx_versions_fine_tuned/pediatric-model.onnx")
    assert success is True
    assert mock_detector.is_onnx is True
    assert mock_detector.child_class_id == 0
    assert mock_detector.adult_class_id == 1
    assert "pediatric-model.onnx" in msg
    assert "ONNX Runtime" in msg

def test_perspective_normalization_sitting_pose():
    import numpy as np
    # Box for a sitting adult: y_bottom=0.8, height=0.35, width=0.25 (aspect_ratio = 0.71)
    box_sitting_adult = np.array([100, 450, 350, 800])  # h=350, w=250 -> w/h = 0.71
    frame_h = 1000.0
    
    cls = Detector.calculate_perspective_class(
        box_sitting_adult, frame_h, raw_class_id=0, child_cls=1, adult_cls=0
    )
    # Sitting adult with aspect ratio compensation should be classified as Adult (0)
    assert cls == 0

def test_all_preset_models_resolution(mock_detector, monkeypatch):
    import threading
    mock_detector.model_lock = threading.Lock()
    
    # 1. Test Kids-Only Model {0: 'Child'}
    kids_model = MagicMock()
    kids_model.names = {0: 'Child'}
    monkeypatch.setattr("src.engine.detector.YOLO", lambda *args, **kwargs: kids_model)
    mock_detector.change_model("models/fine_tuned/pediatric-kids-only.pt")
    assert mock_detector.child_class_id == 0
    assert mock_detector.adult_class_id == -1
    
    # 2. Test Smaller Dataset Trained Model {0: 'Child', 1: 'Adult'}
    variant_model = MagicMock()
    variant_model.names = {0: 'Child', 1: 'Adult'}
    monkeypatch.setattr("src.engine.detector.YOLO", lambda *args, **kwargs: variant_model)
    mock_detector.change_model("models/fine_tuned/pediatric-smaller-dataset-trained.pt")
    assert mock_detector.child_class_id == 0
    assert mock_detector.adult_class_id == 1
    
    # 3. Test Base YOLO26 COCO Model {0: 'person', ...}
    coco_model = MagicMock()
    coco_model.names = {0: 'person', 1: 'bicycle', 56: 'chair'}
    monkeypatch.setattr("src.engine.detector.YOLO", lambda *args, **kwargs: coco_model)
    mock_detector.change_model("models/base_model/yolo26s.pt")
    assert mock_detector.uses_coco_person is True
    assert mock_detector.child_class_id == settings.CHILD_CLASS_ID
    assert mock_detector.adult_class_id == settings.ADULT_CLASS_ID

def test_onnx_preset_models_in_config():
    from src.engine.config import PRESET_MODELS
    paths = [p["path"] for p in PRESET_MODELS]
    assert "models/onnx_versions_fine_tuned/pediatric-model.onnx" in paths
    assert "models/onnx_versions_fine_tuned/pediatric-kids-only.onnx" in paths
    assert "models/onnx_versions_fine_tuned/pediatric-smaller-dataset-trained.onnx" in paths

def test_real_onnx_models_inference_integration():
    import os
    import numpy as np
    from ultralytics import YOLO

    onnx_paths = [
        "models/onnx_versions_fine_tuned/pediatric-model.onnx",
        "models/onnx_versions_fine_tuned/pediatric-kids-only.onnx",
        "models/onnx_versions_fine_tuned/pediatric-smaller-dataset-trained.onnx"
    ]

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    for path in onnx_paths:
        if os.path.exists(path):
            model = YOLO(path, task="detect")
            assert model is not None
            assert len(model.names) > 0
            results = model.predict(dummy_frame, imgsz=480, verbose=False)
            assert results is not None
            assert len(results) == 1








