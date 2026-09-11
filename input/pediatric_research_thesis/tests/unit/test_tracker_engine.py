import pytest
import numpy as np
import supervision as sv
from src.engine.tracker_engine import SceneAnalyzer, MultiTrackerEngine, SUPPORTED_TRACKERS
from src.engine.config import settings

def test_scene_analyzer_camera_motion_estimation():
    # Identical frames -> 0 motion
    frame1 = np.zeros((100, 100), dtype=np.uint8)
    frame2 = np.zeros((100, 100), dtype=np.uint8)
    motion_zero = SceneAnalyzer.estimate_camera_motion(frame1, frame2)
    assert motion_zero == 0.0

    # Moving pixels -> non-zero motion
    frame2[20:80, 20:80] = 255
    motion_high = SceneAnalyzer.estimate_camera_motion(frame1, frame2)
    assert motion_high > 0.0

def test_scene_analyzer_occlusion_density():
    # Single box -> 0 occlusion
    boxes_single = np.array([[10, 10, 50, 50]])
    occ_zero = SceneAnalyzer.estimate_occlusion_density(boxes_single)
    assert occ_zero == 0.0

    # Two overlapping boxes -> IoU > 0
    boxes_overlap = np.array([
        [10, 10, 50, 50],
        [15, 15, 55, 55]
    ])
    occ_high = SceneAnalyzer.estimate_occlusion_density(boxes_overlap)
    assert occ_high > 0.0

def test_scene_analyzer_tracker_recommendation():
    # High motion -> BoT-SORT
    rec_motion = SceneAnalyzer.recommend_tracker(motion_score=20.0, occlusion_score=0.1)
    assert rec_motion == "botsort"

    # High occlusion -> FastTracker
    rec_occlusion = SceneAnalyzer.recommend_tracker(motion_score=2.0, occlusion_score=0.5)
    assert rec_occlusion == "fasttracker"

    # Low motion & occlusion -> ByteTrack
    rec_default = SceneAnalyzer.recommend_tracker(motion_score=2.0, occlusion_score=0.05)
    assert rec_default == "bytetrack"

def test_multi_tracker_engine_mode_switching():
    engine = MultiTrackerEngine(initial_mode="auto")
    assert engine.mode == "auto"
    
    # Switch to manual botsort
    success, msg = engine.set_mode("botsort")
    assert success is True
    assert engine.mode == "botsort"
    assert engine.active_tracker_name == "botsort"
    
    info = engine.get_info()
    assert info["mode"] == "botsort"
    assert "BoT-SORT" in info["status_label"]
    assert "[Manual]" in info["status_label"]

    # Switch back to auto
    success, msg = engine.set_mode("auto")
    assert success is True
    assert engine.mode == "auto"

def test_multi_tracker_engine_update_with_detections():
    engine = MultiTrackerEngine(initial_mode="auto")
    
    # Create dummy Detections
    boxes = np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float32)
    confidence = np.array([0.9], dtype=np.float32)
    class_id = np.array([0], dtype=int)
    
    dets = sv.Detections(
        xyxy=boxes,
        confidence=confidence,
        class_id=class_id
    )
    
    tracked_dets, status_label = engine.update_with_detections(dets)
    assert tracked_dets is not None
    assert "ByteTrack" in status_label or "Auto" in status_label
