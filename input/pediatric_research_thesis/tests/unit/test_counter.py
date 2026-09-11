import pytest
import time
from unittest.mock import Mock

from src.engine.counter import Counter
from src.engine.tracker_manager import TrackerManager
from src.state.telemetry import TelemetryState
from src.engine.config import settings

@pytest.fixture
def mock_counter():
    tracker_manager = TrackerManager(expiry_seconds=30)
    state = TelemetryState()
    # Ensure constants match expected values during tests
    settings.ADULT_CLASS_ID = 0
    settings.CHILD_CLASS_ID = 1
    return Counter(tracker_manager, state)

def test_initial_state(mock_counter):
    assert mock_counter.current_adults == 0
    assert mock_counter.current_children == 0
    assert mock_counter.state.total_daily_adults == 0
    assert mock_counter.state.total_daily_children == 0

def test_process_detection_adds_new(mock_counter):
    # Box format: (x1, y1, x2, y2) -> Centroid = (20, 20)
    box = (10, 10, 30, 30)
    mock_counter.process_detection(track_id=1, class_id=0, box=box)
    
    assert mock_counter.current_adults == 1
    assert mock_counter.state.total_daily_adults == 1
    assert 1 in mock_counter.active_centroids
    assert mock_counter.active_centroids[1] == (20.0, 20.0, 0)

def test_spatial_debouncing_prevents_double_count(mock_counter):
    """Test that a track dropped and picked up with a new ID in the same spot isn't double counted."""
    # 1. Add initial track
    box1 = (10, 10, 30, 30) # Centroid (20, 20)
    mock_counter.process_detection(track_id=1, class_id=0, box=box1)
    
    assert mock_counter.state.total_daily_adults == 1
    
    # 2. Expire track 1 (Simulate tracking loss)
    # Fast forward tracker manager internally to drop the track
    mock_counter.tracker_manager.tracked_ids.clear()
    
    # Run cleanup to move to lost_centroids
    mock_counter.cleanup_lost_tracks()
    
    assert len(mock_counter.lost_centroids) == 1
    assert 1 not in mock_counter.active_centroids
    
    # 3. Add new track in almost exact same location (dist < 150)
    box2 = (15, 15, 35, 35) # Centroid (25, 25), dist = 7.07
    mock_counter.process_detection(track_id=2, class_id=0, box=box2)
    
    # Total shouldn't increase because it was debounced!
    assert mock_counter.state.total_daily_adults == 1
    # The lost centroid should have been consumed
    assert len(mock_counter.lost_centroids) == 0

def test_spatial_debouncing_allows_distant_new_track(mock_counter):
    """Test that a track appearing far away IS counted as a new person."""
    # 1. Add initial track
    box1 = (10, 10, 30, 30) # Centroid (20, 20)
    mock_counter.process_detection(track_id=1, class_id=0, box=box1)
    
    assert mock_counter.state.total_daily_adults == 1
    
    # 2. Expire track 1
    mock_counter.tracker_manager.tracked_ids.clear()
    mock_counter.cleanup_lost_tracks()
    
    # 3. Add new track VERY FAR away (dist > 150 threshold)
    box2 = (500, 500, 520, 520) # Centroid (510, 510)
    mock_counter.process_detection(track_id=2, class_id=0, box=box2)
    
    # Total SHOULD increase
    assert mock_counter.state.total_daily_adults == 2

def test_temporal_debouncing_expiration(mock_counter):
    """Test that lost centroids expire after temporal_threshold (5.0s)."""
    # 1. Add initial track
    box1 = (10, 10, 30, 30)
    mock_counter.process_detection(track_id=1, class_id=0, box=box1)
    
    # 2. Expire track 1
    mock_counter.tracker_manager.tracked_ids.clear()
    mock_counter.cleanup_lost_tracks()
    
    assert len(mock_counter.lost_centroids) == 1
    
    # Modify the timestamp of the lost centroid to be older than temporal_threshold (5.0s)
    old_lc = mock_counter.lost_centroids[0]
    mock_counter.lost_centroids[0] = (old_lc[0], old_lc[1], time.time() - 10.0, old_lc[3])
    
    # 3. Add new track in same location
    box2 = (15, 15, 35, 35) 
    mock_counter.process_detection(track_id=2, class_id=0, box=box2)
    
    # Total SHOULD increase because the lost centroid expired temporally
    assert mock_counter.state.total_daily_adults == 2

def test_dynamic_class_mapping(mock_counter):
    """Test that Counter dynamically respects child_class_id and adult_class_id overrides (e.g. from fine-tuned models)."""
    mock_counter.child_class_id = 0
    mock_counter.adult_class_id = 1
    
    box = (10, 10, 30, 30)
    mock_counter.process_detection(track_id=1, class_id=0, box=box)
    
    assert mock_counter.current_children == 1
    assert mock_counter.current_adults == 0
    assert mock_counter.state.total_daily_children == 1

def test_kids_only_single_class_counter_behavior(mock_counter):
    """Test that Counter correctly handles single-class Kids-Only models where adult_class_id is disabled (-1)."""
    mock_counter.child_class_id = 0
    mock_counter.adult_class_id = -1
    
    box = (10, 10, 30, 30)
    mock_counter.process_detection(track_id=1, class_id=0, box=box)
    
    assert mock_counter.current_children == 1
    assert mock_counter.current_adults == 0
    assert mock_counter.state.total_daily_children == 1
    assert mock_counter.state.total_daily_adults == 0

