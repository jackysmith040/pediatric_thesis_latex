import logging
import time
import math
from src.engine.tracker_manager import TrackerManager
from src.engine.config import settings
from src.state.telemetry import TelemetryState

logger = logging.getLogger(__name__)

class Counter:
    def __init__(self, tracker_manager: TrackerManager, state: TelemetryState):
        self.tracker_manager = tracker_manager
        self.state = state
        self.active_centroids = {} # track_id -> (cx, cy)
        self.lost_centroids = [] # list of (cx, cy, timestamp, class_id)
        self.spatial_threshold = 150 # pixels (tune based on resolution)
        self.temporal_threshold = 5.0 # seconds
        self.adult_class_id = getattr(settings, 'ADULT_CLASS_ID', 0)
        self.child_class_id = getattr(settings, 'CHILD_CLASS_ID', 1)
        
    @property
    def current_adults(self) -> int:
        if self.adult_class_id == -1:
            return 0
        active_ids = self.tracker_manager.get_active_ids(max_idle_seconds=1.5)
        return sum(1 for class_id in active_ids.values() if class_id == self.adult_class_id)

    @property
    def current_children(self) -> int:
        if self.child_class_id == -1:
            return 0
        active_ids = self.tracker_manager.get_active_ids(max_idle_seconds=1.5)
        return sum(1 for class_id in active_ids.values() if class_id == self.child_class_id)


    def process_detection(self, track_id: int, class_id: int, box):
        """
        Process a single detection track with Spatial Debouncing to prevent double counting.
        """
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        
        is_new = self.tracker_manager.update_id(track_id, class_id)
        
        if is_new:
            # Spatial Debounce Check
            current_time = time.time()
            # Clean old lost centroids
            self.lost_centroids = [lc for lc in self.lost_centroids if current_time - lc[2] <= self.temporal_threshold]
            
            # Check for a match
            matched = False
            for i, (lcx, lcy, ts, lcid) in enumerate(self.lost_centroids):
                if lcid == class_id:
                    dist = math.hypot(cx - lcx, cy - lcy)
                    if dist < self.spatial_threshold:
                        matched = True
                        self.lost_centroids.pop(i) # Consume this lost centroid
                        break
            
            if not matched:
                if self.adult_class_id != -1 and class_id == self.adult_class_id:
                    self.state.total_daily_adults += 1
                elif self.child_class_id != -1 and class_id == self.child_class_id:
                    self.state.total_daily_children += 1

        
        self.active_centroids[track_id] = (cx, cy, class_id)
                
        # Always update the state's current counts so NiceGUI updates live
        self.state.current_adults = self.current_adults
        self.state.current_children = self.current_children
        self.state.overcrowding_alert = self.is_overcrowded()
        
    def cleanup_lost_tracks(self):
        """Move tracks that expired from active memory into lost centroids for debouncing."""
        active_ids = self.tracker_manager.get_active_ids()
        current_time = time.time()
        
        expired_track_ids = []
        for tid, (cx, cy, class_id) in list(self.active_centroids.items()):
            if tid not in active_ids:
                # We lost this track, save it for debouncing
                self.lost_centroids.append((cx, cy, current_time, class_id))
                expired_track_ids.append(tid)
                
        for tid in expired_track_ids:
            del self.active_centroids[tid]
                
    def is_overcrowded(self) -> bool:
        if settings.WAITING_ROOM_CAPACITY <= 0:
            return False
        child_percentage = (self.current_children / settings.WAITING_ROOM_CAPACITY) * 100
        return child_percentage >= settings.PEDIATRIC_ALERT_THRESHOLD_PERCENT
