import time
from collections import Counter as PyCounter, defaultdict
from typing import Dict, Any, List
from src.engine.config import settings

class TrackerManager:
    def __init__(self, expiry_seconds: int):
        self.expiry_seconds = expiry_seconds
        # Dictionary structure: { id: {"last_seen": float, "class_id": int, "counted": bool} }
        self.tracked_ids: Dict[int, Dict[str, Any]] = {}
        # Temporal class voting history per track_id
        self.class_history: Dict[int, List[int]] = defaultdict(list)

    def update_id(self, track_id: int, class_id: int) -> bool:
        """
        Updates the tracker memory with the given ID and registers class vote.
        Returns True if this ID is newly registered.
        """
        current_time = time.time()
        is_new = False

        # Accumulate class vote in rolling window
        history = self.class_history[track_id]
        history.append(class_id)
        max_window = getattr(settings, 'TEMPORAL_VOTING_WINDOW', 15)
        if len(history) > max_window:
            history.pop(0)

        voted_class = self.get_voted_class(track_id)
        
        if track_id not in self.tracked_ids:
            self.tracked_ids[track_id] = {
                "last_seen": current_time,
                "class_id": voted_class,
                "counted": True
            }
            is_new = True
        else:
            self.tracked_ids[track_id]["last_seen"] = current_time
            self.tracked_ids[track_id]["class_id"] = voted_class
            
        return is_new

    def get_voted_class(self, track_id: int) -> int:
        """Returns the majority voted class over recent frames for a track ID."""
        history = self.class_history.get(track_id, [])
        if not history:
            return settings.CHILD_CLASS_ID
        counts = PyCounter(history)
        return counts.most_common(1)[0][0]

    def get_active_ids(self, max_idle_seconds: float = 1.5) -> Dict[int, int]:
        """
        Returns a dict of currently active in-frame {track_id: class_id}
        that have been seen within max_idle_seconds.
        """
        current_time = time.time()
        return {
            tid: data["class_id"] 
            for tid, data in self.tracked_ids.items()
            if (current_time - data["last_seen"]) <= max_idle_seconds
        }

    def clean_expired_ids(self):
        """
        Removes IDs that have not been seen for 'expiry_seconds'.
        """
        current_time = time.time()
        expired_ids = [
            tid for tid, data in self.tracked_ids.items()
            if (current_time - data["last_seen"]) > self.expiry_seconds
        ]
        for tid in expired_ids:
            del self.tracked_ids[tid]
            if tid in self.class_history:
                del self.class_history[tid]

