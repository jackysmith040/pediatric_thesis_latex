# `src/engine/tracker_manager.py`

## Purpose
`tracker_manager.py` manages active object tracking IDs and purges stale tracks that have left the camera frame.

---

## Core Methods

### `__init__(expiry_seconds: int = 30)`
Initializes the manager with an expiration timeout (default 30 seconds).

### `update_track(track_id: int)`
Updates or inserts the `last_seen` timestamp for the specified `track_id`.

### `clean_expired_ids() -> List[int]`
Identifies object IDs that haven't been seen within `expiry_seconds`, deletes them from internal tracking memory, and returns the list of purged tracking IDs.
