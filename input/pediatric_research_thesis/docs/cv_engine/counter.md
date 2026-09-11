# `src/engine/counter.py`

## Purpose
`counter.py` provides the `Counter` class, which computes real-time patient occupancy counts, tracks spatial centroids for untracked detections, updates cumulative daily totals, and evaluates clinical overcrowding status.

---

## Core Attributes

* **`tracker_manager`**: Reference to `TrackerManager` instance.
* **`state`**: Reference to shared `TelemetryState` Pydantic instance.
* **`active_centroids`**: Dictionary mapping `track_id -> (cx, cy, class_id)` for tracking active object positions.
* **`seen_ids`**: Set of all object tracking IDs seen since server launch.

---

## Methods

### `update_counts(active_tracks: List[Tuple[int, int]], current_centroids: Dict[int, Tuple[float, float, int]])`
Updates the real-time occupancy counts:
1. Iterates over active object tracking IDs and class IDs (`0` for adult, `1` for child).
2. Registers new tracking IDs in `seen_ids` and updates cumulative daily totals (`total_daily_adults`, `total_daily_children`).
3. Updates `active_centroids` dictionary.
4. Purges expired object IDs via `tracker_manager.clean_expired_ids()`.
5. Syncs results directly into `TelemetryState` attributes (`current_adults`, `current_children`).
6. Evaluates overcrowding alert condition.

### `is_overcrowded() -> bool`
Returns `True` if `current_children` exceeds the calculated threshold based on `WAITING_ROOM_CAPACITY` and `PEDIATRIC_ALERT_THRESHOLD_PERCENT`.

---

## Spatial Centroid Tracking

When ByteTrack fails to assign an ID to a detection (`track_id == -1`), `Counter` provides its active centroid history to `Detector` to perform spatial matching within `UNTRACKED_SPATIAL_MATCH_RADIUS`, preventing transient ID drops.
