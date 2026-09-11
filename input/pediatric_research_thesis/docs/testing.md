# 🧪 Unit Testing & Verification Guide

This document outlines the testing strategy, test suites, mocking patterns, and verification commands for **The Invisible Child**.

---

## 1. Test Framework

The project uses [Pytest](https://docs.pytest.org/) for automated unit testing. Tests are located in `tests/unit/`.

---

## 2. Test Suites

### `tests/unit/test_counter.py`
Tests metrics aggregation, tracking ID registration, daily counters, and overcrowding alert logic:
- `test_counter_initialization`: Verifies zeroed default counts.
- `test_update_counts_registers_new_ids_and_updates_daily_totals`: Verifies cumulative total increments.
- `test_update_counts_purges_expired_ids`: Verifies expired ID removal via `TrackerManager`.
- `test_overcrowding_alert_logic`: Verifies alert triggering when pediatric ratio exceeds threshold.

### `tests/unit/test_detector.py`
Tests spatial centroid matching, synthetic ID assignment, and camera source switching:
- `test_resolve_track_id_returns_original_if_valid_and_unclaimed`: Verifies standard ByteTrack ID resolution.
- `test_resolve_track_id_matches_existing_active_centroid_if_unclaimed`: Verifies untracked detection centroid matching (`track_id == -1`).
- `test_resolve_track_id_prevents_stolen_claimed_ids`: Verifies stolen ID prevention.
- `test_multiple_untracked_detections_receive_distinct_ids`: Verifies distinct synthetic ID generation.
- `test_detector_change_source_success`: Verifies source switching mechanics.

### `tests/unit/test_stream_resolver.py`
Tests source string parsing and resolution:
- `test_is_youtube_url_detection`: Tests YouTube URL detection.
- `test_resolve_webcam_0`: Tests default webcam resolution.
- `test_resolve_preset_streams_list`: Verifies `PRESET_TEST_STREAMS` structure and entries.

---

## 3. Running Unit Tests

To run all unit tests using the project virtual environment:

```bash
.\.venv\Scripts\python.exe -m pytest
```

Output:
```text
collected 18 items

tests\unit\test_counter.py .....                                         [ 27%]
tests\unit\test_detector.py ........                                     [ 72%]
tests\unit\test_stream_resolver.py .....                                 [100%]

======================== 18 passed, 1 warning in 3.32s ========================
```
