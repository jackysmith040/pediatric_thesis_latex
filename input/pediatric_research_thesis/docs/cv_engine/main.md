# `src/main.py`

## Purpose
`main.py` is the application entry point for the Pediatric Monitor Monolith. It bootstraps NiceGUI and FastAPI, initializes global state and engine instances, mounts HTTP routes, and registers UI pages.

---

## Initialization Flow

1. **Global Singletons**:
   - `telemetry_state = TelemetryState()`
   - `tracker_manager = TrackerManager(...)`
   - `counter = Counter(...)`
   - `detector = None` (initialized on startup)

2. **Lifespan Hooks**:
   - `@app.on_startup`: Instantiates `Detector(counter=counter)` to launch video capture and inference threads.
   - `@app.on_shutdown`: Calls `detector.release()` to safely terminate capture threads and release camera hardware locks.

3. **MJPEG Stream Endpoint (`/camera/stream`)**:
   - Returns a `StreamingResponse` (`multipart/x-mixed-replace`) serving live annotated JPEG frame bytes generated asynchronously by `detector.get_latest_jpeg_bytes()`.

4. **Page Registration**:
   - `register_landing()` -> Route `/`
   - `register_dashboard(telemetry_state, lambda: detector)` -> Route `/dashboard`
   - `register_evaluation(telemetry_state, lambda: detector)` -> Route `/video-test`

5. **Server Launch**:
   - Executes `ui.run(title="Pediatric Clinical Command Center")` serving port 8080.
