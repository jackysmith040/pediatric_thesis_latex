# The Invisible Child: Pediatric Intelligence

A dual-stack clinical intelligence system connecting a Python FastAPI YOLOv26 engine to a Laravel 13 Dashboard. It counts adults and children in a waiting room and displays it via a real-time Livewire UI.

## Overview
This system acts as an "unsleeping eye" for high-stakes clinical environments like pediatric Emergency Rooms. The core architecture ensures that the intensive CV processing is completely decoupled from the real-time UI, resulting in a zero-latency video feed while metrics update in parallel via Laravel Reverb WebSockets.

## Architecture & Flow
1. **CV Engine (Python/FastAPI):**
   - Connects to an IP camera or webcam stream.
   - Runs a highly optimized `threading` architecture where the raw video frame buffer is streamed instantly, while YOLOv26 processes objects in a separate background thread.
   - Posts telemetry payload (`current_adults`, `current_children`) to the Laravel Backend every 3 seconds.
2. **Backend (Laravel 13):**
   - Ingests telemetry via REST API and stores it in SQLite (`traffic_logs`).
   - Instantly broadcasts an event (`TelemetryReceived`) to the frontend using Laravel Reverb (`echo:telemetry`).
3. **Frontend (Livewire 4 + Alpine.js + TailwindCSS v4):**
   - Subscribes to the Reverb WebSocket to update the "Command Center" dashboard dynamically.
   - Uses `#[Isolate]` (Livewire Islands) so that telemetry data can update instantly without causing the entire video feed UI to freeze or re-render.
   - Includes graceful degradation states (Offline badges) if the CV Engine or Reverb server crashes.

## Quick Start
1. **Start the Laravel Server (Herd):**
   The application is served at `http://pediatric-dashboard.test` via Laravel Herd.
2. **Start Laravel Reverb:**
   ```bash
   php artisan reverb:start
   ```
3. **Start the CV Engine:**
   ```bash
   cd cv_engine
   source .venv/bin/activate
   uvicorn app.main:app --host 127.0.0.1 --port 5001 --reload
   ```

## Documentation
- Read the detailed architecture decisions in `docs/architecture.md`.
- Original project scope and alignment can be found in `project-plan/`.
