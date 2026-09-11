# 🎨 UI & Frontend Architecture Overview

The UI layer of **The Invisible Child** is built completely in Python using **NiceGUI**, running on FastAPI.

---

## 1. Page Routes & Navigation

| Route | Function | Module | Description |
|---|---|---|---|
| `/` | Landing Page | `src/ui/landing.py` | Clinical overview, key features, architecture specs, and entry buttons |
| `/dashboard` | Command Center | `src/ui/dashboard.py` | Live video feed, real-time telemetry metrics, webcam toggle controls, report downloads |
| `/video-test` | Evaluation Lab | `src/ui/evaluation.py` | Multi-source stream testing lab supporting YouTube feeds, local OPD MP4 video, and custom stream URLs |
| `/camera/stream` | Stream Endpoint | `src/main.py` | Native MJPEG stream endpoint consumed by standard HTML `<img>` elements |

---

## 2. Dynamic UI & State Binding

* **Native Binding**: Statistics cards and linear progress bars bind directly to `TelemetryState` attributes (`current_children`, `current_adults`, `total_daily_children`, `total_daily_adults`, `overcrowding_alert`).
* **Zero WebSockets**: Because NiceGUI runs in-process with the Python CV engine, state updates occur in memory and are sent to the client over NiceGUI's internal socket connection without REST polling or external messaging brokers.
* **Native Desktop Support**: In PyWebView desktop mode, report exports (CSV/PDF) save directly into the local `Downloads/` directory via native filesystem calls.
