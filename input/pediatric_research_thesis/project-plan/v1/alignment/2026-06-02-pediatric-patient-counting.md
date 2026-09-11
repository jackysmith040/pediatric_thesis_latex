## Alignment brief — Pediatric Patient Counting V1
**Date:** 2026-06-02
**Problem:** Resolving the "Invisible Child" phenomenon in resource-constrained hospital waiting environments by accurately enumerating carried, occluded infants using localized Computer Vision (Python/YOLO) and displaying real-time actionable metrics on a Laravel administrative dashboard via decoupled architecture.

### Locked IN
- Python/YOLO CV engine analyzing video locally and sending telemetry data via REST.
- Single camera stream (e.g., smartphone feed via DroidCam) for MVP.
- Laravel Dashboard running locally (Herd/Nginx) displaying live MJPEG stream and real-time WebSocket metrics.
- Native Laravel Reverb for WebSockets (100% PHP, Free, Pusher-compatible for future mobile/desktop apps).
- Use of Laravel Boost and Laravel MCP for package installation and maintaining high code quality.
- Automated nightly PDF analytical report generation.
- UI optimized primarily for a large external monitor (1080p landscape).

### Locked OUT (non-goals)
- Multiple simultaneous camera streams (Deferred).
- Advanced hospital authentication or Active Directory (V1 is open/local development only).
- Responsive mobile dashboard views (Optimized for 1080p TV first).
- Complex retry/caching mechanisms for CV telemetry (If API goes down, it fails open and drops logs).

### Decisions
| Topic | Decision | Rationale |
|-------|----------|-----------|
| CV Engine Networking | Fail-open / Ephemeral | CV engine drops logs if dashboard is offline to avoid buffering memory leaks and keep real-time performance. |
| Dashboard UI | TV Monitor Optimized | Dashboard will be displayed on a public/admin screen in the waiting area, not actively interacted with on mobile. |
| Authentication | No Auth | Accelerates MVP validation; safe since it runs on a local Windows unified environment. |

### Risks & mitigations
| Risk | Mitigation |
|------|------------|
| MJPEG stream latency blocking UI | Direct HTML `<img>` tag embedding bypasses Laravel processing entirely, keeping Laravel lightweight. |
| Real-time WebSocket lag | Using native Laravel Reverb for highly performant local websocket broadcasting without external providers like Pusher. |

### Definition of done
- [ ] CV engine connects to a single video feed and processes YOLO "Adult vs Child" bounding boxes.
- [ ] Live MJPEG stream displays correctly on the Laravel dashboard using port 5000.
- [ ] Restify accepts JSON POSTs and Reverb pushes metrics to the dashboard in real-time.
- [ ] Nightly PDF is generated successfully using dummy or live tracking data.

### Open questions
- None.

### Persona sync
- **Stability:** High. Rely on proven Laravel + YOLO patterns.
- **Feature creep:** Strict. Reject/defer any out-of-scope ideas to V2 unless explicit expand is given.
- **Next skill:** `senior-stable-delivery` → Rabit plan table → implement
