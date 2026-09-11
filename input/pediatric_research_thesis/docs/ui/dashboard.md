# `src/ui/dashboard.py`

## Purpose
`dashboard.py` implements the primary **Clinical Command Center** interface (`/dashboard`).

---

## Layout Sections

1. **Top Navbar**: Displays site header, back navigation button, and green system active status blip.
2. **Left Column (Video Feed Card)**:
   - Contains an HTML `<img>` tag pointing to `/camera/stream` for zero-overhead MJPEG video rendering.
   - Active source status badge displaying current active stream name (e.g. `Webcam 0`).
   - Action controls: "Turn On Live Camera" (activates Webcam 0) and "External Video Testing" (navigates to `/video-test`).
3. **Right Column (Telemetry Sidebar)**:
   - **Alert Banner**: Overcrowding warning banner automatically displayed when `overcrowding_alert` is `True`.
   - **Real-Time Stat Cards**: Live counts for Pediatric and Adult patients.
   - **Daily Aggregate Progress Bars**: Visual ratio of total daily children vs. adults processed today.
   - **Report Generators**: Async buttons triggering `reporter.generate_csv()` and `reporter.generate_pdf()` downloads.
