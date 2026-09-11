# `src/engine/reporter.py`

## Purpose
`reporter.py` handles the creation and formatting of downloadable clinical telemetry reports from the shared `TelemetryState`.

---

## Export Functions

### `generate_csv(state: TelemetryState) -> bytes`
Generates a UTF-8 encoded CSV string containing:
- Metric name
- Current value
- ISO timestamp

### `generate_pdf(state: TelemetryState) -> bytes`
Uses `FPDF` to construct a clinical-grade PDF document containing:
- Header banner ("Pediatric Clinical Command Center")
- Generation timestamp
- Real-time occupancy breakdown (Current Pediatric & Adult counts)
- Daily cumulative volumes
- Overcrowding status banner with alert indicators
