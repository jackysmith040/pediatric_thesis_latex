# `src/state/telemetry.py`

## Purpose
`telemetry.py` defines the `TelemetryState` Pydantic model, which serves as the single source of truth for in-memory telemetry state.

---

## Schema Fields

```python
class TelemetryState(BaseModel):
    current_children: int = Field(default=0, description="Current number of children in view")
    current_adults: int = Field(default=0, description="Current number of adults in view")
    total_daily_children: int = Field(default=0, description="Total children processed today")
    total_daily_adults: int = Field(default=0, description="Total adults processed today")
    overcrowding_alert: bool = Field(default=False, description="Whether overcrowding alert is active")
```

---

## Usage

Instantiated as a global singleton in `src/main.py`. The `Counter` updates attributes in place during frame inference, and NiceGUI components reactively read and render the fields.
