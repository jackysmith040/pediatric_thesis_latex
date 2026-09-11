> [!IMPORTANT]
> **[HISTORICAL / ARCHIVED DOCUMENTATION]**  
> This document describes the legacy Laravel Reverb WebSocket setup used in Phase 1 and Phase 2. In Phase 3, WebSockets were replaced by in-process Pydantic state binding in the **NiceGUI Monolith**. Refer to [docs/architecture.md](../architecture.md) for current architecture details.

# Legacy Laravel Reverb & Echo Setup

## Overview (Phase 1/2)
Laravel Reverb was configured to broadcast telemetry updates (`TelemetryUpdated` event) to the frontend using Laravel Echo.

## Archived Flow
1. FastAPI CV engine posted metrics to `/api/v1/telemetry`.
2. Laravel controller dispatched `TelemetryUpdated` event.
3. Reverb broadcast event to channel `pediatric-telemetry`.
4. Browser client updated Livewire components.

## Current Monolith Replacement
NiceGUI binds directly to the in-memory `TelemetryState` model, delivering real-time UI updates over Uvicorn without needing Reverb, Echo, or external WebSocket servers.
