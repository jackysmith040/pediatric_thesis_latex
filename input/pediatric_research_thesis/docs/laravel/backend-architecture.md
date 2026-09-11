> [!IMPORTANT]
> **[HISTORICAL / ARCHIVED DOCUMENTATION]**  
> This document describes the initial dual-stack architecture (Laravel + FastAPI) used in Phase 1 and Phase 2. The project was subsequently refactored into a single **NiceGUI Python Monolith** (Phase 3+). Refer to [docs/architecture.md](../architecture.md) for the active monolith system architecture.

# Legacy Laravel Backend Architecture

## Overview (Phase 1/2)
In the initial dual-stack prototype, Laravel served as the web command center receiving HTTP telemetry payloads posted by an external FastAPI CV engine.

## Archived Components
- **Laravel REST API Endpoints**: Accepted telemetry payloads from `telemetry.py`.
- **Livewire Components**: Rendered real-time counts received over WebSockets.
- **Laravel Reverb**: WebSocket server broadcasting telemetry updates to browser clients.

## Reason for Pivot
The dual-stack approach introduced inter-process networking overhead across localhost, required maintaining two runtime environments (PHP 8.4 + Python 3.14), and added latency. Transitioning to the NiceGUI Monolith consolidated vision processing, telemetry state, and UI rendering into a unified Python application.
