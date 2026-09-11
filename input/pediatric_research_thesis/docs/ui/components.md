# `src/ui/components.py`

## Purpose
`components.py` defines reusable UI context managers and styling primitives for the Impeccable design system.

---

## Design System Tokens

- **Primary Accent**: `#4f46e5` (Indigo-600)
- **Background Layer**: `slate-950`
- **Surface Cards**: `slate-900` / `slate-800`
- **Text Hierarchy**: Primary (`white`), Secondary (`slate-400`), Muted (`slate-500`)

---

## Context Managers & Utilities

* **`page_container(scrollable: bool = False)`**: Standard full-screen wrapper applying dark background and padding.
* **`navbar(title: str)`**: Sticky top navigation bar with system active pulsing status blip.
* **`dashboard_card(max_width='max-w-2xl')`**: Rounded 2xl dark card container.
* **`video_feed_card(title: str)`**: Bounded container for video streaming HTML element.
* **`stat_card(title: str, obj, attr: str, is_primary: bool)`**: Metric card bound to telemetry property.
* **`alert_banner(title: str, message: str, obj, attr: str)`**: Red alert banner bound to boolean visibility property.
