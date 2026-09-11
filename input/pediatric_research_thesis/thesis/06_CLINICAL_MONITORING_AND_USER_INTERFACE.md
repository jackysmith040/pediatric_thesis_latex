# Chapter 6: Clinical Intelligence, Capacity Analytics & User Interface

---

## 6.1 Clinical Telemetry Metric Computation

The clinical intelligence layer translates low-level object bounding boxes and spatial tracking trajectories into actionable hospital occupancy statistics.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CLINICAL TELEMETRY STATE MODEL                       │
│                                                                         │
│  Metric Name          Type       Clinical Meaning & Source              │
│  ─────────────────────────────────────────────────────────────────────  │
│  current_children     int        Instantaneous pediatric occupancy      │
│  current_adults       int        Instantaneous adult caregiver count    │
│  total_daily_children int        Cumulative unique pediatric intake     │
│  total_daily_adults   int        Cumulative unique adult intake         │
│  overcrowding_alert   bool       True if pediatric load >= 30% capacity │
│  camera_id            str        Unique clinical CCTV identifier        │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.1.1 Instantaneous Occupancy Computation
To prevent temporary occlusions or visual flickers from creating erratic fluctuations on the dashboard, instantaneous counts are computed dynamically via active track filtering:

```python
@property
def current_children(self) -> int:
    if self.child_class_id == -1:
        return 0
    active_ids = self.tracker_manager.get_active_ids(max_idle_seconds=1.5)
    return sum(1 for class_id in active_ids.values() if class_id == self.child_class_id)
```

#### Mathematical Logic:
Let $\mathcal{T} = \{ (\text{ID}_k, c_k, t_{\text{last\_seen}, k}) \}$ be the set of all active tracking records maintained by `TrackerManager`. The active subset $\mathcal{T}_{\text{active}}(t)$ at current timestamp $t$ is:

$$\mathcal{T}_{\text{active}}(t) = \left\{ (\text{ID}_k, c_k) \in \mathcal{T} \mid (t - t_{\text{last\_seen}, k}) \le 1.5\text{ seconds} \right\}$$

$$\text{current\_children}(t) = \sum_{(\text{ID}_k, c_k) \in \mathcal{T}_{\text{active}}(t)} \mathbb{I}(c_k = \text{child\_class\_id})$$

$$\text{current\_adults}(t) = \sum_{(\text{ID}_k, c_k) \in \mathcal{T}_{\text{active}}(t)} \mathbb{I}(c_k = \text{adult\_class\_id})$$

Where $\mathbb{I}(\cdot)$ is the indicator function. The $1.5\text{-second}$ idle grace window ensures that when an infant is briefly obscured by a moving nurse or clipboard, the displayed tally remains rock-solid.

---

## 6.2 Pediatric Overcrowding Warning Logic & Alert Triggers

In emergency department triage, pediatric patients require significantly more clinical nursing attention per capita than adult patients. Pediatric overcrowding drastically increases the risk of unobserved respiratory arrest and dehydration shock.

### 6.2.1 Mathematical Formulation of Overcrowding Warning
The overcrowding threshold is parameterized in `src/engine/config.py`:
- `WAITING_ROOM_CAPACITY` ($K_{\text{cap}} = 50\text{ persons}$)
- `PEDIATRIC_ALERT_THRESHOLD_PERCENT` ($\theta_{\text{alert}} = 30.0\%$)

$$\text{Pediatric Load Percentage } C_{\text{child}}\%(t) = \left( \frac{\text{current\_children}(t)}{K_{\text{cap}}} \right) \times 100\%$$

$$\text{overcrowding\_alert}(t) = \begin{cases} \text{True} & \text{if } K_{\text{cap}} > 0 \land C_{\text{child}}\%(t) \ge \theta_{\text{alert}} \\ \text{False} & \text{otherwise} \end{cases}$$

```python
def is_overcrowded(self) -> bool:
    if settings.WAITING_ROOM_CAPACITY <= 0:
        return False
    child_percentage = (self.current_children / settings.WAITING_ROOM_CAPACITY) * 100
    return child_percentage >= settings.PEDIATRIC_ALERT_THRESHOLD_PERCENT
```

### 6.2.2 Clinical Alert Propagation
When `overcrowding_alert` triggers:
1. **Visual Banner Activation:** The clinical dashboard immediately renders a high-visibility amber/red alert banner (`"WARNING: Pediatric Patient Surge Detected: Escalate Triage Staffing"`).
2. **Badge Morphing:** The pediatric counter card transitions from calm cyan to pulsing alert styling.
3. **Telemetry Flagging:** The boolean flag is recorded in all downstream CSV and PDF audit logs to document the exact duration of clinical capacity strain.

---

## 6.3 Clinical Design System & Dark-Mode Ergonomics

Hospital clinical workstations operating across 24-hour shifts require thoughtful user interface ergonomics. Bright white dashboards cause visual fatigue and disrupt circadian alertness during night shifts.

The system implements a **Dark-Mode Clinical Design System** (`src/ui/components.py`):
- **Background Palette:** Deep slate (`bg-slate-950` / `#020617`) paired with elevated card surfaces (`bg-slate-900` / `#0f172a`) and subtle borders (`border-slate-800`).
- **Semantic Color Tokens:**
  - **Electric Cyan (`text-cyan-400` / `#22d3ee`):** Denotes Pediatric / Child metrics.
  - **Clinical Indigo (`text-indigo-400` / `#818cf8`):** Denotes Adult Caregiver metrics.
  - **Emerald Green (`text-emerald-400` / `#34d399`):** Denotes healthy capacity and active video streaming status.
  - **Rose / Amber (`text-rose-500` / `text-amber-400`):** Denotes overcrowding alarms and hardware warnings.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CLINICAL DASHBOARD WIREFRAME                         │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ 🏥 THE INVISIBLE CHILD  -  CLINICAL COMMAND CENTER      [🔴 LIVE]   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────┐  ┌────────────────────────────────┐  │
│  │ LIVE CAMERA FEED              │  │ REAL-TIME CLINICAL METRICS     │  │
│  │                               │  │                                │  │
│  │   ┌───────────────────────┐   │  │  ┌──────────────────────────┐  │  │
│  │   │ Child #1 [ID: 104]    │   │  │  │ PEDIATRIC PATIENTS       │  │  │
│  │   │ (Sci-Fi Brackets)     │   │  │  │        04                │  │  │
│  │   │  o---o (Trail)        │   │  │  └──────────────────────────┘  │  │
│  │   └───────────────────────┘   │  │  ┌──────────────────────────┐  │  │
│  │                               │  │  │ ADULT CAREGIVERS         │  │  │
│  │                               │  │  │        12                │  │  │
│  │                               │  │  └──────────────────────────┘  │  │
│  │                               │  │  ┌──────────────────────────┐  │  │
│  │                               │  │  │ OVERCROWDING ALARM: OK   │  │  │
│  │                               │  │  └──────────────────────────┘  │  │
│  │  └───────────────────────────────┘  └────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ CONTROLS: [AI Model Select ▼] [Camera Source ▼] [Export PDF / CSV]│  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.3.1 Video Annotation Rendering Style
To ensure video overlays do not obscure critical medical observations:
- **Translucent Fill:** Bounding boxes feature a 15% semi-transparent fill (`cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)`).
- **Corner Brackets:** Modern 15-pixel corner bracket lines frame the target without solid, intrusive borders.
- **Fading Centroid Motion Trails:** 30-frame temporal history trails with quadratic thickness fading (`thickness = int(sqrt(i) * 0.8) + 1`) and a double-ring centroid dot visualize patient movement pathways.

---

## 6.4 Multi-Model Comparative Evaluation Lab (`/video-test`)

In addition to the operational command center (`/dashboard`), the system provides a specialized **Interactive Evaluation & Stream Testing Lab** (`src/ui/evaluation.py`):
- **Live Side-by-Side Model Swapping:** Allows clinicians, researchers, and audit personnel to dynamically toggle between:
  1. Base Pretrained YOLO26s ($M_1$)
  2. Traditional Fine-Tuned YOLO26s ($M_2$)
  3. DINOv3 Distilled YOLO26s ($M_3$ - SOTA)
  4. Sliced SAHI Configurations ($M_4, M_5, M_6$)
- **Real-Time Feature Inspection:** Displays bounding box confidence histograms, IoU distribution maps, CLAHE contrast toggles, and live FPS/latency diagnostics.

---

## 6.5 Automated Audit Logging: CSV Telemetry & Formatted PDF Reports

Hospital quality assurance and regulatory accreditation require structured audit trails of emergency department waiting room occupancy.

The system incorporates an automated reporting engine (`src/engine/reporter.py`):

```
       Clinical Telemetry State + Historical Logs
                         │
           ┌─────────────┴─────────────┐
           ▼                           ▼
     CSV Telemetry Log           Formatted PDF Report
     (src/engine/reporter.py)    (FPDF Clinical Template)
           │                           │
           ▼                           ▼
     Structured Data             Executive Document
     - Timestamp                 - Hospital Header
     - Current Child/Adult       - Peak Occupancy Summary
     - Cumulative Totals         - Overcrowding Incidents
     - Overcrowding Flag         - Clinical Recommendations
```

### 6.5.1 Structured CSV Telemetry Export (`Reporter.generate_csv_log`)
Generates standardized tabular telemetry logs formatted for ingestion by Hospital Information Systems (HIS) and data analysis tools:
```csv
timestamp,camera_id,current_adults,current_children,total_daily_adults,total_daily_children,overcrowding_alert
2026-06-01 14:30:00,outpatient_waiting_cctv_01,12,4,145,62,False
2026-06-01 14:35:00,outpatient_waiting_cctv_01,18,16,163,78,True
```

### 6.5.2 Clinical PDF Capacity Audit Report (`Reporter.generate_pdf_report`)
Utilizes `FPDF` to construct publication-ready clinical capacity audits:
- **Header:** Clinical facility name, date/time stamp, camera identifier, and supervising triage officer metadata.
- **Summary Metrics Table:** Peak pediatric occupancy, total daily patients, average pediatric-to-adult ratio, and total overcrowding alert durations.
- **Operational Recommendations:** Automated heuristic recommendations (e.g., *"Pediatric capacity exceeded 30% for 45 cumulative minutes. Recommend deploying +2 dedicated pediatric triage nurses during 14:00 - 17:00 shift."*).
- **Native Download Integration:** In PyWebView desktop mode, files are written directly to `Downloads/`; in browser mode, files are delivered via standard HTTP file attachment triggers.
