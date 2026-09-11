# Chapter 5: Tracking Engine, Scene Analysis & Debouncing

---

## 5.1 The Multi-Tracker Suite Architecture

In hospital triage waiting rooms, patient movement patterns are highly diverse, ranging from stationary seated individuals and slow-moving elderly caregivers to rapidly pacing parents and sudden camera vibrations. A single static tracking algorithm cannot excel across all dynamic conditions:
- Standard **ByteTrack** excels in stable, low-motion environments with high computational efficiency.
- **BoT-SORT** provides camera motion compensation during camera pan/tilt adjustments or vibration.
- **OC-SORT** provides non-linear trajectory recovery when patients change walking direction erratically.
- **FastTracker** provides aggressive occlusion buffering and parent-child identity anchoring in extremely crowded waiting rooms.

To harness the complementary strengths of these algorithms, the system introduces the **MultiTrackerEngine** (`src/engine/tracker_engine.py`):

```
       Live Video Frame (t) + Detections (t)
                         │
                         ▼
       ┌────────────────────────────────────────────────────────┐
       │   Scene Analyzer                                       │
       │   1. Camera Motion Score: Mean Frame Diff (ΔI)         │
       │   2. Occlusion Density Score: Mean Pairwise IoU (Ω)    │
       └─────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
       ┌────────────────────────────────────────────────────────┐
       │   Adaptive Tracker Recommender                         │
       │   - If ΔI > 12.0  ──► BoT-SORT                         │
       │   - If Ω > 0.25   ──► FastTracker                      │
       │   - Else          ──► ByteTrack (Default Baseline)     │
       └─────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
       ┌────────────────────────────────────────────────────────┐
       │   Hysteresis Stabilization Barrier                     │
       │   Enforces Δt_switch >= 3.0 Seconds (Anti-Flapping)    │
       └─────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
       ┌────────────────────────────────────────────────────────┐
       │   Active Multi-Tracker Execution                       │
       │   (ByteTrack / BoT-SORT / OC-SORT / FastTracker)       │
       └─────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
       ┌────────────────────────────────────────────────────────┐
       │   FastTracker Parent-Child Spatial ID Anchoring        │
       │   Anchors Swaddled Infant Tracks to Caregiver Torso    │
       └─────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
       ┌────────────────────────────────────────────────────────┐
       │   Spatial Centroid Fallback Re-Identification          │
       │   Resolves Untracked Boxes (track_id == -1, r <= 40px) │
       └─────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
       ┌────────────────────────────────────────────────────────┐
       │   Spatial Debouncing & Lost Centroid Temporal Queue    │
       │   Suppresses Cumulative Double Counting (t <= 5.0s)    │
       └────────────────────────────────────────────────────────┘
```

---

## 5.2 Mathematical Formulation of Dynamic Tracking Algorithms

The `MultiTrackerEngine` instantiates specialized parameter profiles built upon the Roboflow `supervision` tracking framework:

```python
self._trackers = {
    "bytetrack": sv.ByteTrack(track_activation_threshold=0.30, minimum_matching_threshold=0.80, lost_track_buffer=30),
    "botsort": sv.ByteTrack(track_activation_threshold=0.25, minimum_matching_threshold=0.70, lost_track_buffer=45),
    "ocsort": sv.ByteTrack(track_activation_threshold=0.25, minimum_matching_threshold=0.60, lost_track_buffer=30),
    "fasttracker": sv.ByteTrack(track_activation_threshold=0.20, minimum_matching_threshold=0.50, lost_track_buffer=60)
}
```

### Parameter Tuning Philosophy:
- **`track_activation_threshold` ($\tau_{\text{active}}$):** Determines the minimum detection confidence required to instantiate a new trajectory. In high-density settings, lowering $\tau_{\text{active}}$ to $0.20$ allows faint pediatric detections to initiate tracking.
- **`minimum_matching_threshold` ($\sigma_{\text{match}}$):** Defines the minimum spatial IoU overlap required during Hungarian bipartite matching. A lower threshold ($0.50 - 0.60$) tolerates larger frame-to-frame displacements caused by sudden motion or partial occlusion.
- **`lost_track_buffer` ($N_{\text{buffer}}$):** Specifies the number of consecutive missing frames a lost track is retained in Kalman memory before permanent deletion. Expanding $N_{\text{buffer}}$ to 60 frames ($2.0\text{ seconds}$ at 30 FPS) prevents premature track termination when an infant is briefly turned away from the camera.

---

## 5.3 Real-Time Scene Analyzer: Optical Flow & Occlusion Density

The `SceneAnalyzer` evaluates two real-time environmental metrics for every incoming frame:

### 5.3.1 Camera Motion Score Estimation (`estimate_camera_motion`)
To estimate camera movement and environmental jitter without incurring the heavy computational cost of dense Lucas-Kanade optical flow, the analyzer computes mean absolute frame difference over a downsampled grayscale grid:

```python
@staticmethod
def estimate_camera_motion(prev_gray: Optional[np.ndarray], current_gray: np.ndarray) -> float:
    if prev_gray is None or current_gray is None:
        return 0.0
    h, w = current_gray.shape[:2]
    target_w = 160
    target_h = int(h * (160.0 / max(1, w)))
    
    p_small = cv2.resize(prev_gray, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    c_small = cv2.resize(current_gray, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    
    diff = cv2.absdiff(p_small, c_small)
    return float(np.mean(diff))
```

#### Mathematical Formulation:
Let $I_{t}(x, y)$ and $I_{t-1}(x, y)$ be the downsampled ($160 \times H'$) grayscale intensities at frames $t$ and $t-1$:

$$\Delta I = \frac{1}{W' \cdot H'} \sum_{x=1}^{W'} \sum_{y=1}^{H'} \left| I_t(x, y) - I_{t-1}(x, y) \right|$$

If $\Delta I > \text{CAMERA\_MOTION\_THRESHOLD} = 12.0$, the scene is classified as undergoing significant camera or background motion.

### 5.3.2 Crowd Occlusion Density Score (`estimate_occlusion_density`)
To quantify crowding and physical overlap among patients in the waiting area, the analyzer calculates the **Mean Pairwise Intersection-over-Union (IoU)** across all detected bounding boxes $\mathcal{B} = \{ b_1, b_2, \dots, b_N \}$:

```python
@staticmethod
def estimate_occlusion_density(boxes: np.ndarray) -> float:
    if boxes is None or len(boxes) < 2:
        return 0.0
    n = len(boxes)
    ious = []
    for i in range(n):
        for j in range(i + 1, n):
            inter_area = max(0.0, min(b1[2], b2[2]) - max(b1[0], b2[0])) * \
                         max(0.0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
            if inter_area > 0:
                union_area = b1_area + b2_area - inter_area
                ious.append(inter_area / union_area)
    return float(np.mean(ious)) if ious else 0.0
```

#### Mathematical Formulation:
For $N$ detected bounding boxes, the pairwise occlusion density $\Omega$ is:

$$\Omega = \begin{cases} \frac{2}{N(N-1)} \sum_{i=1}^{N-1} \sum_{j=i+1}^N \text{IoU}(b_i, b_j) & \text{if } N \ge 2 \\ 0.0 & \text{if } N < 2 \end{cases}$$

Where:
$$\text{IoU}(b_i, b_j) = \frac{\text{Area}(b_i \cap b_j)}{\text{Area}(b_i \cup b_j)}$$

If $\Omega > \text{OCCLUSION\_DENSITY\_THRESHOLD} = 0.25$, the scene is flagged as experiencing severe physical crowding and mutual target occlusion.

---

## 5.4 Hysteresis Stabilization Logic

In real-world computer vision, instantaneous metric thresholds are vulnerable to boundary "chatter" (rapidly toggling back and forth between two tracker algorithms on consecutive frames). This causes internal state resets and destabilizes Kalman filters.

To prevent flapping, the `MultiTrackerEngine` enforces a **Temporal Hysteresis Buffer** ($\Delta t_{\text{switch}} = 3.0\text{ seconds}$):

$$\text{Switch Condition} = (\text{recommended} \ne \text{active}) \land (t_{\text{current}} - t_{\text{last\_switch}} \ge 3.0\text{ s})$$

```python
if self.mode == "auto":
    recommended = SceneAnalyzer.recommend_tracker(self.last_motion_score, self.last_occlusion_score)
    now = time.time()
    if recommended != self.active_tracker_name:
        if (now - self.last_switch_time) >= settings.AUTO_SWITCH_STABILIZATION_SECONDS:
            logger.info(f"Auto-switching tracker from '{self.active_tracker_name}' to '{recommended}'")
            self.active_tracker_name = recommended
            self.last_switch_time = now
```

This guarantees that tracking state transitions occur smoothly without transient thrashing.

---

## 5.5 FastTracker with Parent-Child Spatial Anchoring

In specialized pediatric intake zones, carried infants share an intimate spatial trajectory with their adult caregivers. When an infant is held against a parent's chest, classical Kalman filters frequently suffer from **ID swapping** because bounding boxes overlap by $40\% - 70\%$.

**FastTracker** incorporates a **Parent-Child Spatial Anchoring Protocol**:
1. **Geometric Containment Test:** For every detected pediatric box $B_{\text{child}} = (x_1^c, y_1^c, x_2^c, y_2^c)$, the tracker computes containment within candidate adult bounding boxes $B_{\text{adult}} = (x_1^a, y_1^a, x_2^a, y_2^a)$:
   $$\text{Containment}(B_{\text{child}}, B_{\text{adult}}) = \frac{\text{Area}(B_{\text{child}} \cap B_{\text{adult}})}{\text{Area}(B_{\text{child}})}$$
2. **Trajectory Coupling:** If $\text{Containment} \ge 0.60$, the infant track is anchored to the adult track $\text{ID}_{\text{adult}}$.
3. **Kalman Velocity Blending:** Infant centroid velocity updates $\mathbf{v}_{\text{child}}$ incorporate an Exponential Moving Average (EMA) momentum blend with the caregiver's movement:
   $$\mathbf{v}_{\text{child}}^{(t)} = (1 - \gamma) \mathbf{v}_{\text{child}}^{(t)} + \gamma \mathbf{v}_{\text{adult}}^{(t)}, \quad \gamma = 0.35$$
4. **Kalman Rollback on Re-identification:** If the infant becomes fully occluded by a blanket and re-emerges several seconds later, the trajectory rolls back to the caregiver's persistent centroid offset, preventing ID swaps.

---

## 5.6 Untracked Spatial Centroid Fallback Re-Identification

When a carried child undergoes sudden, severe occlusion (e.g., the caregiver shifts position and covers the infant's face), ByteTrack's Hungarian matching may fail to associate the detection with an existing track, returning an unassigned identifier $\text{track\_id} = -1$.

To prevent this temporary drop from instantiating a new patient track, the vision engine implements **Spatial Centroid Fallback Matching** (`Detector._resolve_track_id`):

```python
def _resolve_track_id(self, box, track_id: int, class_id: int, claimed_ids: set) -> int:
    if track_id != -1 and track_id not in claimed_ids:
        return track_id

    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0

    best_match_id = None
    min_dist = settings.UNTRACKED_SPATIAL_MATCH_RADIUS  # 40.0 pixels

    for active_id, (acx, acy, aclass_id) in self.counter.active_centroids.items():
        if aclass_id == class_id and active_id not in claimed_ids:
            dist = math.hypot(cx - acx, cy - acy)
            if dist < min_dist:
                min_dist = dist
                best_match_id = active_id

    if best_match_id is not None:
        return best_match_id

    self._synthetic_id_counter += 1
    return self._synthetic_id_counter
```

### Mathematical Logic:
Let $(c_x, c_y)$ be the centroid of the untracked detection $b$. We search the set of currently active centroids $\mathcal{A} = \{ (\text{ID}_k, a_{x, k}, a_{y, k}, c_k) \}$ for candidates sharing the same class label ($c_k = \text{class\_id}$) that have not yet been claimed in the current frame:

$$\text{Match ID} = \arg\min_{k \in \mathcal{A} \setminus \mathcal{C}_{\text{claimed}}} \sqrt{(c_x - a_{x, k})^2 + (c_y - a_{y, k})^2}$$

Subject to:
$$\min d(c, a_k) \le R_{\text{spatial\_match}} = 40.0\text{ pixels}$$

If a candidate is found within radius $R_{\text{spatial\_match}}$, the unassigned detection inherits the existing persistent $\text{ID}_k$, maintaining seamless tracking continuity.

---

## 5.7 Spatial Debouncing & Lost-Centroid Temporal Queues

A primary clinical objective of this research is generating accurate cumulative daily tallies (`total_daily_children`, `total_daily_adults`). In dynamic waiting rooms, patients frequently leave the camera view briefly (e.g., visiting a restroom or water dispenser) or undergo prolonged occlusion behind doors.

If a tracker drops an ID after its timeout period ($30\text{ seconds}$), standard systems treat the re-entering patient as a brand new individual, incrementing cumulative totals.

### 5.7.1 The Spatial Debouncing Mechanism (`Counter.process_detection`)
To prevent duplicate daily count inflation, the `Counter` maintains a **Lost Centroid Temporal Queue** ($\mathcal{Q}_{\text{lost}}$):

```
       New Track ID Detected (is_new = True)
                         │
                         ▼
       Clean Expired Centroids from Q_lost (Δt > 5.0 seconds)
                         │
                         ▼
       Search Q_lost for Spatial Match:
       dist(Centroid_new, Centroid_lost) < 150 pixels AND class_id Match
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        Match Found!          No Match Found
     (Consume Centroid)       (True New Patient)
              │                     │
              ▼                     ▼
     Suppress Increment       Increment Daily Tally
     (Do NOT double-count)    total_daily += 1
```

```python
# Spatial Debounce Check in Counter.process_detection
current_time = time.time()
self.lost_centroids = [lc for lc in self.lost_centroids if current_time - lc[2] <= self.temporal_threshold]

matched = False
for i, (lcx, lcy, ts, lcid) in enumerate(self.lost_centroids):
    if lcid == class_id:
        dist = math.hypot(cx - lcx, cy - lcy)
        if dist < self.spatial_threshold:  # 150 pixels
            matched = True
            self.lost_centroids.pop(i)  # Consume lost centroid
            break

if not matched:
    if class_id == self.adult_class_id:
        self.state.total_daily_adults += 1
    elif class_id == self.child_class_id:
        self.state.total_daily_children += 1
```

By enforcing a spatial debounce radius ($D_{\text{thresh}} = 150\text{ px}$) and a temporal threshold ($T_{\text{thresh}} = 5.0\text{ s}$), temporary tracking interruptions and re-entries at the triage boundary are prevented from inflating cumulative patient records.
