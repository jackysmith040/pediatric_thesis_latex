# Chapter 11: Appendices

---

## Appendix A: System Configuration & Hyperparameter Schema

The system configuration is centralized within the `Settings` class (`src/engine/config.py`), which introspects environment variables (`.env`) with strict typing enforced via Pydantic:

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Neural Model Parameters
    MODEL_PATH: str = "models/fine_tuned/yolo26s_distilled.pt"
    CONFIDENCE_THRESHOLD: float = 0.45
    IOU_THRESHOLD: float = 0.40
    PEDIATRIC_CONF_THRESHOLD: float = 0.30
    MAX_FRAME_WIDTH: int = 1280

    # Knowledge Distillation & SAHI Parameters
    ENABLE_SAHI: bool = True
    SAHI_SLICE_HEIGHT: int = 640
    SAHI_SLICE_WIDTH: int = 640
    SAHI_OVERLAP_RATIO: float = 0.20

    # CLAHE Illumination Preprocessing
    ENABLE_CLAHE: bool = True
    CLAHE_CLIP_LIMIT: float = 2.0
    CLAHE_TILE_GRID_SIZE: int = 8

    # Roboflow Supervision & ByteTrack Parameters
    TRACKER_CONFIG: str = "bytetrack.yaml"
    ID_EXPIRY_SECONDS: int = 30
    UNTRACKED_SPATIAL_MATCH_RADIUS: float = 40.0
    BYTETRACK_TRACK_THRESH: float = 0.30
    BYTETRACK_MATCH_THRESH: float = 0.80
    BYTETRACK_FRAME_RATE: int = 30

    # Perspective Normalization & Aspect Ratio Heuristics
    ENABLE_PERSPECTIVE_CORRECTION: bool = True
    PERSPECTIVE_HORIZON_Y: float = 0.20
    PERSPECTIVE_FAR_HEIGHT_RATIO: float = 0.22
    PERSPECTIVE_NEAR_HEIGHT_RATIO: float = 0.55
    TEMPORAL_VOTING_WINDOW: int = 15

    # Multi-Tracker Engine & Adaptive Scene Switching
    DEFAULT_TRACKER_MODE: str = "auto"
    AUTO_SWITCH_STABILIZATION_SECONDS: float = 3.0
    CAMERA_MOTION_THRESHOLD: float = 12.0
    OCCLUSION_DENSITY_THRESHOLD: float = 0.25

    # Clinical Triage Overcrowding Logic
    WAITING_ROOM_CAPACITY: int = 50
    PEDIATRIC_ALERT_THRESHOLD_PERCENT: float = 30.0
    CAMERA_ID: str = "outpatient_waiting_cctv_01"
    VIDEO_SOURCE: str = "0"

    # Class Mappings (Fine-Tuned Weights: 0=Adult, 1=Child)
    ADULT_CLASS_ID: int = 0
    CHILD_CLASS_ID: int = 1
```

---

## Appendix B: Mathematical Derivations

### B.1 Derivation of Complete Intersection-over-Union (CIoU) Loss
Let $B = (x, y, w, h)$ be the predicted bounding box and $B^{\text{gt}} = (x^{\text{gt}}, y^{\text{gt}}, w^{\text{gt}}, h^{\text{gt}})$ be the ground-truth box.

The intersection $\mathcal{I}$ and union $\mathcal{U}$ are:
$$\mathcal{I} = \max(0, \min(x_2, x_2^{\text{gt}}) - \max(x_1, x_1^{\text{gt}})) \times \max(0, \min(y_2, y_2^{\text{gt}}) - \max(y_1, y_1^{\text{gt}}))$$
$$\mathcal{U} = \text{Area}(B) + \text{Area}(B^{\text{gt}}) - \mathcal{I}$$
$$\text{IoU} = \frac{\mathcal{I}}{\mathcal{U}}$$

The centroid Euclidean distance penalty is:
$$\mathcal{R}_{\text{distance}} = \frac{\rho^2(b, b^{\text{gt}})}{c^2} = \frac{(x - x^{\text{gt}})^2 + (y - y^{\text{gt}})^2}{c_w^2 + c_h^2}$$

Where $c$ is the diagonal length of the smallest enclosing box covering both $B$ and $B^{\text{gt}}$.

The aspect ratio consistency parameter $v$ and balancing weight $\alpha$ are:
$$v = \frac{4}{\pi^2} \left( \arctan\frac{w^{\text{gt}}}{h^{\text{gt}}} - \arctan\frac{w}{h} \right)^2$$
$$\alpha = \frac{v}{(1 - \text{IoU}) + v}$$

The complete loss minimized during backpropagation is:
$$\mathcal{L}_{\text{CIoU}} = 1 - \text{IoU} + \frac{\rho^2(b, b^{\text{gt}})}{c^2} + \alpha v$$

### B.2 Derivation of Dense Representation Cosine Feature Loss
Let $\mathbf{u} = F_T(i, j) \in \mathbb{R}^D$ be the teacher's latent representation at patch $(i, j)$ and $\mathbf{v} = \mathcal{P}(F_S)(i, j) \in \mathbb{R}^D$ be the student's projected feature vector.

The cosine similarity is:
$$\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2 + \epsilon} = \frac{\sum_{d=1}^D u_d v_d}{\sqrt{\sum_{d=1}^D u_d^2} \sqrt{\sum_{d=1}^D v_d^2} + \epsilon}$$

The cosine loss minimized over all spatial locations $H \times W$ is:
$$\mathcal{L}_{\text{cos}} = 1 - \frac{1}{HW} \sum_{i=1}^H \sum_{j=1}^W \cos(F_T(i, j), \mathcal{P}(F_S)(i, j))$$

The gradient with respect to student projection activation $\mathbf{v}$ is:
$$\frac{\partial \mathcal{L}_{\text{cos}}}{\partial \mathbf{v}} = -\frac{1}{HW} \left[ \frac{\mathbf{u}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} - \frac{(\mathbf{u} \cdot \mathbf{v}) \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2^3} \right]$$

This gradient directly pushes the student feature vector to align with the teacher's semantic orientation in latent space.

### B.3 Derivation of COCO 101-Point Interpolated Average Precision
For a given recall threshold $r \in [0.0, 1.0]$, the interpolated precision $P_{\text{interp}}(r)$ is defined as the maximum precision found for any recall $\tilde{r} \ge r$:

$$P_{\text{interp}}(r) = \max_{\tilde{r} \ge r} P(\tilde{r})$$

The 101-point interpolated Average Precision at IoU threshold $\tau$ is:
$$\text{AP}_{\tau} = \frac{1}{101} \sum_{k=0}^{100} P_{\text{interp}}\left(\frac{k}{100}\right)$$

The primary COCO benchmark metric is obtained by averaging across 10 IoU thresholds from $0.50$ to $0.95$:
$$\text{mAP@[50:95]} = \frac{1}{10} \sum_{m=0}^9 \text{AP}_{\tau = 0.50 + 0.05m}$$

---

## Appendix C: Automated Test Suite Architecture & Verification Protocols

The system includes an exhaustive automated test suite (`tests/unit/`) executed via Pytest:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     UNIT TEST SUITE ARCHITECTURE                        │
│                                                                         │
│  Test Module                  Tests Covered    Status                   │
│  ─────────────────────────────────────────────────────────────────────  │
│  test_counter.py              7 Test Cases     PASS (100%)              │
│  test_detector.py             17 Test Cases    PASS (100%)              │
│  test_evaluation_metrics.py   12 Test Cases    PASS (100%)              │
│  test_stream_resolver.py      5 Test Cases     PASS (100%)              │
│  test_tracker_engine.py       5 Test Cases     PASS (100%)              │
│  ─────────────────────────────────────────────────────────────────────  │
│  TOTAL SUITE VERIFICATION:    46 / 46 PASSED   PASS (Zero Errors)       │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Test Cases Verified:
- `test_compute_distillation_fidelity_identical`: Asserts cosine similarity $= 1.0$ and normalized MSE $= 0.0$ on aligned features.
- `test_compute_ap_coco_perfect`: Asserts exact $1.00$ average precision under 101-point COCO interpolation.
- `test_run_4way_ablation_matrix_and_exports`: Asserts that the 6-way ablation matrix correctly computes mAP@50, mAP@[50:95], and generates valid LaTeX `booktabs` code.
- `test_profile_model_hardware_cpu`: Validates multi-threaded CPU hardware profiling, median p50 latency, and throughput (FPS).
- `test_counter_spatial_debounce`: Asserts that re-appearing lost tracks within $150\text{ px}$ radius do not increment cumulative patient totals.
- `test_detector_clahe_preprocessing`: Asserts that CLAHE preserves BGR array dimensions and improves luminance contrast.
- `test_tracker_engine_auto_switching`: Verifies that camera motion ($\Delta I > 12.0$) and crowd occlusion ($\Omega > 0.25$) trigger tracker auto-switching subject to $3.0\text{-second}$ hysteresis barriers.
