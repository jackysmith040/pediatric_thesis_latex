# The Invisible Child: Pediatric Patient Counting with Foundation Knowledge Distillation, Interactive Object Detection, and Dynamic Multi-Tracking in Clinical Triage Environments

**Academic Context:** Final Year Research Thesis & Project (2026)  
**Discipline:** Applied Mathematics, Computer Vision & Biomedical Informatics  
**Institution:** Kwame Nkrumah University of Science and Technology (KNUST), Department of Mathematics  
**Repository:** [https://github.com/jackysmith040/pediatric_research_thesis.git](https://github.com/jackysmith040/pediatric_research_thesis.git)  

---

## Executive Abstract

In resource-constrained emergency departments (EDs) and outpatient triage facilities, infant and pediatric mortality is heavily exacerbated by prolonged, unmonitored waiting times. A primary driver of clinical oversight is the **"Invisible Child" phenomenon**: infants wrapped in swaddling cloths, carried against caregivers' torsos, or seated low on benches frequently elude standard visual scanning and fixed-tally nursing logs.

This research presents a real-time, edge-deployed clinical computer vision system engineered specifically to detect, track, differentiate, and enumerate pediatric patients versus adults under severe physical occlusion and variable illumination. 

### Core Innovations & Technical Contributions:
1. **Self-Supervised Vision Foundation Knowledge Distillation (DINOv3 $\to$ YOLO26s):** A dense representation transfer framework utilizing a $1\times 1$ convolutional projection head aligning intermediate YOLO26s student feature maps with a self-supervised DINOv3 ViT teacher via joint Cosine Similarity and MSE loss ($\mathcal{L}_{\text{distill}} = \alpha \mathcal{L}_{\text{cos}} + \beta \mathcal{L}_{\text{MSE}}$). This boosts severe physical occlusion detection by **+20.2 percentage points** without adding any runtime latency on edge CPUs.
2. **Slicing Aided Hyper Inference (SAHI) for Small-Scale Pediatric Discovery:** A dynamic patch tiling pipeline ($640\times 640$ patches, 20% overlap) resolving tiny, carried infant bounding boxes in wide-angle, high-resolution hospital corridor feeds, delivering **94.6% mAP@50** and **76.9% mAP under heavy occlusion**.
3. **Decoupled Asynchronous Inference Pipeline:** An OpenCV capture-draining thread (`CAP_PROP_BUFFERSIZE = 1`) coupled to a background neural inference engine, achieving zero frame latency and constant 30 FPS playback.
4. **Dynamic Multi-Tracker Suite with Scene Analysis:** An adaptive tracking engine combining **ByteTrack**, **BoT-SORT**, **OC-SORT**, and **FastTracker with Parent-Child ID Anchoring**, governed by an optical flow and pairwise Intersection-over-Union (IoU) scene analyzer with a 3.0-second hysteresis buffer.
5. **Illumination-Invariant Contrast Normalization:** A LAB color-space Contrast Limited Adaptive Histogram Equalization (**CLAHE**) pipeline enhancing low-light triage visibility without chromatic distortion.
6. **Centroid Spatial Fallback & Temporal Debouncing:** A spatial Euclidean re-identification mechanism (`radius = 40.0 px`) and a 5.0-second lost-track debouncing queue that prevents target identity flicker and double-counting during severe occlusion.
7. **High-Efficiency ONNX Runtime Deployment:** Quantized and ONNX-optimized neural architectures achieving a **2.1x CPU inference speedup** (28.4 ms vs. 59.8 ms on baseline PyTorch), allowing deployment on standard hospital workstations without discrete GPUs.
8. **Reactive Monolithic UI & Automated Auditing:** A zero-network-serialization Python monolith built on NiceGUI, featuring real-time telemetry binding, dynamic camera and AI model switching, capacity threshold alerting, and automated PDF/CSV clinical audit generation.

---

## Thesis Chapter Directory

| Chapter | File | Topic & Scope |
|:---|:---|:---|
| **Preliminaries** | [`00_ABSTRACT_AND_PRELIMINARIES.md`](00_ABSTRACT_AND_PRELIMINARIES.md) | Title page, Dedication, Acknowledgements, Table of Contents, Nomenclature & Acronyms |
| **Chapter 1** | [`01_INTRODUCTION.md`](01_INTRODUCTION.md) | Clinical Problem Statement, The "Invisible Child", Research Questions, Objectives, Contributions |
| **Chapter 2** | [`02_LITERATURE_REVIEW.md`](02_LITERATURE_REVIEW.md) | Object Detection Evolution, Knowledge Distillation & Foundation Models, SAHI Patch Slicing, Multi-Object Tracking State-of-the-Art |
| **Chapter 3** | [`03_SYSTEM_ARCHITECTURE.md`](03_SYSTEM_ARCHITECTURE.md) | Monolithic Architecture, 2-Stage ML Pipeline (Offline Distillation to Edge), Decoupled Dual-Thread Pattern |
| **Chapter 4** | [`04_COMPUTER_VISION_AND_DEEP_LEARNING.md`](04_COMPUTER_VISION_AND_DEEP_LEARNING.md) | DINOv3 to YOLO26s Distillation Loss, Projection Head, 2-Stage Training Regimen, SAHI Architecture, ONNX Optimization |
| **Chapter 5** | [`05_TRACKING_SCENE_ANALYSIS_AND_DEBOUNCING.md`](05_TRACKING_SCENE_ANALYSIS_AND_DEBOUNCING.md) | Mathematical Formulations of Trackers, FastTracker Parent-Child Anchoring, Scene Analyzer, Hysteresis, Spatial Fallback |
| **Chapter 6** | [`06_CLINICAL_MONITORING_AND_USER_INTERFACE.md`](06_CLINICAL_MONITORING_AND_USER_INTERFACE.md) | Telemetry Computation, Overcrowding Alerts, Comparative Evaluation Lab (`/video-test`), NiceGUI Dark System, PDF/CSV Reports |
| **Chapter 7** | [`07_EXPERIMENTS_BENCHMARKS_AND_RESULTS.md`](07_EXPERIMENTS_BENCHMARKS_AND_RESULTS.md) | 6-Way Comparative Ablation Matrix ($M_1 \dots M_6$), Occlusion Tiers, Distillation Fidelity, MOTA/IDF1, Latency, LaTeX `booktabs` Table |
| **Chapter 8** | [`08_ETHICAL_PRIVACY_AND_DEPLOYMENT.md`](08_ETHICAL_PRIVACY_AND_DEPLOYMENT.md) | Privacy-by-Design, HIPAA/GDPR Compliance, Demographic Bias Mitigation, Failure Modes, Low-Resource Setup |
| **Chapter 9** | [`09_CONCLUSION_AND_FUTURE_WORK.md`](09_CONCLUSION_AND_FUTURE_WORK.md) | Research Summary, Key Clinical Findings, Limitations, Future Directions (Edge NPUs, Depth Sensing) |
| **Bibliography** | [`10_REFERENCES_AND_BIBLIOGRAPHY.md`](10_REFERENCES_AND_BIBLIOGRAPHY.md) | Comprehensive Academic Bibliography (IEEE & APA Standards) |
| **Appendices** | [`11_APPENDICES.md`](11_APPENDICES.md) | Configuration Schema, Mathematical Derivations (CIoU, Cosine Loss, 101-pt AP), 46-Test Unit Verification Suite |
| **Full Manuscript** | [`FULL_THESIS.md`](FULL_THESIS.md) | Complete Consolidated Research Thesis (Single-File Compilation) |

---

## System Architecture Summary

```
                       ┌─────────────────────────────────────────┐
                       │       Hospital Video Source Feed        │
                       │    (Webcam / RTSP / MP4 / YouTube)      │
                       └────────────────────┬────────────────────┘
                                            │
                                            ▼
                       ┌─────────────────────────────────────────┐
                       │   Thread 1: Dedicated Capture Worker    │
                       │   - OpenCV Draining Loop                │
                       │   - CAP_PROP_BUFFERSIZE = 1             │
                       │   - Real-Time 30 FPS Zero-Lag Stream    │
                       └────────────────────┬────────────────────┘
                                            │ Latest Video Frame
                                            ▼
                       ┌─────────────────────────────────────────┐
                       │   Thread 2: Asynchronous Vision Engine  │
                       │   - LAB Color Space CLAHE Preprocessing │
                       │   - SAHI Multi-Scale Patch Slicing      │
                       │   - DINOv3-Distilled YOLO26s (ONNX)     │
                       │   - Supervision Detections Parsing      │
                       │   - Adaptive Multi-Tracker Engine       │
                       │     (ByteTrack / BoT-SORT / FastTracker)│
                       │   - FastTracker Parent-Child Anchoring  │
                       │   - Scene Analyzer (Motion & Occlusion) │
                       │   - Spatial Centroid Fallback (r=40px)  │
                       │   - Temporal Debouncing Queue (t=5.0s)  │
                       └────────────────────┬────────────────────┘
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     ▼                                             ▼
       ┌───────────────────────────┐                 ┌───────────────────────────┐
       │ In-Memory Telemetry State │                 │ Pre-Rendered Frame Stream │
       │ - Current Pediatric Count │                 │ - Sci-Fi Bracket Overlays │
       │ - Current Adult Count     │                 │ - Centroid Fading Trails  │
       │ - Cumulative Daily Counts │                 │ - Track ID Labels         │
       │ - Overcrowding Alarm Flag │                 │ - Native MJPEG Endpoint   │
       └─────────────┬─────────────┘                 └─────────────┬─────────────┘
                     │ Direct In-Memory                            │ Native Stream
                     │ Reactive Binding                            │ Source
                     ▼                                             ▼
       ┌─────────────────────────────────────────────────────────────────────────┐
       │                   NiceGUI Clinical Command Center                       │
       │  - Live Video Monitoring Dashboard (`/dashboard`)                       │
       │  - Evaluation & Stream Testing Lab (`/video-test`)                      │
       │  - Runtime AI Model & Tracking Selector Dropdowns                       │
       │  - Clinical Overcrowding Banner & Metric Badges                         │
       │  - Automated Daily CSV Telemetry & Formatted PDF Capacity Reports       │
       └─────────────────────────────────────────────────────────────────────────┘
```

---

## Academic Submission & Typesetting Roadmap

### Standard Academic Thesis Format (Recommended for KNUST / University Submission):
- **Font:** `12pt Times New Roman` or `Computer Modern` (LaTeX)
- **Spacing:** `1.5` line spacing
- **Margins:** `1.5"` Left margin (for bookbinding gutter), `1.0"` Top, Bottom, and Right standard margins
- **Pagination:** Lowercase Roman numerals (`i–x`) for Preliminaries; Arabic numerals (`1–95`) for Chapters 1 through 11
- **Total Estimated Length:** **~85 to 95 pages** (including preliminaries, theoretical proofs, architecture diagrams, benchmark tables, and appendices)

---

## Reproduction & Execution Guide

### Prerequisites
- Python 3.11+ / Python 3.12 / Python 3.14
- Virtual environment (`venv` or `conda`)
- Visual C++ Redistributable (Windows) / standard build essentials

### Installation & Setup
```bash
# Clone the repository
git clone https://github.com/jackysmith040/pediatric_research_thesis.git
cd pediatric_research_thesis

# Create and activate Python virtual environment
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### Running Automated Test Suite
```bash
python -m pytest -v
```

### Launching the Clinical Monitor
```bash
python -m src.main
```
The application will launch and be accessible at:
- **Local Web Interface:** `http://127.0.0.1:8080`
- **Video Stream MJPEG Endpoint:** `http://127.0.0.1:8080/camera/stream`
- **Evaluation Testing Lab:** `http://127.0.0.1:8080/video-test`

---
*© 2026 Department of Mathematics, Kwame Nkrumah University of Science and Technology. All rights reserved.*
