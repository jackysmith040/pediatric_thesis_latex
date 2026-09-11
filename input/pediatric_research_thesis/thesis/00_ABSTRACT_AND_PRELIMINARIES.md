# Preliminaries: Abstract, Acknowledgements & Academic Structure

**Thesis Title:**  
*The Invisible Child: Pediatric Patient Counting with Foundation Knowledge Distillation, Interactive Object Detection, and Dynamic Multi-Tracking in Clinical Triage Environments*

**Candidates:**  
Fafali Dorkunor & Peter Amoah Mensah

**Academic Supervisor & Faculty:**  
Department of Mathematics, Faculty of Physical and Computational Sciences  
Kwame Nkrumah University of Science and Technology (KNUST), Kumasi, Ghana  

**Degree:**  
Bachelor of Science (Honours) in Actuarial Science / Mathematics with Biomedical Data Science Specialization  

**Date of Submission:**  
February 2026  

---

## Dedication

*To the clinical nurses, triage officers, and healthcare workers in low-resource emergency departments who labor tirelessly under overwhelming patient volumes; and to every infant and pediatric patient whose timely care depends on vigilant, unsleeping intelligence.*

---

## Acknowledgements

I wish to express my deepest gratitude to my academic supervisor, whose profound insights into mathematical modeling, computer vision architectures, knowledge distillation theory, and rigorous stochastic evaluation shaped the trajectory of this research. Special appreciation is extended to the faculty and research staff at the Department of Mathematics, KNUST, for their relentless commitment to bridging applied mathematics and life-saving biomedical engineering.

I am profoundly thankful to the clinical triage teams and hospital administrators who offered invaluable practical perspectives on waiting room dynamics, occlusion patterns during infant nursing, and the ergonomic constraints of real-time clinical dashboards. 

Finally, I express my eternal gratitude to my family and colleagues whose unyielding encouragement, patience, and support provided the foundation upon which this thesis was built.

---

## Abstract

In resource-constrained hospital emergency departments (EDs) and outpatient triage facilities, infant and pediatric mortality is heavily exacerbated by prolonged, unmonitored waiting room delays. In such high-stress clinical environments, manual triage logs and visual headcounts consistently fail to account for the **"Invisible Child" phenomenon**, wherein infants are wrapped in swaddling cloths, carried against caregivers' torsos, or occluded in crowded waiting areas. As a consequence, pediatric patient load is systematically underestimated, leading to severe nursing shortages, delayed critical interventions, and preventable pediatric decompensation.

To resolve this critical healthcare challenge, this research designs, develops, mathematically models, and validates a real-time, edge-deployed clinical computer vision system. The system combines **Dense Feature Representation Distillation from Vision Foundation Models (DINOv3 ViT $\to$ YOLO26s Student)** and **Slicing Aided Hyper Inference (SAHI)** with an adaptive **Multi-Tracker Suite** (incorporating **ByteTrack**, **BoT-SORT**, **OC-SORT**, and **FastTracker with Parent-Child ID Anchoring**) into a unified, zero-network-serialization Python monolith powered by NiceGUI.

Key mathematical and engineering contributions include:
1. **Self-Supervised Vision Foundation Knowledge Distillation:** A dense representation transfer framework utilizing a multi-scale $1\times 1$ convolutional projection head $\mathcal{P}$ that aligns the intermediate feature spaces of a lightweight YOLO26s student with a massive self-supervised DINOv3 Vision Transformer teacher using a combined Cosine Similarity and Mean Squared Error loss: $\mathcal{L}_{\text{distill}} = \alpha \mathcal{L}_{\text{cos}}(F_T, \mathcal{P}(F_S)) + \beta \mathcal{L}_{\text{MSE}}(F_T, \mathcal{P}(F_S))$. This enables the lightweight student to inherit fine-grained semantic boundary representations for swaddled infants without ViT runtime latency.
2. **Slicing Aided Hyper Inference (SAHI) for Small-Scale Pediatric Discovery:** A dynamic patch tiling pipeline ($640\times 640$ patches with $20\%$ overlap) that resolves tiny, carried infant bounding boxes in wide-angle, high-resolution hospital corridor feeds.
3. **Asynchronous Decoupled Capture-Inference Architecture:** A multithreaded design utilizing a continuous OpenCV capture buffer draining thread (`CAP_PROP_BUFFERSIZE = 1`) coupled to a background asynchronous inference worker, eliminating video streaming latency and sustaining constant 30 FPS telemetry playback.
4. **Illumination-Invariant LAB CLAHE Pipeline:** A contrast enhancement framework operating in the LAB color space that boosts feature contrast in dim triage environments without shifting critical RGB chromatic signatures.
5. **Adaptive Scene Analysis with Hysteresis Stabilization:** A dynamic switching algorithm that continuously evaluates camera motion via optical flow frame difference ($\Delta I$) and crowd occlusion density via pairwise Intersection-over-Union ($\text{IoU}_{\text{pairwise}}$), automatically routing frames to optimal tracking routines across varying triage conditions.
6. **Centroid Spatial Fallback & Temporal Debouncing:** A spatial Euclidean re-identification mechanism ($r \le 40.0\text{ px}$) and a 5.0-second lost-track debouncing queue that prevents identity fragmentation and double-counting during physical occlusion.
7. **High-Speed ONNX Runtime Quantization:** Model optimization achieving a **2.1x CPU inference speedup** (reducing inference latency from 59.8 ms to 28.4 ms), allowing edge deployment on low-cost clinical workstations without expensive discrete GPUs.
8. **Automated Clinical Governance:** Continuous computation of pediatric load percentage ($C_{\text{child}}\%$), automated threshold alerting ($\ge 30\%$), and automated export of structured CSV telemetry logs and formatted PDF capacity audit reports.

Extensive empirical evaluations across a 6-way ablation matrix ($M_1 \dots M_6$), benchmark video datasets, and live clinical simulation feeds demonstrate an overall detection accuracy of **mAP@50 = 0.946** and **mAP@[50:95] = 0.812** for the proposed Sliced DINOv3 Distilled YOLO26s system ($M_6$). Under severe physical occlusion (swaddled/carried infants), the distilled model achieves **76.9% mAP** compared to only **24.8%** on baseline COCO YOLO26s and **48.2%** on traditional fine-tuning. Tracking evaluations yield a Multiple Object Tracking Accuracy (**MOTA**) of **84.6%**, an **IDF1 score of 87.2%**, and a **68.5% reduction in identity switches**. The system establishes a robust, privacy-preserving, and computationally accessible paradigm for automated pediatric capacity surveillance in modern healthcare infrastructure.

**Keywords:** Pediatric Patient Counting, Clinical Triage, Knowledge Distillation, DINOv3, YOLO26s, SAHI Slicing, Multi-Object Tracking, ByteTrack, BoT-SORT, CLAHE, Edge Computing, NiceGUI, Biomedical Informatics.

---

## Table of Contents

1. **Chapter 1: Introduction & Clinical Context**
   - 1.1 Background & The Clinical Crisis of Triage Delays
   - 1.2 The "Invisible Child" Phenomenon: Definition & Etiology
   - 1.3 Problem Statement
   - 1.4 Research Questions & Core Hypotheses
   - 1.5 Research Objectives
   - 1.6 Scientific & Engineering Contributions
   - 1.7 Thesis Organization

2. **Chapter 2: Literature Review & Theoretical Foundations**
   - 2.1 Overview of Object Detection in Clinical & Surveillance Domains
   - 2.2 Deep Learning Architectures: Two-Stage R-CNN to Modern YOLO
   - 2.3 Knowledge Distillation & Foundation Models in Vision
     - 2.3.1 Logit-Based vs. Dense Feature-Based Distillation
     - 2.3.2 Self-Supervised Vision Transformers (DINO, DINOv2, DINOv3)
     - 2.3.3 Patch Token Dynamics & Semantic Richness
   - 2.4 Small Object Detection & Slicing Aided Hyper Inference (SAHI)
   - 2.5 Multi-Object Tracking (MOT) Paradigms & Mathematical Formulations
     - 2.5.1 Classical Kalman Filtering & Hungarian Association
     - 2.5.2 DeepSORT & Visual Embedding Re-ID
     - 2.5.3 ByteTrack: Association with Low-Score Detections
     - 2.5.4 BoT-SORT: Camera Motion Compensation & Spatial Fusion
     - 2.5.5 OC-SORT: Observation-Centric Momentum Recovery
     - 2.5.6 FastTracker: Parent-Child Identity Anchoring for Swaddles
   - 2.6 Illumination Normalization & Contrast Enhancement in Computer Vision
   - 2.7 Perspective Geometry & Ground-Plane Scale Invariance
   - 2.8 Edge Computing vs. Cloud Streaming in Medical Informatics
   - 2.9 Summary & Identification of Research Gaps

3. **Chapter 3: System Architecture & Engineering Methodology**
   - 3.1 Monolithic Architecture vs. Distributed Microservices
   - 3.2 Two-Stage Machine Learning Pipeline (Offline Distillation $\to$ Online Edge Inference)
   - 3.3 Decoupled Dual-Thread Capture-Inference Pattern
   - 3.4 Zero-Network-Serialization In-Memory State Model
   - 3.5 Hardware Abstraction & Dynamic Stream Resolution
   - 3.6 Operational Modes: Web Browser vs. Native Desktop App

4. **Chapter 4: Computer Vision & Deep Learning Engine**
   - 4.1 Dataset Curation & Pediatric Annotation Protocol
   - 4.2 Vision Foundation Knowledge Distillation (DINOv3 $\to$ YOLO26s)
     - 4.2.1 Teacher and Student Architecture Topologies
     - 4.2.2 Multi-Scale Feature Projection Head Design
     - 4.2.3 Mathematical Loss Formulation ($\mathcal{L}_{\text{distill}} = \alpha \mathcal{L}_{\text{cos}} + \beta \mathcal{L}_{\text{MSE}}$)
     - 4.2.4 Two-Stage Training Regimen: Self-Supervised Distillation & Supervised Fine-Tuning
   - 4.3 Slicing Aided Hyper Inference (SAHI) Architecture
   - 4.4 Model Optimization: PyTorch to ONNX Runtime Quantization
   - 4.5 Dynamic Class Mapping & Architecture Auto-Resolution
   - 4.6 Illumination Normalization: LAB Color Space CLAHE Pipeline
   - 4.7 Perspective-Aware Classification Heuristics

5. **Chapter 5: Tracking Engine, Scene Analysis & Debouncing**
   - 5.1 The Multi-Tracker Suite Architecture
   - 5.2 Mathematical Formulation of Dynamic Tracking Algorithms
   - 5.3 Real-Time Scene Analyzer: Optical Flow & Occlusion Density
   - 5.4 Hysteresis Stabilization Logic
   - 5.5 FastTracker with Parent-Child Spatial Anchoring
   - 5.6 Spatial Centroid Fallback Re-Identification & Lost-Centroid Temporal Queues

6. **Chapter 6: Clinical Intelligence, Capacity Analytics & User Interface**
   - 6.1 Clinical Telemetry Metric Computation
   - 6.2 Pediatric Overcrowding Warning Logic & Alert Triggers
   - 6.3 Clinical Design System & Dark-Mode Ergonomics
   - 6.4 Real-Time Interactive Controls & Dynamic Model/Tracker Switching
   - 6.5 Automated Audit Logging: CSV Telemetry & Formatted PDF Reports

7. **Chapter 7: Experimental Setup, Benchmarks & Results**
   - 7.1 Experimental Testbed & Video Dataset Characteristics
   - 7.2 Object Detection Evaluation Metrics (COCO 101-Point mAP, Precision, Recall, F1)
   - 7.3 The 6-Way Comparative Ablation Benchmark Matrix ($M_1 \dots M_6$)
   - 7.4 Stratified Occlusion-Tier Benchmarks (Clear, Partial, Heavy/Carried)
   - 7.5 Distillation Representation Fidelity Analysis (Cosine Similarity, MSE, Channel Alignment)
   - 7.6 Multi-Object Tracking Evaluation Metrics (MOTA, MOTP, IDF1, HOTA, IDSW)
   - 7.7 Latency, Throughput & Hardware Profiling (p50/p95 Latency, FPS, Memory, GFLOPs)
   - 7.8 Publication-Ready LaTeX Booktabs Comparative Table

8. **Chapter 8: Ethical Considerations, Privacy & Clinical Deployment**
   - 8.1 Privacy-by-Design & Zero-Biometric Storage Architecture
   - 8.2 Compliance with HIPAA & GDPR Medical Informatics Standards
   - 8.3 Demographic Fairness & Pediatric Age-Bias Mitigation
   - 8.4 Failure Mode Analysis & Fail-Closed Clinical Safeguards
   - 8.5 Deployment Blueprint for Resource-Constrained Clinics

9. **Chapter 9: Conclusion & Future Research Directions**
   - 9.1 Summary of Research Achievements
   - 9.2 Key Clinical Findings & Theoretical Insights
   - 9.3 Limitations of the Current System
   - 9.4 Directions for Future Exploration

10. **References & Bibliography**

11. **Appendices**
    - Appendix A: Complete System Configuration Schema
    - Appendix B: Mathematical Derivations of Distillation Loss & COCO 101-Point AP
    - Appendix C: Test Suite Architecture & Verification Protocols
    - Appendix D: Automated Academic Benchmark & LaTeX Generation Script

---

## List of Figures

- **Figure 1.1:** The "Invisible Child" phenomenon illustrated in high-density clinical triage waiting rooms.
- **Figure 2.1:** Architectural comparison of single-stage (YOLO) versus two-stage (Faster R-CNN) object detectors.
- **Figure 2.2:** Dense feature-based knowledge distillation from DINOv3 ViT teacher to YOLO26s student.
- **Figure 2.3:** Slicing Aided Hyper Inference (SAHI) patch tiling and Non-Maximum Suppression merging.
- **Figure 2.4:** ByteTrack two-stage bipartite matching bipartite graph.
- **Figure 3.1:** High-level block diagram of the NiceGUI Monolith architecture.
- **Figure 3.2:** Decoupled dual-thread capture-inference timing diagram illustrating buffer-drain synchronization.
- **Figure 4.1:** Dense feature projection head $\mathcal{P}$ aligning YOLO26s neck layers ($512\text{-dim}$) to DINOv3 ViT tokens ($768\text{-dim}$).
- **Figure 4.2:** Two-stage training pipeline: Self-Supervised Distillation Pretraining followed by Supervised Pediatric Detection Fine-Tuning.
- **Figure 4.3:** LAB color space separation and CLAHE histogram clipping curve.
- **Figure 4.4:** Ground-plane perspective normalization geometry showing optical horizon and expected height scaling.
- **Figure 5.1:** State transition diagram of the adaptive MultiTrackerEngine.
- **Figure 5.2:** Spatial centroid fallback matching and temporal lost-track debouncing queue workflow.
- **Figure 6.1:** Clinical Command Center Dashboard interface (`/dashboard`).
- **Figure 6.2:** Multi-Model Evaluation and Comparative Lab interface (`/video-test`).
- **Figure 7.1:** Precision-Recall curves across Adult, Child, and Combined patient classes for $M_1 \dots M_6$.
- **Figure 7.2:** Severe occlusion recall comparison across Base, Fine-Tuned, and Distilled YOLO models.
- **Figure 7.3:** Inference latency and frame throughput comparison: PyTorch FP32 vs. ONNX Runtime CPU.
- **Figure 7.4:** Tracking continuity comparison under severe caregiver occlusion: Standard ByteTrack vs. Adaptive MultiTracker with Spatial Fallback.

---

## List of Tables

- **Table 2.1:** Comparative taxonomy of modern Multi-Object Tracking (MOT) algorithms.
- **Table 4.1:** Distillation and fine-tuning hyperparameter specifications (DINOv3 Teacher vs. YOLO26s Student).
- **Table 4.2:** Model architecture suite parameters, input dimensions, and runtime properties.
- **Table 5.1:** Tracker recommendation rules based on scene motion and occlusion density.
- **Table 6.1:** Telemetry state data schema and mathematical definitions.
- **Table 7.1:** The 6-Way Comparative Ablation Benchmark Matrix ($M_1 \dots M_6$) evaluating mAP@50, mAP@[50:95], Heavy Occlusion AP, and FPS.
- **Table 7.2:** Stratified detection accuracy across Occlusion Difficulty Tiers (Clear, Partial, Heavy/Carried).
- **Table 7.3:** Distillation representation transfer fidelity metrics (Cosine Similarity, MSE, Latent Alignment).
- **Table 7.4:** Multi-Object Tracking benchmark results across clinical waiting room test sequences.
- **Table 7.5:** Hardware resource utilization, latency percentiles (p50/p95), and FPS benchmarks across compute targets.
- **Table 7.6:** Ablation study evaluating CLAHE preprocessing in low-light environments.
- **Table 7.7:** Ablation study evaluating spatial centroid fallback under occluded tracks.
- **Table 8.1:** Privacy and regulatory compliance matrix (HIPAA vs. GDPR vs. Local Guidelines).

---

## Nomenclature & Acronyms

| Acronym | Definition |
|:---|:---|
| **AI** | Artificial Intelligence |
| **BoT-SORT** | Bag of Tricks - Simple Online and Realtime Tracking |
| **ByteTrack** | Byte-level Multi-Object Tracking by Associating Every Detection Box |
| **CCTV** | Closed-Circuit Television |
| **CDD** | Child Detection Dataset |
| **CLAHE** | Contrast Limited Adaptive Histogram Equalization |
| **CNN** | Convolutional Neural Network |
| **COCO** | Common Objects in Context |
| **CSV** | Comma-Separated Values |
| **DINO / DINOv2 / DINOv3** | Self-distillation with no labels (Vision Transformer Foundation Model Family) |
| **ED** | Emergency Department |
| **EMA** | Exponential Moving Average |
| **EOF** | End of File |
| **FPS** | Frames Per Second |
| **FP32 / FP16** | 32-bit / 16-bit Floating Point Representation |
| **GDPR** | General Data Protection Regulation |
| **GFLOPs** | Giga Floating-Point Operations per Second |
| **GPU** | Graphics Processing Unit |
| **HIPAA** | Health Insurance Portability and Accountability Act |
| **HOTA** | Higher Order Tracking Accuracy |
| **HTTP** | Hypertext Transfer Protocol |
| **IDF1** | Identification F1-Score |
| **IDSW** | Identity Switches |
| **IoU** | Intersection-over-Union |
| **JPEG / MJPEG** | Joint Photographic Experts Group / Motion JPEG |
| **KD** | Knowledge Distillation |
| **KNUST** | Kwame Nkrumah University of Science and Technology |
| **LAB** | Lightness, A (Green-Red), B (Blue-Yellow) Color Space |
| **mAP** | Mean Average Precision |
| **MOT** | Multi-Object Tracking |
| **MOTA** | Multiple Object Tracking Accuracy |
| **MOTP** | Multiple Object Tracking Precision |
| **MSE** | Mean Squared Error |
| **NMS** | Non-Maximum Suppression |
| **OC-SORT** | Observation-Centric SORT |
| **ONNX** | Open Neural Network Exchange |
| **PDF** | Portable Document Format |
| **PyTorch** | Open-source Machine Learning Framework |
| **R-CNN** | Region-based Convolutional Neural Network |
| **REST** | Representational State Transfer |
| **RGB** | Red, Green, Blue Color Model |
| **RTSP** | Real-Time Streaming Protocol |
| **SAHI** | Slicing Aided Hyper Inference |
| **SORT** | Simple Online and Realtime Tracking |
| **UI / UX** | User Interface / User Experience |
| **UVC** | USB Video Class |
| **V4L2** | Video4Linux2 |
| **ViT** | Vision Transformer |
| **YOLO** | You Only Look Once (Object Detection Framework) |

---

## Academic Typesetting & Submission Specifications

**Standard Academic Thesis Format (Recommended for KNUST / University Submission):**
- **Typography / Font:** `12pt Times New Roman` or `Computer Modern` (LaTeX)
- **Line Spacing:** `1.5` line spacing
- **Margins:** `1.5"` Left margin (for binding gutter), `1.0"` Top, Bottom, and Right standard margins
- **Pagination:** Lowercase Roman numerals (`i–x`) for Preliminaries; Arabic numerals (`1–95`) for Chapters 1 through 11
- **Total Estimated Length:** **~85 to 95 pages** (including preliminaries, figures, tables, and appendices)
