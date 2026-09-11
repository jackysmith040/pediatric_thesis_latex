# Conscience OS Kanban Board — Supervisor Review Remediation Backlog

Use this single board to track execution state across the 47-point supervisor review remediation.
Status markers: `[ ]` To Do, `[/]` In Progress, `[x]` Done.

---

## 🎯 Phase 1: Title, Scope, Terminology & Research Questions (Points 1–5, 16–19, 35–36, 40–42)
- [x] **1.1 Update Thesis Title**: Change title to *"Pediatric Person Detection and Counting Using Computer Vision and Multi-Object Tracking in Clinical Environments"* across all frontmatter, headers, cover, and `thesis_main.tex`. Remove ungrounded terms "Interactive Computer Vision" and "Dynamic Multi-Tracking". *(Point 1)*
- [x] **1.2 Standardize Architecture Naming**: Formally designate the system as a **multi-stage pipeline** (Stage 1 Detection $\to$ Stage 2 Crop Classification $\to$ Stage 3 Allometric/Geometric Verification $\to$ Stage 4 Multi-Object Tracking & Smoothing, + optional Re-ID). Eliminate "dual-stage" confusion. *(Point 2)*
- [x] **1.3 Conceptual Shift: "Patient" $\to$ "Person"**: Clarify across Chapters 1–5 that computer vision detects *pediatric vs adult persons*, not clinical status. Clarify that patient identification requires registration linkage or queue-zone rules. *(Points 3, 4)*
- [x] **1.4 Staff & Clinic Filtering**: Explain how staff (e.g., Table 4.3 "Adult Counter Staff") are handled/filtered or categorized as non-patients prior to LHIMS dispatch. *(Point 5)*
- [x] **1.5 Define Pediatric Cohort**: Define pediatric age threshold (e.g. under-five vs child $\le 12$) in Chapter 1 & 3, and state how ground-truth age labels were established. *(Point 17)*
- [x] **1.6 Refactor Temporal Filtering (Bayesian $\to$ Temporal Smoothing)**: Rename "recursive Bayesian updating" to *temporal consensus filtering / weighted temporal smoothing* unless formal Bayes prior/likelihood is derived. Update equations and Research Question 4 accordingly. *(Points 18, 19)*
- [x] **1.7 Disentangle Ratios in RQ3**: Disentangle head-to-body ratio ($R_c \approx 0.25, R_a \approx 0.125$) from cephalocaudal torso-to-leg ratio ($R_{\text{ceph}}$ with threshold $\tau = 0.80$). *(Point 16)*
- [x] **1.8 Refactor RQs & Research Objectives**: Rewrite non-empirical/leading questions (e.g. separable convolution arithmetic) into measurable empirical questions evaluating occlusion, classification, tracking, and edge throughput. Refocus objectives accordingly. *(Points 41, 42)*
- [x] **1.9 Scope & Claim Calibration**: Soften LHIMS integration to "designed to interface via REST APIs", qualify decision-support claims ("could support staffing decisions"), and moderate research gap assertions to "Among the studies reviewed in this thesis...". *(Points 35, 36, 40, 46)*

---

## 🎯 Phase 2: Methodology & Algorithmic Rigor (Points 6–7, 20–22, 26–28, 37–39)
- [ ] **2.1 Add Data Collection & Annotation Section**: Add dedicated Section in Chapter 3 detailing clinic image collection: frame counts, sources, periods, camera hardware/settings, adult/child counts, annotation tools (e.g., CVAT/LabelImg), protocol, and train/val/test splits. *(Point 6)*
- [ ] **2.2 Document Pediatric-Model-ONNX Classifier**: Detail the secondary crop classifier architecture (backbone, input shape $224\times 224$), training dataset, child/adult sample sizes, train/val/test split, training hyperparameters, and standalone pre-pipeline validation accuracy. *(Point 7)*
- [ ] **2.3 Define $\theta_{\text{child}}$ in Eq 3.23**: Fully specify both $\theta_{\text{adult}} = 0.32$ and $\theta_{\text{child}}$ in the normalized bounding box height rule. *(Point 20)*
- [ ] **2.4 Threshold Justifications**: Provide calibration data, sensitivity analyses, or literature citations for all empirical thresholds: $\theta_{\text{adult}}=0.32$, child classification $0.60$, 5 consecutive frames, 70/30 temporal weights, 15-frame window, 0.80 allometric $\tau$, 0.15 EMA, 0.78 Re-ID, and seated deduplication. *(Point 21)*
- [ ] **2.5 Perspective & Projection Calibration**: Qualify 2D projection assumptions by explicitly discussing camera tilt, mounting height, and viewing angle constraints. *(Point 22)*
- [ ] **2.6 Contextualize Mathematical Background**: Separate fundamental theory (image quantization, Sobel kernels, separable convolutions) from actual implemented pipeline stages; qualify the 33.3% arithmetic reduction and decouple it from whole-pipeline real-time throughput. *(Points 26, 27, 28)*
- [ ] **2.7 Dedicated Ethics & Data Governance Section**: Add dedicated Section in Chapter 3 covering ethical approval/waiver, hospital administrative permissions, consent protocols, patient privacy risks beyond facial recognition, research image storage/security, and the zero-retention RAM deployment model. *(Points 37, 38, 39)*

---

## 🎯 Phase 3: Results, Benchmarks & Consistency (Points 8–15, 23–25, 29–34)
- [ ] **3.1 Document YOLO26s Training/Fine-Tuning**: Present detector training parameters, sample sizes, epochs, data splits, and mAP evaluation in Chapter 4 rather than isolated abstract mentions. *(Point 8)*
- [ ] **3.2 Reconcile Test Cohort Counts (45 vs 46)**: Establish verified ground-truth individual count across abstract, Chapter 4, and Chapter 5 (eliminating the 45 vs 46 contradiction). *(Point 9)*
- [ ] **3.3 Fix Table 4.2 (Full 7 Sequences)**: Expand Table 4.2 to display all 7 standardized test sequences with individual sequence metrics summing exactly to the total evaluated cohort. *(Point 10)*
- [ ] **3.4 Qualify 100% Metrics & Small Cohort Caveats**: Add immediate qualifying notes in Table 4.2, results text, and abstract explaining that 100% metrics reflect curated test sequences, explicitly noting baseline detector misses, occlusion limits, and false blanket detections. *(Points 11, 12)*
- [ ] **3.5 Expand Allometric Evidence & Reconcile Ratios**: Document pediatric range $[0.941, 0.982]$ with empirical observations; reconcile separation margin consistency ($\Delta R \ge 0.204$ vs $\Delta R \ge 0.245$) across Chapters 3, 4, and 5. *(Points 13, 14, 15)*
- [ ] **3.6 Component Ablation Study**: Add structured ablation analysis (Baseline $\to$ +CLAHE $\to$ +SAHI $\to$ +Crop Classifier $\to$ +Allometry $\to$ +Temporal Smoothing) and qualify/evidence the CLAHE 34.2% contrast claim. *(Points 23, 24)*
- [ ] **3.7 Qualify Knowledge Distillation Claims**: Reframe distillation underperformance as a plausible annotation noise hypothesis rather than an isolated proven cause. *(Point 25)*
- [ ] **3.8 Clarify Real-Time Definition & Table 4.5 Latencies**: Define real-time processing requirements (frame sampling/skipping vs 25–30 FPS CCTV stream); reconcile ONNX latency (110.2 ms) vs batch elapsed times, define standalone vs full-pipeline FPS, and clarify the $3.98\times$ speedup derivation. *(Points 29, 30, 31)*
- [ ] **3.9 Reconcile Hardware Contradictions**: Reconcile GPU teacher benchmark in Table 4.1 with CPU-only testbed statements in the appendix. *(Point 32)*
- [ ] **3.10 Differentiate Clinical vs Non-Clinical Environments**: Clearly separate hospital triage test feeds from kindergarten playroom / indoor activity robustness tests in Table 4.4 and discussion. *(Point 33)*
- [ ] **3.11 Quantitative Tracking Metrics**: Report standard tracking measures (ID switches, track fragmentation, or clearly bounded continuity in evaluated sequences) to substantiate identity persistence claims. *(Point 34)*

---

## 🎯 Phase 4: Citations, Bibliography & Abstract Rewrite (Points 43–45, 47)
- [ ] **4.1 Fix Raw Citation Keys**: Replace raw bracketed keys (`[zheng2020distance]`, `[aykyal2022sahi]`) in Chapter 3 with proper numbered bibtex citations. *(Point 43)*
- [ ] **4.2 Audit Bibliography Relevance**: Review every citation to verify that cited papers directly support the text claims (Ghanaian carrying practices, clinical workflows, triage bottlenecks). *(Point 44)*
- [ ] **4.3 Modernize Architecture Citations**: Cite official documentation/papers for YOLO26/ultralytics, ONNX Runtime, OpenVINO, and pose estimation. *(Point 45)*
- [ ] **4.4 Abstract Overhaul**: Rewrite the abstract incorporating all changes: revised title, multi-stage architecture, person vs patient clarification, reconciled 45/46 count, qualified accuracy figures, and grounded real-time CPU throughput. *(Point 47)*
- [ ] **4.5 Tectonic Compilation & PyMuPDF Margin Audit**: Recompile full thesis via Tectonic and verify zero margin overflows across all pages.

---

## 🚧 In Progress
- [ ] Phase 2: Methodology & Algorithmic Rigor (Data Collection, ONNX Classifier, Threshold Calibration, Ethics Section)

---

## ✅ Done
- [x] Initialized comprehensive 47-point supervisor remediation backlog in `memory/KANBAN.md`
- [x] Phase 1: Title, Scope, Terminology & Research Questions (Points 1–5, 16–19, 35–36, 40–42) completely executed and verified on branch `phase-1-remediation`
