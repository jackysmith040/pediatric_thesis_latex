# 🏥 Pediatric Patient Detection and Counting with Interactive Computer Vision and Dynamic Multi-Tracking in Clinical Triage Environments

[![Thesis Status](https://img.shields.io/badge/Thesis%20Status-Submission%20Ready-brightgreen.svg)](#)
[![Pages](https://img.shields.io/badge/Pagination-70%20Pages%20(KNUST)-blue.svg)](#)
[![Compiler](https://img.shields.io/badge/Compiler-Tectonic%20v0.15.0-orange.svg)](#)
[![Language](https://img.shields.io/badge/Writing-ASD--STE100%20Verified-blueviolet.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Undergraduate Thesis submitted to the Department of Mathematics, Kwame Nkrumah University of Science and Technology (KNUST), Kumasi, Ghana, in partial fulfilment of the requirements for the degree of Bachelor of Science in Mathematics.**

---

## 👥 Authors & Supervision

- **Candidate Authors**:
  - **Fafali Dorkunor** (Index Number: `20912614`)
  - **Peter Amoah Mensah** (Index Number: `20889789`)
- **Academic Supervisors**:
  - **Prof. Peter Amoako-Yirenkyi** (*Department of Mathematics, KNUST*)
  - **Prof. Charles Sebil** (*Department of Mathematics, KNUST*)
- **Institutional Host**:
  - Department of Mathematics, Faculty of Physical Sciences, College of Science, KNUST, Kumasi, Ghana.
  - Date: **September 2026**

---

## 🎯 Executive Summary & Core Contributions

In sub-Saharan African pediatric outpatient departments (OPDs), overcrowded waiting rooms and severe caregiver-child occlusion cause severe patient undercounting. Traditional computer vision detectors misclassify toddlers held on caregiver laps or carried behind adults. 

This thesis develops a mathematically grounded, two-stage computer vision system running entirely on edge CPU hardware without high-end GPUs:

1. **Stage 1 (Coarse Primary Bounding)**: Lightweight YOLO26s detector identifies candidate patient bounding boxes at low inference thresholds.
2. **Stage 2 (Margin-Expanded Crop Classification)**: A specialized $224 \times 224$ classifier evaluates the candidate crop with contextual margin to resolve heavy occlusion.
3. **Bayesian Height Prior Calibration**: Implements a calibrated standing adult height prior threshold ($\theta_{\text{adult}} = 0.32$) and seated occupant bounding-box aspect ratio threshold ($\mathrm{AR} < 1.15$) to prevent parent-child identity confusion.
4. **Allometric Cephalocaudal Ratio ($R_{\text{ceph}}$)**: Verifies child anatomy through age-stratified head-to-body proportion analysis ($R_{\text{ceph}} \ge 0.20$ for pediatric subjects vs. $R_{\text{ceph}} \le 0.14$ for adults).
5. **ByteTrack 8D Kalman State Estimation**: Continuously updates tracklets across temporary dropouts without fragmenting unique patient counts.
6. **Zero-Margin Overfull Layout**: Fully audited 70-page dissertation strictly conforming to KNUST Mathematics binding margin standards ($4.0\text{ cm}$ left, $2.5\text{ cm}$ right).

---

## 📊 Empirical Benchmarks (Intel Core i7-10610U Edge CPU)

Evaluations performed on a standard clinical hospital workstation (**Intel Core i7-10610U CPU @ 1.80 GHz**, 16.0 GB RAM, AVX2 acceleration):

| Model / Architecture | Input Res | Precision | Frame Latency | Throughput (FPS) | mAP@50 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Single-Stage YOLOv8n | $640 \times 640$ | FP32 | $147.1\text{ ms}$ | $6.8\text{ FPS}$ | $61.4\%$ |
| Single-Stage YOLOv11n | $640 \times 640$ | FP32 | $138.9\text{ ms}$ | $7.2\text{ FPS}$ | $64.2\%$ |
| **Proposed Dual-Stage Pipeline** | $640 \times 640$ | **INT8 / AVX2** | **$123.5\text{ ms}$** | **$8.1\text{ FPS}$** | **$67.1\%$** |

### ⚠️ Generalization Disclaimer
As explicitly documented in Chapter 4 and Chapter 5:
- Real-world clinical performance achieved $67.1\%$ raw detection mAP@50 across standard OPD test sequences.
- Occlusion mitigation is bounded by the visual camera angle and ambient hospital illumination.
- Multi-camera handoff currently uses upper/lower body color appearance histograms; severe lighting transitions across distinct clinical wards warrant future metric deep-learning re-ID integration.

---

## 🔗 Public Research Repositories

This thesis is accompanied by 4 open-source repositories:

1. **[`jackysmith040/pediatric_thesis_latex`](https://github.com/jackysmith040/pediatric_thesis_latex)** *(This Repository)*:
   - Complete 70-page LaTeX thesis source code, native KNUST frontmatter, chapter files, high-resolution figures, and compiled PDF deliverables.
2. **[`jackysmith040/pediatric_final_year_project`](https://github.com/jackysmith040/pediatric_final_year_project)**:
   - Production dual-stage crop classifier implementation, ByteTrack 8D Kalman filter integration, and hospital calibration profiles (`pediatric_counter`).
3. **[`jackysmith040/pediatric_research_thesis`](https://github.com/jackysmith040/pediatric_research_thesis)**:
   - Theoretical framework, research notebooks, and mathematical proof formulations.
4. **[`jackysmith040/pediatric_research_lab`](https://github.com/jackysmith040/pediatric_research_lab)**:
   - Model weights, INT8 quantization pipelines, and raw benchmark logs.

---

## 🛠️ Cross-Platform Compilation Guide

The repository includes a self-bootstrapping, cross-platform build toolchain based on [Tectonic](https://tectonic-typesetting.github.io/). No massive 5GB+ TeX Live or MiKTeX installation is required. Tectonic downloads required CTAN packages dynamically on demand.

### 1. Prerequisites
Clone this repository and ensure Python 3.10+ is available:
```bash
git clone git@github.com:jackysmith040/pediatric_thesis_latex.git
cd pediatric_thesis_latex
```

### 2. Install Tectonic (Zero-Config LaTeX Engine)
- **Windows**: Built-in or via Scoop:
  ```powershell
  scoop install tectonic
  ```
- **macOS** (Apple Silicon M1–M4 or Intel):
  ```bash
  brew install tectonic
  ```
- **Ubuntu / Debian / Linux**:
  ```bash
  sudo apt install tectonic
  # or standalone:
  curl --proto '=https' --tlsv1.2 -fsSL https://drop-sh.tectonic-typesetting.net | sh
  ```

### 3. Compile the Thesis
Run the compiler from the project root:
```bash
tectonic -X compile template_override/latex/thesis_main.tex --outdir output/thesis
```
The compiled 70-page dissertation will be generated in [`output/thesis/`](output/thesis/):
- `output/thesis/Pediatric Patient Detection and Counting with Interactive Computer Vision and Dynamic Multi-Tracking in Clinical Triage Environments.pdf`

---

## 📁 Repository Layout

```text
pediatric_thesis_latex/
├── template_override/
│   └── latex/
│       ├── thesis_main.tex           # Master LaTeX root document
│       ├── frontmatter/              # Native KNUST frontmatter
│       │   ├── titlepage.tex         # KNUST official title page
│       │   ├── declaration.tex       # Signed candidate & supervisor declarations
│       │   ├── dedication.tex        # Dedication
│       │   ├── abstract.tex          # Calibrated abstract
│       │   ├── acknowledgment.tex    # Acknowledgments
│       │   └── abbreviations.tex     # List of abbreviations
│       ├── chapters/                 # Core dissertation chapters
│       │   ├── 01_introduction.tex   # Chapter 1: Introduction & Research Questions
│       │   ├── 02_literature_review.tex # Chapter 2: Literature Review & Gaps
│       │   ├── 03_methodology.tex    # Chapter 3: Methodology & Mathematical Formulations
│       │   ├── 04_results_and_discussion.tex # Chapter 4: Empirical Results & Disclaimers
│       │   ├── 05_conclusion.tex     # Chapter 5: Conclusions & Future Research
│       │   └── appendix.tex          # Official Appendix with Repository Catalog
│       ├── styles/                   # KNUST geometry and package styling
│       │   └── mystyle.sty
│       ├── bibliography/             # BibTeX reference database
│       │   └── references.bib
│       └── images/                   # 25+ verified architectural & empirical figures
├── output/
│   └── thesis/                       # Final compiled 70-page submission PDFs
├── input/                            # Research markdown notes & benchmark data
├── agent_helper/                     # Standalone Python compiler & verification engine
├── memory/                           # Autopoietic Conscience memory & episodic logs
├── HANDOFF.md                        # Session continuity & verification ledger
├── pyproject.toml                    # Python project packaging
└── README.md                         # Project documentation
```

---

## 📜 Compliance & Verification Checklist

- [x] **KNUST Mathematics Margins**: $4.0\text{ cm}$ left, $2.5\text{ cm}$ right ($411.02\text{ pt}$ printable boundary).
- [x] **Zero Margin Overfulls**: Audited via PyMuPDF across all 70 pages.
- [x] **ASD-STE100 Compliant**: Plain, unambiguous technical English; zero AI promotional jargon.
- [x] **Empirically Grounded**: Real benchmark numbers ($67.1\%$ mAP@50, $8.1\text{ FPS}$) without unbacked 100% claims.
- [x] **Git Cleanliness**: `.venv/`, binaries, and temporary caches excluded; thesis sources and PDFs tracked.

---

## 📄 License

This work is licensed under the [MIT License](LICENSE).  
Copyright (c) 2026 Fafali Dorkunor, Peter Amoah Mensah, Kwame Nkrumah University of Science and Technology.
