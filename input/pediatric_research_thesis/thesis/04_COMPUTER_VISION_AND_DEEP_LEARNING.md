# Chapter 4: Computer Vision & Deep Learning Engine

---

## 4.1 Dataset Curation & Pediatric Annotation Protocol

Supervised training of clinical pediatric detectors requires specialized datasets capturing the nuanced anatomical, morphological, and positional characteristics of children in clinical waiting environments. Standard pedestrian datasets (such as MS COCO, Pascal VOC, or Cityscapes) treat all human instances as a single homogeneous `person` class, failing to differentiate pediatric patients from adult caregivers.

### 4.1.1 Dataset Sources & Composition
The training corpus was compiled through an integrated curation pipeline combining the **Child Detection Dataset (CDD)** aggregated via Roboflow with specialized clinical hospital triage CCTV video sequences:
1. **`unified-dataset.ndjson`:** 2,414 annotated pediatric triage images and YOLO bounding box records capturing diverse outpatient waiting scenarios.
2. **`qh.ndjson`:** 507 high-resolution pediatric hospital CCTV frames capturing dense intake corridors, swaddled infants, and dim evening waiting rooms.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      DATASET COMPOSITION & PROTOCOL                     │
│                                                                         │
│  Class Category      Training Images   Validation Images   Test Images  │
│  ────────────────────────────────────────────────────────────────────   │
│  Adult Caregivers         4,820              1,240             620      │
│  Pediatric / Children     3,950              1,010             505      │
│  Carried / Occluded       1,830                460             230      │
│  ────────────────────────────────────────────────────────────────────   │
│  Total Annotated Boxes   10,600              2,710           1,355      │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.1.2 Annotation Protocols for Carried & Occluded Infants
To specifically address the "Invisible Child" phenomenon, annotation protocols incorporated rigorous bounding box guidelines stratified across three occlusion difficulty tiers:
1. **Clear Tier (Uninhibited Line of Sight):** Pediatric patients walking or sitting upright with $>80\%$ body visibility.
2. **Partial Tier (Moderate Physical Occlusion):** Children partially obscured by gurneys, waiting room benches, or caregiver limbs ($30\% - 60\%$ occlusion).
3. **Heavy / Carried Tier (Severe Swaddling & Torso Hugging):** 
   - **Fabric Slings & Kangaroo Wraps:** A distinct `Child` bounding box is annotated around the visible infant head and torso, even when $60\% - 85\%$ of the lower body is occluded by fabric wraps or adult arms.
   - **Caregiver Enclosure:** The `Adult` bounding box encompasses the full standing/sitting adult silhouette, creating an overlapping bounding box structure with the carried infant ($\text{IoU} \approx 0.30 - 0.65$).
   - **Severe Blanketing:** If facial features, ears, or swaddled contours are visible, a `Child` annotation is placed around the swaddled mass.

### 4.1.3 Data Augmentation Suite
To prevent overfitting and simulate harsh hospital environments, extensive offline and online augmentations were applied:
- **Photometric Distortions:** Hue jitter ($\pm 15^\circ$), Saturation ($\pm 25\%$), Value/Brightness ($\pm 35\%$) to simulate dim night shifts and harsh daylight glare.
- **Geometric Transformations:** Random horizontal flipping ($p=0.5$), random affine scaling ($0.8\times \text{ to } 1.2\times$), and mosaic stitching ($4\text{-image mosaic blending}$) to improve small-scale infant detection.
- **Random Erasing (Cutout):** Simulating occlusion by zeroing out random rectangular patches ($10\% - 25\%$ of bounding box area).

---

## 4.2 Vision Foundation Knowledge Distillation (DINOv3 $\to$ YOLO26s)

Standard supervised fine-tuning from COCO weights struggles with severe physical occlusion because convolutional backbones lack the global contextual attention required to differentiate folded blankets from swaddled infants. This research introduces a **Self-Supervised Vision Foundation Knowledge Distillation** framework transferring dense semantic representations from a **DINOv3 Vision Transformer Teacher** ($\mathcal{M}_T$) into a lightweight **YOLO26s Student** ($\mathcal{M}_S$).

```
       Input Image (640x640x3)
                 │
      ┌──────────┴────────────────────────────────────────┐
      ▼                                                   ▼
┌─────────────────────────────────────────┐   ┌─────────────────────────────────────────┐
│ TEACHER: DINOv3 ViT Foundation Model    │   │ STUDENT: YOLO26s Convolutional Backbone │
│ - ViT-S/16 or ViT-B/16 Self-Attention   │   │ - C2f / CSPDarknet Feature Pyramid      │
│ - Patch Tokens: P = 14x14               │   │ - Neck Output: F_S (Dim = 512)          │
│ - Dense Feature Map: F_T (Dim = 768)    │   └────────────────────┬────────────────────┘
└────────────────────┬────────────────────┘                        │
                     │                                             ▼
                     │                                ┌─────────────────────────┐
                     │                                │ Projection Head P(F_S)  │
                     │                                │ 1x1 Conv + LayerNorm    │
                     │                                │ (512-dim -> 768-dim)    │
                     │                                └────────────┬────────────┘
                     │                                             │
                     ▼                                             ▼
       ┌────────────────────────────────────────────────────────────────────────┐
       │               JOINT FEATURE DISTILLATION LOSS COMPUTATION              │
       │                                                                        │
       │    L_distill = α * L_cosine(F_T, P(F_S)) + β * L_MSE(F_T, P(F_S))      │
       └────────────────────────────────────────────────────────────────────────┘
```

### 4.2.1 Teacher and Student Architecture Topologies
- **Teacher ($\mathcal{M}_T$):** DINOv3 ViT-S/16 (21.8M params) or ViT-B/16 (85.8M params), pre-trained self-supervised on diverse web-scale imagery without labels. Yields patch token representations $F_T \in \mathbb{R}^{B \times C_T \times H_T \times W_T}$ where $C_T = 768$ and $(H_T, W_T) = (40, 40)$.
- **Student ($\mathcal{M}_S$):** YOLO26s (11.2M params), single-stage convolutional detector with PANet feature neck. Intermediate feature maps at stage P4 have shape $F_S \in \mathbb{R}^{B \times C_S \times H_S \times W_S}$ where $C_S = 512$ and $(H_S, W_S) = (40, 40)$.

### 4.2.2 Multi-Scale Feature Projection Head ($\mathcal{P}$)
Because the channel dimensions of the student ($C_S = 512$) and teacher ($C_T = 768$) differ, an intermediate trainable $1\times 1$ convolutional projection head $\mathcal{P}$ is inserted:

$$\mathcal{P}(F_S) = \text{GELU}\left( \text{LayerNorm}\left( \mathbf{W}_{\text{proj}} * F_S + \mathbf{b}_{\text{proj}} \right) \right)$$

Where $\mathbf{W}_{\text{proj}} \in \mathbb{R}^{768 \times 512 \times 1 \times 1}$. This aligns student latent vectors into the teacher's metric embedding space without modifying student runtime inference structures.

### 4.2.3 Mathematical Formulation of Distillation Loss
The feature distillation loss combines **Normalized Spatial Cosine Similarity Loss** ($\mathcal{L}_{\text{cos}}$) and **Normalized Mean Squared Error Loss** ($\mathcal{L}_{\text{MSE}}$):

$$\mathcal{L}_{\text{distill}} = \alpha \mathcal{L}_{\text{cos}}(F_T, \mathcal{P}(F_S)) + \beta \mathcal{L}_{\text{MSE}}(F_T, \mathcal{P}(F_S))$$

Where:
$$\mathcal{L}_{\text{cos}} = 1 - \frac{1}{H \cdot W} \sum_{i=1}^H \sum_{j=1}^W \frac{F_T(i, j) \cdot \mathcal{P}(F_S)(i, j)}{\|F_T(i, j)\|_2 \|\mathcal{P}(F_S)(i, j)\|_2 + \epsilon}$$

$$\mathcal{L}_{\text{MSE}} = \frac{1}{C \cdot H \cdot W} \sum_{c=1}^C \sum_{i=1}^H \sum_{j=1}^W \left( \hat{F}_T(c, i, j) - \widehat{\mathcal{P}(F_S)}(c, i, j) \right)^2$$

Here, $\hat{F}$ denotes L2-normalized channel feature activations. Cosine loss aligns directional semantic orientation (enforcing part-level object saliency), while MSE loss preserves activation magnitude calibration. Hyperparameters are tuned to $\alpha = 0.60$ and $\beta = 0.40$.

### 4.2.4 Two-Stage Training Regimen
The model is trained in two sequential phases:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TWO-STAGE TRAINING REGIMEN                           │
│                                                                         │
│  STAGE 1: Dense Representation Distillation Pre-Training (25 Epochs)    │
│  - Dataset: Unlabeled clinical & triage images (hospital corridors)     │
│  - Loss: L_distill = α * L_cos + β * L_MSE                              │
│  - Optimizes: Student backbone & Neck + Projection Head P               │
│  - Result: Student inherits rich DINOv3 semantic boundary representations│
│                                                                         │
│  STAGE 2: Supervised Pediatric Fine-Tuning (30 Epochs)                  │
│  - Dataset: Labeled pediatric dataset (pediatric_data.yaml, 10.6K boxes)│
│  - Loss: L_total = L_CIoU + L_DFL + L_BCE                               │
│  - Head: Dual-Class (Adult vs. Child)                                   │
│  - Checkpoint: yolo26s_distilled.pt (Published Final Model)             │
└─────────────────────────────────────────────────────────────────────────┘
```

```
Table 4.1: Distillation and fine-tuning hyperparameter specifications
+-----------------------------+-----------------------+-----------------------+
| Hyperparameter              | Stage 1: Distillation | Stage 2: Fine-Tuning  |
+-----------------------------+-----------------------+-----------------------+
| Teacher Architecture        | DINOv3 ViT-S/16       | None (Detached)       |
| Student Architecture        | YOLO26s (11.2M)       | YOLO26s Distilled     |
| Input Image Resolution      | 640 x 640 x 3         | 640 x 640 x 3         |
| Optimizer                   | AdamW (lr = 1e-4)     | SGD (lr = 1e-2, mom=0.937)
| Weight Decay                | 1e-4                  | 5e-4                  |
| Batch Size                  | 16                    | 16                    |
| Epochs                      | 25                    | 30                    |
| Loss Functions              | Cosine + MSE Loss     | CIoU + DFL + BCE      |
| Cosine Weight (α)           | 0.60                  | N/A                   |
| MSE Weight (β)              | 0.40                  | N/A                   |
| Hardware Acceleration       | CUDA (GPU Cloud)      | CUDA (GPU Cloud)      |
+-----------------------------+-----------------------+-----------------------+
```

---

## 4.3 Slicing Aided Hyper Inference (SAHI) Architecture

To resolve tiny carried infants in high-mounted, wide-angle 1080p hospital triage cameras, the vision engine incorporates **Slicing Aided Hyper Inference (SAHI)** (Akyon et al., 2022).

```
┌──────────────────────────────────────────────────────────────────────────┐
│                   SAHI MULTI-SCALE PATCH SLICING PIPELINE                │
│                                                                          │
│  Input 1080p CCTV Frame (1920x1080)                                      │
│  ┌──────────────────────────────────────────────┐                        │
│  │  Slice 1 (640x640)     Slice 2 (640x640)     │ 20% Overlap Ratio      │
│  │  ┌────────────┐        ┌────────────┐        │ (dx = 512, dy = 512)   │
│  │  │   Infant   │        │            │        │                        │
│  │  └────────────┘        └────────────┘        │                        │
│  │  Slice 3 (640x640)     Slice 4 (640x640)     │                        │
│  └──────────────────────────────────────────────┘                        │
│                         │                                                 │
│                         ▼ Parallel Batched ONNX Inference                 │
│  Predictions across Slices {B_s} + Full Downsampled Frame {B_full}        │
│                         │                                                 │
│                         ▼ Coordinate Reconstruction: (x_g, y_g)           │
│  x_g = x_s + x_offset,  y_g = y_s + y_offset                              │
│                         │                                                 │
│                         ▼ Non-Maximum Suppression (IoU Thresh = 0.45)     │
│  Final Deduplicated Global Bounding Boxes                                │
└──────────────────────────────────────────────────────────────────────────┘
```

### SAHI Algorithmic Execution:
1. **Grid Generation:** The image of size $(W, H)$ is tiled into $N_{\text{slices}}$ overlapping windows of size $M \times M$ ($640\times 640$) with step size $\Delta = M(1 - r_{\text{overlap}})$.
2. **Batched Local Inference:** Patches are batched and executed through the distilled ONNX detector, preventing scale collapse for infants $<32\times 32\text{ px}$.
3. **Global Spatial Mapping:** For each slice at offset $(x_{\text{off}}, y_{\text{off}})$, detected box coordinates are shifted:
   $$[x_1^{\text{global}}, y_1^{\text{global}}, x_2^{\text{global}}, y_2^{\text{global}}] = [x_1 + x_{\text{off}}, y_1 + y_{\text{off}}, x_2 + x_{\text{off}}, y_2 + y_{\text{off}}]$$
4. **NMS Union:** Sliced detections and full-frame global predictions are merged via Non-Maximum Suppression at $\text{IoU} = 0.45$.

---

## 4.4 Model Optimization: PyTorch to ONNX Runtime Quantization

While PyTorch models (.pt) offer seamless training and debugging, raw PyTorch FP32 inference introduces substantial computational overhead on consumer CPUs due to dynamic computational graph overhead and unvectorized matrix multiplication operations.

To enable high-throughput edge execution, all models were exported to the **Open Neural Network Exchange (ONNX)** format and executed via the **ONNX Runtime engine** (`onnxruntime-cpu`):

$$\text{PyTorch (.pt Graph)} \xrightarrow{\text{torch.onnx.export}} \text{Static ONNX Computation Graph} \xrightarrow{\text{Graph Optimization}} \text{ONNX Runtime Execution}$$

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PYTORCH vs. ONNX RUNTIME BENCHMARK                   │
│                                                                         │
│  Model Architecture           Framework        CPU Latency    FPS Rate  │
│  ─────────────────────────────────────────────────────────────────────  │
│  yolo26s_distilled.pt         PyTorch FP32       59.8 ms      16.7 FPS  │
│  yolo26s_distilled.onnx       ONNX Runtime CPU   28.4 ms      35.2 FPS  │
│  ─────────────────────────────────────────────────────────────────────  │
│  OVERALL SPEEDUP FACTOR:      2.1x FASTER (Zero GPU Dependency)         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.4.1 ONNX Metadata Extraction & AST Literal Parsing
A critical engineering challenge in dynamic ONNX model execution is extracting class label mappings (`names`) directly from exported `.onnx` binary protobuf graphs.

In `Detector._init_model`, the system implements an automated metadata parsing fallback:
```python
if (not names or len(names) == 0) and is_onnx:
    try:
        import onnx
        import ast
        loaded_onnx = onnx.load(selected_path)
        props = {p.key: p.value for p in loaded_onnx.metadata_props}
        if 'names' in props:
            names = ast.literal_eval(props['names'])
    except Exception as meta_err:
        logger.warning(f"Could not extract metadata class names: {meta_err}")
```
This guarantees seamless introspection of class names directly from ONNX binary headers without external `.yaml` configuration dependencies.

---

## 4.5 Dynamic Class Mapping & Architecture Auto-Resolution

To support heterogeneous model architectures at runtime (distilled models, dual-class fine-tuned models, kids-only models, or base COCO detectors), the vision engine implements an automated **Class Resolution Algorithm** in `src/engine/detector.py`:

```
                       Loaded Model Names Dict
                                 │
           ┌─────────────────────┼─────────────────────┐
           ▼                     ▼                     ▼
   Contains 'child' /     Contains 'child' /    Contains only
   'kid' AND 'adult'      'kid' ONLY            'person' (COCO)
           │                     │                     │
           ▼                     ▼                     ▼
    Dual-Class Mode        Kids-Only Mode        COCO Heuristic Mode
    child_id = found       child_id = found      child_id = 1, adult_id = 0
    adult_id = found       adult_id = -1         uses_coco_person = True
```

### Mathematical Logic for Class ID Assignment:
Let $\mathcal{N} = \{ (k, v) \}$ be the dictionary mapping class indices $k \in \mathbb{N}$ to class label strings $v \in \Sigma^*$.

$$\text{child\_id} = \begin{cases} k & \text{if } \exists (k, v) \in \mathcal{N} \text{ s.t. } \text{lower}(v) \in \{\text{'child'}, \text{'kid'}, \text{'pediatric'}\} \\ -1 & \text{otherwise} \end{cases}$$

$$\text{adult\_id} = \begin{cases} k & \text{if } \exists (k, v) \in \mathcal{N} \text{ s.t. } \text{'adult'} \in \text{lower}(v) \\ -1 & \text{otherwise} \end{cases}$$

This automated introspection allows clinical staff to toggle between models on the fly via UI dropdowns without server reboots or configuration edits.

---

## 4.6 Illumination Normalization: LAB Color Space CLAHE Pipeline

Clinical emergency rooms exhibit extreme lighting fluctuations. To maintain high pediatric detection recall during dark night shifts without introducing color-shift artifacts, the engine applies Contrast Limited Adaptive Histogram Equalization (**CLAHE**) in the CIE $L^*a^*b^*$ color space.

```
       Raw BGR Frame
             │
             ▼
     cv2.cvtColor(BGR2LAB)
             │
             ├──────────────────────┬──────────────────────┐
             ▼                      ▼                      ▼
     L* (Luminance Channel)   a* (Green-Red)         b* (Blue-Yellow)
             │                      │                      │
             ▼                      │                      │
     CLAHE Equalization             │                      │
     - clip_limit = 2.0             │                      │
     - tile_grid = (8, 8)           │                      │
             │                      │                      │
             ▼                      │                      │
     L*_enhanced                    │                      │
             │                      │                      │
             └──────────────────────┼──────────────────────┘
                                    ▼
                          cv2.merge((L*, a*, b*))
                                    │
                                    ▼
                          cv2.cvtColor(LAB2BGR)
                                    │
                                    ▼
                         Enhanced Output Frame
```

### Mathematical Implementation (`Detector.apply_clahe`):
Let $I_{\text{BGR}}(x, y)$ be the input color image.
1. Convert to LAB color space: $I_{\text{LAB}}(x, y) = \mathcal{T}_{\text{BGR}\to\text{LAB}}(I_{\text{BGR}}(x, y))$.
2. Separate into luminance $L^*(x, y)$ and chrominance $a^*(x, y), b^*(x, y)$.
3. Apply localized adaptive histogram clipping with clip limit $\beta = 2.0$ over an $8 \times 8$ grid of contextual tiles:

$$L_{\text{enhanced}}^*(x, y) = \text{CLAHE}\left( L^*(x, y); \text{clip\_limit}=2.0, \text{grid}=(8, 8) \right)$$

4. Recombine and convert back to BGR space:

$$I_{\text{out}}(x, y) = \mathcal{T}_{\text{LAB}\to\text{BGR}}\left( \left[ L_{\text{enhanced}}^*(x, y), a^*(x, y), b^*(x, y) \right] \right)$$

This preprocessing step increases pediatric edge gradient contrast by **34.2%** in low-light conditions ($<50\text{ lux}$), boosting small-target detection recall while preserving skin tone fidelity.

---

## 4.7 Perspective-Aware Classification Heuristics

When operating with un-fine-tuned generic COCO detectors (where all humans are classified as generic `person` class 0), the system activates a specialized **Ground-Plane Perspective Normalization** algorithm (`Detector.calculate_perspective_class`).

```
       Top of Frame (y = 0.0)
       ┌────────────────────────────────────────────────────────┐
       │   Perspective Horizon Y_h (y = 0.20)                   │
       │   ──────────────────────────────────────────────────   │
       │   Far Field: Expected Adult Height h_far = 0.22        │
       │                                                        │
       │                                                        │
       │   Near Field: Expected Adult Height h_near = 0.55      │
       │   ──────────────────────────────────────────────────   │
       └────────────────────────────────────────────────────────┘
       Bottom of Frame (y = 1.0)
```

### Perspective Normalization Formulation:
1. **Vertical Bottom Coordinate:** Let $y_{\text{bottom}} = \frac{y_2}{H_{\text{frame}}} \in [0, 1]$ represent the normalized ground-contact point.
2. **Normalized Depth Coordinate:**

$$\tilde{y} = \max\left( 0.0, \min\left( 1.0, \frac{y_{\text{bottom}} - Y_{\text{horizon}}}{1.0 - Y_{\text{horizon}}} \right) \right)$$

Where $Y_{\text{horizon}} = 0.20$.

3. **Expected Adult Pixel Height:**

$$h_{\text{expected\_adult}}(\tilde{y}) = h_{\text{far}} + \tilde{y} \cdot (h_{\text{near}} - h_{\text{far}})$$

Where $h_{\text{far}} = 0.22$ and $h_{\text{near}} = 0.55$.

4. **Sitting-Pose Aspect Ratio Compensation:**  
   When adult patients sit on benches, their projected vertical height decreases by $35\% - 50\%$, while their bounding box aspect ratio ($AR = w / h$) expands. To prevent seated adults from being misclassified as children:

$$\text{Effective Height } h_{\text{eff}} = \begin{cases} h_{\text{box}} \cdot 1.50 & \text{if } \frac{w_{\text{box}}}{h_{\text{box}}} > 0.55 \\ h_{\text{box}} & \text{otherwise} \end{cases}$$

5. **Normalized Scale Ratio & Classification Threshold:**

$$R_{\text{norm}} = \frac{h_{\text{eff}}}{h_{\text{expected\_adult}}(\tilde{y})}$$

$$\text{Class} = \begin{cases} \text{Child} & \text{if } R_{\text{norm}} < 0.70 \\ \text{Adult} & \text{if } R_{\text{norm}} \ge 0.85 \\ \text{Child} & \text{if } 0.70 \le R_{\text{norm}} < 0.85 \land AR > 0.60 \\ \text{Adult} & \text{otherwise} \end{cases}$$

This geometric formulation enables robust adult-versus-child categorization even when relying on off-the-shelf, non-fine-tuned base models.
