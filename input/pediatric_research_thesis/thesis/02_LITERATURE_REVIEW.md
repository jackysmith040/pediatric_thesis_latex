# Chapter 2: Literature Review & Theoretical Foundations

---

## 2.1 Overview of Object Detection in Clinical & Surveillance Domains

Computer vision in healthcare surveillance has transitioned from basic background subtraction algorithms to deep convolutional and transformer-based neural networks capable of complex scene comprehension. Early clinical monitoring systems relied on classical motion cues, background mixture models (e.g., Gaussian Mixture Models, GMMs), and handcrafted feature extractors such as Histograms of Oriented Gradients (HOG) coupled with Support Vector Machines (SVM) (Dalal & Triggs, 2005). 

While HOG-SVM detectors achieved baseline success in pedestrian detection under controlled laboratory environments, they exhibited extreme fragility in real-world clinical waiting rooms. In clinical triage environments, non-rigid body deformations, variable patient sitting postures, severe occlusion from blankets and furniture, and shifting illumination frequently caused severe feature degradation, resulting in prohibitive false-positive and false-negative rates.

The advent of Deep Convolutional Neural Networks (CNNs) initiated a paradigm shift in visual object recognition. Modern object detectors automatically learn hierarchical visual representations, ranging from low-level edge and texture primitives in shallow layers to complex semantic concepts (such as human torsos, facial profiles, and carried infants) in deep layers (Krizhevsky et al., 2012; LeCun et al., 2015).

### 2.1.1 Mathematical Foundations of Computer Vision: From Finite Differences to Convolution

To fully appreciate the representational power of modern Deep Convolutional Neural Networks, it is essential to establish the mathematical foundations of classical computer vision, tracing the evolution from continuous calculus to discrete tensor convolution.

**Image Representation (Continuous to Discrete)**
A physical visual scene is fundamentally a continuous two-dimensional light signal $f(x, y)$. For computational processing, this continuous signal must be sampled and quantized into a discrete mathematical structure. A grayscale image is represented as a matrix $I \in \mathbb{R}^{H \times W}$, where $H$ and $W$ represent height and width, and each element corresponds to an 8-bit quantized intensity value $[0, 255]$. For color images, this extends to a three-dimensional tensor $I \in \mathbb{R}^{H \times W \times 3}$, encoding the Red, Green, and Blue (RGB) color channels. By mapping physical light to discrete integer spaces, the image becomes a purely linear algebra construct, optimized for differential operations.

**The Calculus of Edges**
In computer vision, an edge corresponds to a spatial region exhibiting rapid intensity variation. In continuous calculus, the rate of change is measured using the 1D derivative:

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

Because digital images are discrete pixel grids, the infinitesimally small step $h \to 0$ cannot be achieved. The smallest possible step size is $h = 1$ pixel. Consequently, we approximate the continuous derivative using discrete 2D partial derivatives (finite differences) to compute the horizontal and vertical rates of change:

$$\frac{\partial I}{\partial x} \approx I(x+1, y) - I(x-1, y)$$
$$\frac{\partial I}{\partial y} \approx I(x, y+1) - I(x, y-1)$$

**Deriving the Sobel Kernel**
A naive finite difference kernel such as $[-1, 0, 1]$ is highly susceptible to high-frequency image noise. The Sobel operator elegantly combines spatial differentiation with orthogonal smoothing. Mathematically, the horizontal Sobel operator $G_x$ is derived through the outer product of a 1D vertical smoothing vector $S_y$ and a 1D horizontal derivative vector $D_x$:

$$G_x = S_y \otimes D_x = \begin{bmatrix} 1 \\ 2 \\ 1 \end{bmatrix} \begin{bmatrix} -1 & 0 & 1 \end{bmatrix} = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}$$

This derivation ensures that $G_x$ extracts horizontal gradients while concurrently applying a localized low-pass Gaussian-like filter vertically to suppress random static. A complementary $G_y$ kernel is derived by transposing the operation to detect vertical gradients.

**Convolution Arithmetic ($I * G_x$)**
The convolution operator ($*$) applies these discrete kernels across the entire image space. A spatial sliding window mechanism performs element-wise Hadamard multiplication between the kernel and the corresponding image patch, subsequently summing the resulting products into a singular scalar response:

$$S(x, y) = (I * G_x)(x, y) = \sum_{m=-1}^{1} \sum_{n=-1}^{1} I(x-m, y-n) G_x(m, n)$$

A high absolute sum strictly signifies the mathematical presence of a sharp structural edge.

**Gradient Vectors (Magnitude and Direction)**
Applying both $G_x$ and $G_y$ yields a two-dimensional Gradient Vector $\nabla I$ for every spatial pixel. The structural strength of an edge is captured by the Gradient Magnitude, formulated via the Pythagorean theorem:

$$|G| = \sqrt{G_x^2 + G_y^2}$$

The spatial orientation of the edge, pointing strictly towards the direction of steepest intensity ascent (from dark to light), is computed as the Gradient Angle:

$$\theta = \text{atan2}(G_y, G_x)$$

The two-argument $\text{atan2}$ function is computationally vital, as it captures the full $360^\circ$ angular range based on the coordinate signs of the partial derivatives, a requirement impossible with standard arctangent functions.

**Mathematical Noise Filtering: Gaussian Blurring and NMS**
To prevent false-positive edge detections triggered by camera sensor static, images undergo 2D Gaussian Blurring prior to differentiation. The continuous Gaussian kernel places maximum statistical weight on the central pixel, decaying exponentially outward:

$$G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}$$

Furthermore, raw gradient magnitudes often produce thick, blurry structural boundaries. To thin these boundaries into precise single-pixel edges, Non-Maximum Suppression (NMS) is applied. NMS mathematically evaluates the gradient magnitude $|G|$ of a pixel along its gradient vector $\theta$. If the pixel is not the absolute local maximum compared to its adjacent neighbors along the vector path, its intensity is forcefully suppressed to $0$.

**Transitioning to Deep Learning**
The transition from finite differences to Convolutional Neural Networks (CNNs) represents a fundamental shift in mathematical optimization. While classical vision relies on manually derived, fixed-weight kernels (like the Sobel operator), deep CNNs (such as the YOLO architecture) treat the values of convolutional kernels as learnable weights $\mathbf{W}$. These weights are iteratively optimized via backpropagation and stochastic gradient descent to extract highly complex, hierarchical features automatically. 

Once optimal features are mathematically extracted, deep learning reformulates the problem from structural extraction into bounding box coordinate regression. The spatial center coordinates $(t_x, t_y)$ of a predicted pediatric bounding box are bound strictly within the grid cell limits $(0, 1)$ utilizing the logistic Sigmoid activation function:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

The final spatial accuracy of these regression coordinates is evaluated by the Intersection over Union (IoU) ratio, determining the precise area of overlap between the predicted bounding box ($B_p$) and the empirical ground truth ($B_g$):

$$\text{IoU} = \frac{\text{Area}(B_p \cap B_g)}{\text{Area}(B_p \cup B_g)}$$

In summation, whether computing explicit outer products to construct a Sobel filter, or optimizing vast tensor weights via backpropagation in YOLO, the entire architecture of computer vision fundamentally rests upon the unified mathematical pillars of linear algebra, differential calculus, and spatial convolution.

---

## 2.2 Deep Learning Architectures: Two-Stage vs. Single-Stage Detectors

Deep learning object detection frameworks are broadly categorized into two structural paradigms: **Two-Stage Detectors** and **Single-Stage Detectors**.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                   TWO-STAGE DETECTOR (e.g., Faster R-CNN)                │
│                                                                          │
│  Input Image ──► Backbone (ResNet) ──► Region Proposal Network (RPN)     │
│                                                   │                      │
│                                                   ▼                      │
│                                     RoI Pooling / RoIAlign               │
│                                                   │                      │
│                                                   ▼                      │
│                                       Classification & Bounding Box Reg  │
│                                       (High Latency: 80 - 150 ms)        │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                   SINGLE-STAGE DETECTOR (e.g., YOLO Series)              │
│                                                                          │
│  Input Image ──► Backbone (CSPDarknet) ──► Neck (PANet/FPN) ──► Head     │
│                                                                   │      │
│                                                                   ▼      │
│                                              Dense Bounding Box & Class  │
│                                              (Low Latency: 15 - 35 ms)   │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.2.1 Two-Stage Object Detectors
Two-stage frameworks, epitomized by the R-CNN family, specifically R-CNN (Girshick et al., 2014), Fast R-CNN (Girshick, 2015), Faster R-CNN (Ren et al., 2015), and Mask R-CNN (He et al., 2017), divide detection into two sequential subtasks:
1. **Region Proposal Generation:** A Region Proposal Network (RPN) slides over convolutional feature maps to propose candidate regions of interest (RoIs) likely to contain objects.
2. **Feature Extraction and Classification:** RoI Pooling or RoIAlign extracts fixed-size feature vectors from proposed regions, feeding them into fully connected layers for category classification and bounding box coordinate refinement.

While two-stage detectors historically achieved superior localization accuracy on dense, small-scale objects, their multi-step computational pipeline introduces significant latency (often $80\text{ to }150\text{ ms per frame}$ on high-end GPUs). In edge-deployed clinical monitoring where multiple video streams must be processed synchronously with zero delay, the computational overhead of two-stage architectures is prohibitively expensive.

### 2.2.2 Single-Stage Object Detectors: The YOLO Revolution
Single-stage detectors, pioneered by the **You Only Look Once (YOLO)** framework (Redmon et al., 2016) and the Single Shot MultiBox Detector (SSD) (Liu et al., 2016), reframe object detection as a unified spatial regression problem. Rather than generating separate region proposals, single-stage networks process the entire image in a single feedforward pass, predicting bounding box coordinates $(x, y, w, h)$, objectness confidence scores $C$, and class probability distributions $P(\text{Class}_i)$ across dense spatial grid cells simultaneously.

The YOLO architecture has undergone profound evolutionary milestones:
- **YOLOv3 & YOLOv4 (Redmon & Farhadi, 2018; Bochkovskiy et al., 2020):** Introduced multi-scale Feature Pyramid Networks (FPN), Path Aggregation Networks (PANet), and Cross-Stage Partial Connections (CSPNet), significantly enhancing multi-scale feature representation.
- **YOLOv8 & YOLOv11 (Ultralytics, 2023; 2024):** Adopted an **anchor-free** detection head, eliminating predefined anchor box heuristics and directly predicting offsets from bounding box centers. This eliminated manual anchor hyperparameter tuning and dramatically improved localization accuracy for irregular object aspect ratios.
- **YOLO26s Architecture:** Incorporates optimized C2f building blocks and decoupled regression-classification heads with dynamic channel scaling, minimizing parameter redundancy while maintaining high representational capacity.
- **Loss Formulations in Modern YOLO:** Modern single-stage networks employ composite multi-task loss functions combining Distribution Focal Loss (DFL) and Complete Intersection-over-Union (CIoU) loss for spatial regression:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{box}} \mathcal{L}_{\text{CIoU}} + \lambda_{\text{dfl}} \mathcal{L}_{\text{DFL}} + \lambda_{\text{cls}} \mathcal{L}_{\text{BCE}}$$

Where:
$$\mathcal{L}_{\text{CIoU}} = 1 - \text{IoU} + \frac{\rho^2(b, b^{\text{gt}})}{c^2} + \alpha v$$

Here, $\rho(b, b^{\text{gt}})$ represents the Euclidean distance between predicted and ground-truth bounding box centroids, $c$ is the diagonal length of the smallest enclosing box, and $\alpha v$ is an aspect ratio consistency penalty. This formulation ensures rapid convergence and high geometric fidelity on non-standard bounding box shapes characteristic of carried pediatric patients.

---

## 2.3 Knowledge Distillation & Foundation Models in Computer Vision

While lightweight single-stage convolutional networks (such as YOLO26s) satisfy edge inference constraints, their limited parameter capacity ($11.2\text{ M}$ parameters) causes severe feature collapse when encountering heavily occluded, non-rigid, or swaddled infants. Conversely, massive **Vision Foundation Models** (such as DINOv2 and DINOv3 Vision Transformers) possess extraordinary zero-shot semantic representation capabilities, but require hundreds of millions of parameters, preventing real-time CPU deployment. **Knowledge Distillation (KD)** bridges this fundamental divide.

```
┌──────────────────────────────────────────────────────────────────────────┐
│            DENSE FOUNDATION KNOWLEDGE DISTILLATION (DINOv3 -> YOLO26s)   │
│                                                                          │
│   Teacher: DINOv3 ViT Foundation Model (85M+ Params, Patch Tokens)       │
│   Dense Semantic Features F_T (Dim = 768)                                │
│                     │                                                    │
│                     ├──► Cosine Similarity Loss L_cos ◄──┐               │
│                     │                                    │               │
│                     └──► Mean Squared Error Loss L_mse ◄─┼───────────────┤
│                                                          │               │
│   Projected Student Features P(F_S) (1x1 Conv: 512 -> 768)               │
│   Student: YOLO26s Convolutional Neck (11.2M Params, Real-Time CPU)      │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.3.1 Logit-Based vs. Dense Feature-Based Distillation
Knowledge distillation, originated by Hinton et al. (2015), initially operated on output logit probability distributions using temperature-scaled softmax cross-entropy:

$$\mathcal{L}_{\text{logit}} = \tau^2 \cdot \text{KL}\left( \sigma\left(\frac{\mathbf{z}_S}{\tau}\right) \, \Big|\Big| \, \sigma\left(\frac{\mathbf{z}_T}{\tau}\right) \right)$$

While logit distillation transfers inter-class dark knowledge in classification tasks, it provides insufficient spatial guidance for dense object detection.

To guide bounding box localization and fine-grained visual discovery under severe occlusion, **Feature-Based Knowledge Distillation** (Romero et al., 2014, *FitNets*; Zagoruyko & Komodakis, 2017; Tian et al., 2020) aligns intermediate latent activation tensors between the teacher network $\mathcal{M}_T$ and student network $\mathcal{M}_S$.

### 2.3.2 Self-Supervised Vision Transformers: The DINO Architecture Family
The **DINO (Self-distillation with no labels)** family, spanning DINO (Caron et al., 2021), DINOv2 (Oquab et al., 2023), and DINOv3 (Meta AI, 2024), trains Vision Transformers (ViTs) on billions of unlabeled images using a self-supervised student-teacher momentum framework without human class labels.

DINO models partition an input image $I \in \mathbb{R}^{H \times W \times 3}$ into a grid of non-overlapping patches $p \in \mathbb{R}^{N \times (P^2 \cdot C)}$ (where $P=14$ or $16$), processing them through multi-head self-attention layers:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

A remarkable emergent property of DINO representations is that their intermediate patch tokens encode **explicit semantic segmentations and part-level boundaries** without ever being trained on segmentation masks. A DINOv3 Vision Transformer easily distinguishes the structural boundary of an infant head against a caregiver's clothing because its multi-head attention heads capture global relational context across the entire image.

### 2.3.3 Distilling Foundation Priors into Edge Convolutional Detectors
By distilling DINOv3 latent token representations into intermediate YOLO26s feature pyramid layers during a self-supervised pre-training phase, the lightweight convolutional student inherits the foundation model's invariant semantic priors. When the student is subsequently fine-tuned on clinical triage data, it retains high sensitivity to obscured infant contours, directly neutralizing the Invisible Child phenomenon while maintaining $30\text{ FPS}$ edge CPU execution speeds.

---

## 2.4 Small Object Detection & Slicing Aided Hyper Inference (SAHI)

In clinical triage surveillance, high-mounted wide-angle cameras capture entire waiting rooms ($1920\times 1080$ resolution). However, an infant carried by a caregiver in the background of such a scene may occupy a minuscule pixel bounding box of only $24\times 24$ pixels ($<0.03\%$ of total image area).

Standard convolutional detectors downsample input images to fixed sizes (e.g., $640\times 640$). Under $32\times$ downsampling in deep backbone layers (P5 feature maps), a $24\times 24$ infant bounding box is reduced to less than a single spatial pixel ($0.75\times 0.75$), causing catastrophic feature vanishing.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                   SLICING AIDED HYPER INFERENCE (SAHI)                   │
│                                                                          │
│   Full 1080p Image (1920x1080)                                           │
│   ┌──────────────────────────────────────────────┐                       │
│   │  [Slice 1]    [Slice 2]    [Slice 3]         │ Overlapping Patches   │
│   │  ┌──────┐     ┌──────┐     ┌──────┐          │ (e.g. 640x640, 20% ov)│
│   │  │ Infant│◄───┼──────┼─────┼──────┤          │                       │
│   │  └──────┘     └──────┘     └──────┘          │                       │
│   │  [Slice 4]    [Slice 5]    [Slice 6]         │                       │
│   └──────────────────────────────────────────────┘                       │
│                         │                                                │
│                         ▼ Parallel Patch Inference                       │
│   Batch Detections across all Slices + Full Frame                        │
│                         │                                                │
│                         ▼ Non-Maximum Suppression (NMS) Coordinate Shift │
│   Final Merged High-Resolution Global Detections                         │
└──────────────────────────────────────────────────────────────────────────┘
```

**Slicing Aided Hyper Inference (SAHI)** (Akyon et al., 2022) overcomes this scale vanishing limitation:
1. **Dynamic Patch Partitioning:** The high-resolution frame is sliced into overlapping patches $P_k$ of dimension $M \times N$ with overlap ratio $r_{\text{overlap}} \in [0.15, 0.25]$.
2. **Local Fine-Grained Inference:** Each slice is fed to the object detector at full native resolution, magnifying small-scale carried infants by $3\times \text{ to } 5\times$ in pixel density.
3. **Coordinate Transformation & Global Merging:** Predicted slice coordinates $(x_s, y_s)$ are mapped back to full-frame global coordinates $(x_g, y_g) = (x_s + x_{\text{offset}}, y_s + y_{\text{offset}})$.
4. **NMS Post-Processing:** Standard Non-Maximum Suppression merges duplicate detections across overlapping tile boundaries.

---

## 2.5 Multi-Object Tracking (MOT) Paradigms & Mathematical Foundations

While object detection identifies target coordinates within individual frames, **Multi-Object Tracking (MOT)** establishes temporal correspondence across successive video frames, assigning a unique persistent tracking identifier $\text{ID}_k$ to each distinct patient trajectory $\mathcal{T}_k = \{ \mathbf{z}_t \}_{t=1}^T$.

### 2.5.1 Classical Kalman Filtering & Hungarian Association (SORT)
The Simple Online and Realtime Tracking (**SORT**) algorithm (Bewley et al., 2016) established the benchmark for high-speed tracking-by-detection. SORT models target motion in image space using a linear constant-velocity Kalman filter (Kalman, 1960).

The target state vector $\mathbf{x}_t$ is defined as:
$$\mathbf{x}_t = [u, v, s, r, \dot{u}, \dot{v}, \dot{s}]^T$$

Where:
- $(u, v)$: 2D pixel coordinates of the bounding box center.
- $s = w \cdot h$: Bounding box scale (area).
- $r = w / h$: Bounding box aspect ratio (assumed constant).
- $(\dot{u}, \dot{v}, \dot{s})$: Corresponding velocity components.

The discrete Kalman state transition and measurement equations are:
$$\mathbf{x}_t = \mathbf{F} \mathbf{x}_{t-1} + \mathbf{w}_t, \quad \mathbf{w}_t \sim \mathcal{N}(0, \mathbf{Q})$$
$$\mathbf{z}_t = \mathbf{H} \mathbf{x}_t + \mathbf{v}_t, \quad \mathbf{v}_t \sim \mathcal{N}(0, \mathbf{R})$$

Frame-to-frame data association is solved via the **Hungarian Algorithm** (Kuhn, 1955) on an Intersection-over-Union (IoU) distance cost matrix $\mathbf{C}_{\text{IoU}}$:
$$\mathbf{C}_{\text{IoU}}(i, j) = 1 - \text{IoU}(\mathbf{z}_i, \hat{\mathbf{z}}_j)$$

Where $\hat{\mathbf{z}}_j = \mathbf{H} \hat{\mathbf{x}}_j^{-}$ denotes the Kalman filter predicted bounding box measurement.

```
       Detections (t)          Track Predictions (t)
      ┌──────────────┐        ┌───────────────────┐
      │ Detection 1  │        │ Track A (Kalman)  │
      │ Detection 2  │        │ Track B (Kalman)  │
      │ Detection 3  │        │ Track C (Kalman)  │
      └───────┬──────┘        └─────────┬─────────┘
              │                         │
              └────────────┬────────────┘
                           ▼
              ┌────────────────────────┐
              │ IoU Cost Matrix C(i,j) │
              └────────────┬───────────┘
                           ▼
              ┌────────────────────────┐
              │  Hungarian Algorithm   │
              └────────────┬───────────┘
                           ▼
        ┌──────────────────┴──────────────────┐
        ▼                                     ▼
Matched Pairs (Update Kalman)     Unmatched (Spawn / Terminate)
```

**Limitations of SORT in Clinical Waiting Halls:**  
SORT relies solely on spatial overlap and velocity continuity. When patients sit motionless, pass behind other individuals, or are carried in occluded blankets, detection boxes drop below detection thresholds or spatial IoU becomes zero. Consequently, SORT loses track continuity and instantiates a new ID when the target reappears, causing severe cumulative double-counting in daily hospital logs.

### 2.5.2 DeepSORT: Visual Embeddings & Edge Bottlenecks
To overcome the spatial limitations of SORT during occlusion, **DeepSORT** (Wojke et al., 2017) integrated visual appearance embeddings extracted via a deep convolutional Re-Identification (Re-ID) network. The association cost metric combines Mahalanobis spatial distance $d^{(1)}(i, j)$ and cosine visual appearance distance $d^{(2)}(i, j)$:

$$c_{i, j} = \lambda d^{(1)}(i, j) + (1 - \lambda) d^{(2)}(i, j)$$

While DeepSORT successfully recovers trajectories across short occlusions, running a separate deep convolutional feature extraction pass for every cropped bounding box in every frame imposes enormous computational overhead ($>300\%$ increase in CPU execution time), violating the low-latency constraints of resource-constrained edge hardware.

### 2.5.3 ByteTrack: Association with Low-Score Detections
**ByteTrack** (Zhang et al., 2022) revolutionized multi-object tracking by demonstrating that true targets under occlusion are not absent; rather, their detection confidence scores simply drop below standard filtering thresholds (e.g., falling between $0.10$ and $0.45$). 

Instead of discarding low-confidence bounding boxes, ByteTrack introduces a **two-stage bipartite matching algorithm**:
1. **First Association Stage:** High-confidence detections ($\mathcal{D}_{\text{high}}$, score $\ge \tau_{\text{high}}$) are associated with active tracks $\mathcal{T}$ via Hungarian IoU matching.
2. **Second Association Stage:** Remaining unmatched active tracks ($\mathcal{T}_{\text{remain}}$) are matched against low-confidence detections ($\mathcal{D}_{\text{low}}$, $\tau_{\text{low}} \le \text{score} < \tau_{\text{high}}$).

This preserves track continuity during heavy occlusions without requiring expensive deep visual appearance extractors.

```
                  All Detections at Frame t
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   D_high (Score >= 0.45)            D_low (0.10 <= Score < 0.45)
            │                                 │
            ▼                                 │
   Hungarian Matching vs                      │
   Active Tracks (T_active)                   │
            │                                 │
     ┌──────┴──────┐                          │
     ▼             ▼                          ▼
  Matched     Unmatched Tracks ──────► Hungarian Matching vs
  Tracks      (T_remain)               D_low Detections
                                              │
                                       ┌──────┴──────┐
                                       ▼             ▼
                                    Matched      Unmatched
                                    Tracks       (Buffer)
```

### 2.5.4 BoT-SORT: Motion Compensation & State Expansion
**BoT-SORT** (Aharon et al., 2022) addresses the vulnerability of classical Kalman filters to camera ego-motion. In mobile or panning surveillance feeds, camera jitter shifts pixel coordinates, invalidating constant-velocity assumptions. BoT-SORT integrates:
1. **Camera Motion Compensation (CMC):** Uses OpenCV Global Motion Estimation (GME) with affine transformation matrices $\mathbf{M} \in \mathbb{R}^{2 \times 3}$ to warp the prior state covariance and mean into the new frame reference frame:
   $$\mathbf{x}_{t|t-1}' = \mathbf{M} \mathbf{x}_{t|t-1}, \quad \mathbf{P}_{t|t-1}' = \mathbf{M} \mathbf{P}_{t|t-1} \mathbf{M}^T$$
2. **Modified State Vector:** Directly tracks bounding box width $w$ and height $h$ instead of scale and aspect ratio, improving tracking responsiveness on non-rigid pedestrian bounding boxes.

### 2.5.5 OC-SORT: Observation-Centric Momentum Recovery
**OC-SORT** (Cao et al., 2023) resolves error accumulation in Kalman filters during long-term occlusions. When an object is occluded for multiple frames, linear state extrapolation diverges from reality. OC-SORT introduces **Observation-Centric Momentum (OCM)** and **Observation-Centric Online Smoothing (OCOS)**, re-calculating motion vectors retrospectively once an observation is re-acquired:

$$\mathbf{v}_{\text{corrected}} = \frac{\mathbf{z}_{t_{\text{reappear}}} - \mathbf{z}_{t_{\text{lost}}}}{t_{\text{reappear}} - t_{\text{lost}}}$$

This prevents catastrophic track drift during non-linear patient movement in crowded waiting rooms.

### 2.5.6 FastTracker with Parent-Child Spatial Anchoring
In pediatric triage scenarios where infants are carried against an adult's torso, bounding boxes of the adult and infant exhibit sustained, high-IoU geometric overlap ($\text{IoU} \approx 0.30 - 0.65$) while sharing identical spatial velocity vectors:

$$\mathbf{v}_{\text{child}} \approx \mathbf{v}_{\text{parent}}$$

Standard trackers frequently cause **ID swapping** between the parent and child when bounding boxes intersect. **FastTracker** introduces a **Parent-Child Spatial Anchoring Protocol**:
- When a pediatric bounding box $B_{\text{child}}$ is detected within the convex spatial hull of an adult bounding box $B_{\text{parent}}$, the child's track ID is anchored to the parent's trajectory.
- Velocity updates for the infant track incorporate an exponential moving average (EMA) blend with the parent's centroid displacement:
$$\mathbf{v}_{\text{child}}^{(t)} = (1 - \gamma)\mathbf{v}_{\text{child}}^{(t)} + \gamma \mathbf{v}_{\text{parent}}^{(t)}$$
- If the child detection is temporarily lost due to swaddling, the anchor maintains the child track's virtual position at a fixed relative offset $(\Delta x, \Delta y)$ from the parent centroid, preventing track termination and duplicate re-spawning.

---

## 2.6 Illumination Normalization & Contrast Enhancement in Computer Vision

Clinical emergency waiting rooms frequently exhibit suboptimal lighting conditions, ranging from harsh overhead fluorescent glare to dim night-shift ambient illumination. In poorly lit environments, pixel intensity histograms become compressed into narrow low-value ranges, degrading edge gradients and causing convolutional kernels to miss faint infant body contours.

### 2.6.1 Global Histogram Equalization (GHE) vs. CLAHE
Standard Global Histogram Equalization (GHE) flattens the global image histogram by applying a monotonic mapping derived from the Cumulative Distribution Function (CDF):

$$s_k = T(r_k) = (L - 1) \sum_{j=0}^k p_r(r_j) = \frac{L - 1}{N} \sum_{j=0}^k n_j$$

While GHE enhances global contrast, it over-amplifies background sensor noise and washes out subtle clinical features in localized regions.

**Contrast Limited Adaptive Histogram Equalization (CLAHE)** (Pizer et al., 1987; Reza, 2004) resolves this by:
1. Partitioning the image into an $M \times N$ grid of contextual tiles (e.g., $8 \times 8$ pixels).
2. Calculating localized histograms for each tile.
3. Clipping histogram bins that exceed a predefined clip limit $\beta$:

$$N_{\text{clip}} = \frac{N_{\text{pixels}}}{N_{\text{bins}}} \left( 1 + \frac{\beta}{100} (N_{\text{bins}} - 1) \right)$$

4. Redistributing clipped pixels uniformly across all bins prior to CDF calculation.
5. Combining tile boundaries via bilinear interpolation to eliminate artificial block boundary artifacts.

```
       Input Image Tile (8x8)                Histogram Clipping
      ┌─────────────────────────┐           │   |
      │ Local Contrast Gradient │           │ ┌─┴─┐  <- Clipped Area Redistributed
      │ (Low-light patient area)│           │ │   │     Uniformly
      └────────────┬────────────┘           │ │   │
                   ▼                        └─┴───┴──────────
      CLAHE Local Mapping + Bilinear Blend ──► Enhanced Contrast Output
```

### 2.6.2 LAB Color Space vs. RGB Processing
Applying CLAHE directly to independent Red, Green, and Blue (RGB) color channels causes severe chromatic distortion and unnatural color shifts. This thesis implements CLAHE within the **CIE $L^*a^*b^*$ color space**:
- **$L^*$ Channel:** Represents perceptual lightness ($0 \le L^* \le 100$).
- **$a^*$ Channel:** Represents green-to-red chromaticity.
- **$b^*$ Channel:** Represents blue-to-yellow chromaticity.

By isolating and applying CLAHE exclusively to the $L^*$ luminance channel while preserving $a^*$ and $b^*$ chromatic channels unchanged, the system maximizes structural edge contrast without altering the visual appearance or skin tones of pediatric patients.

---

## 2.7 Perspective Geometry & Ground-Plane Scale Invariance

In typical clinical CCTV installations, cameras are mounted on walls or ceilings at an elevated angle $\theta$, pointing downward toward the triage floor. This creates a perspective projection where the physical ground distance $Z$ from the camera correlates non-linearly with vertical pixel position $y$.

```
              Elevated Clinical Camera
                    ▲
                   / \
                  /   \  Line of Sight
                 /  θ  \
                /       \
  Horizon (Y_h)───────────┐ Far Field (Z_far): Small pixel height (h_far)
                          │
                          │
                          │ Near Field (Z_near): Large pixel height (h_near)
  Ground Plane ───────────┴───────────────────────────────► Distance Z
```

Under standard pinhole camera projection, the projected height $h_{\text{img}}$ of a patient with physical height $H_{\text{real}}$ at distance $Z$ is governed by focal length $f$:

$$h_{\text{img}} = \frac{f \cdot H_{\text{real}}}{Z}$$

In uncalibrated surveillance cameras lacking depth sensors, vertical bounding box bottom coordinate $y_{\text{bottom}}$ serves as a robust proxy for ground plane depth $Z$. Establishing a normalized optical perspective gradient allows geometric disambiguation between an upright child standing near the camera and an adult sitting far in the background.

---

## 2.8 Edge Computing vs. Cloud Streaming in Medical Informatics

Deploying artificial intelligence in clinical environments requires strict adherence to privacy regulations (HIPAA in the United States, GDPR in the European Union) and high operational resilience (Shickel et al., 2019).

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    EDGE COMPUTING vs. CLOUD STREAMING                   │
│                                                                         │
│   Dimension            Cloud Streaming              Edge Monolith (Ours)│
│   ───────────────────────────────────────────────────────────────────   │
│   Privacy / HIPAA      High Risk (Raw video over    Zero Risk (Transient│
│                        public internet WAN)         in-memory inference)│
│   Bandwidth Demand     Heavy (5-15 Mbps / camera)   Zero WAN Bandwidth  │
│   Network Latency      Variable (150 - 800 ms)      Zero Lag (< 30 ms)  │
│   Hardware Cost        Low Local / High Cloud OpEx  One-time Low Cost PC│
│   Failure Mode         Internet down = System blind Full Local Autonomy │
└─────────────────────────────────────────────────────────────────────────┘
```

Edge computing, defined as executing deep learning inference directly on local on-premise hardware, ensures that raw patient video streams never leave the local clinical network boundary. This eliminates data leakage vulnerabilities and guarantees continuous operational autonomy even during complete hospital internet outages.

---

## 2.9 Summary & Identification of Research Gaps

The literature establishes powerful individual components: YOLO for single-stage detection, DINOv3 for foundation vision representations, SAHI for high-resolution patch slicing, ByteTrack for low-score tracking, and CLAHE for illumination enhancement. However, critical research gaps remain unaddressed in existing literature:

1. **Lack of Foundation Knowledge Distillation for Clinical Pediatric Occlusion:** Existing pediatric detection models rely exclusively on standard supervised transfer learning from COCO weights. No prior work has explored dense intermediate feature distillation from massive self-supervised Vision Transformers (DINOv3) into lightweight edge convolutional models (YOLO26s) specifically targeting swaddled and carried infants under physical occlusion.
2. **Absence of Real-Time Dynamic Tracker Orchestration:** Existing MOT benchmarks evaluate trackers under static conditions. In clinical triage, sudden camera vibrations or sudden crowd occlusions degrade static tracker performance. No existing framework dynamically assesses live scene motion ($\Delta I$) and occlusion density ($\text{IoU}_{\text{pairwise}}$) to route frames adaptively across specialized tracking algorithms with hysteresis stabilization.
3. **Absence of Spatial Centroid Fallback & Parent-Child Anchoring:** Standard trackers drop tracks when infants are occluded by caregivers' bodies, leading to duplicate ID assignment upon re-emergence. A dedicated spatial centroid fallback and parent-child anchoring mechanism is needed to resolve untracked detections without incurring deep neural Re-ID latency.
4. **Decoupled Architecture Void:** Most medical CV implementations rely on heavy multi-service architectures (FastAPI backends + React/Laravel frontends connected via WebSockets), introducing synchronization lag and dropped frames. A unified, in-memory monolithic paradigm is required to deliver true zero-latency clinical intelligence on edge hardware.

This thesis directly resolves these research gaps through the architecture, algorithms, and empirical evaluations formulated in subsequent chapters.
