---
neuron_id: clinical_queue_motion_intuition
title: Clinical Queue Motion Intuition and Frame Sampling
synaptic_weight: 45
corpus_callosum: clinical_real_time_latency_derivation
blindspot: false
summary: Physical intuition for why slow hospital outpatient movement permits low frame rates without track loss.
---

# Clinical Queue Motion Intuition and Frame Sampling

## Physical Dynamics
In pediatric outpatient waiting areas, patient flow is sluggish:
- Caregivers walk slowly with children ($v < 1.2\text{ m/s}$).
- Individuals remain stationary on wooden triage benches for tens of minutes.

## Tracking Invariance
Because displacement between sampled frames is small relative to bounding box scale:
$$\Delta x = v \cdot \Delta t \ll W_{\text{box}}$$
At $6.8\text{--}9.1\text{ FPS}$ ($\Delta t \approx 110\text{--}147\text{ ms}$), Kalman filter covariance envelopes readily overlap adjacent detections. High frame rate (30 FPS) provides redundant spatial telemetry while choking edge hardware.
