# 3D-imaging-pipeline

> Large-scale 3D reconstruction using structured light and binary fringe coding — sub-millimeter resolution with consumer-grade hardware.

![Python](https://img.shields.io/badge/Python-3.10-blue) ![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green) ![Resolution](https://img.shields.io/badge/Resolution-1mm-brightgreen) ![License](https://img.shields.io/badge/license-MIT-orange)

---

## Overview

This project implements a full 3D imaging pipeline for large objects (1–4 m²) using structured light projection and binary fringe coding. The system uses a standard video projector and webcam to achieve sub-millimeter spatial resolution — targeting industrial applications in aeronautics and naval engineering.

**IMT Atlantique · UE Pronto 2026 · Team project**  
*Contributors: Boulard G., Cordier L., Julliere T., Ramsis Z., Rannou M.*

---

## How it works

The system projects a sequence of N binary fringe patterns onto an object. Each camera pixel receives a binary code (white/black sequence) across N frames — uniquely identifying the projector stripe it corresponds to. Combined with calibrated projection matrices, this enables full 3D reconstruction via triangulation.

```
Projector → Binary fringe sequence (N=10 frames)
    ↓
Object surface → Reflected fringe images
    ↓
Camera → Captured image tensor (N × 1920 × 1080)
    ↓
Binary decoding → Localization matrix LC
    ↓
Triangulation → 3D point cloud
```

With N=10 binary projections → 2¹⁰ = 1024 distinct positions → 1mm lateral resolution on a 1m object.

---

## Key specifications

| Parameter | Value |
|---|---|
| Target object size | 1 – 4 m² |
| Lateral resolution | 1 mm |
| Binary frames required | N = 10 |
| Projector | EPSON EB-W49 (1280×800) |
| Camera | Logitech C920E (1920×1080) |
| Projector-camera baseline | L ≈ 950 mm |
| Output format | 3D point cloud + optional mesh |

---

## Pipeline modules

### 1. Calibration
Estimates normalized projection matrices for both emitter (MEN) and receiver (MRN) using a checkerboard target.

- `Mire_damier.py` — generates the calibration checkerboard pattern
- `Calib_recepteur.py` — calibrates the camera from known 3D reference points
- `Calib_emetteur.py` — calibrates the projector via indirect method using already-calibrated camera

### 2. Forward problem (simulation)
- `Trames_binaires.py` — generates N binary fringe frames (PNG, lossless)
- `Objet.py` — models the target object surface
- `franges_objet.py` — simulates fringe projection onto object
- `franges_recepteur.py` — simulates camera capture

### 3. Inverse problem (reconstruction)
- `Local_cotes_franges.py` — extracts fringe edge positions
- `Local_franges.py` — localizes fringes and maps to 3D coordinates
- `Coord3D_objet.py` — generates final 3D point cloud via triangulation
- `Comparateur.py` — computes reconstruction error vs ground truth

### 4. Signal processing
- Adaptive thresholding (Otsu method) for robust binarization under varying reflectance
- Noise filtering on raw camera images
- Thermal hysteresis correction on multi-frame acquisitions

---

## Repository structure

```
3D-imaging-pipeline/
├── calibration/        # Calibration scripts and checkerboard generator
├── forward/            # Simulation scripts (binary frames, object modeling)
├── reconstruction/     # Inverse problem: fringe localization, 3D triangulation
├── data/               # Acquired images and calibration parameters (.txt)
├── figures/            # Output point clouds and reconstruction plots
└── README.md
```

---

## Tech stack

- Python 3.10
- OpenCV (camera acquisition, image processing)
- NumPy (tensor operations, matrix algebra)
- Matplotlib (3D point cloud visualization)

---

## Results

The system achieves sub-millimeter spatial resolution on large-scale objects using only consumer-grade hardware (projector + webcam). Validated on a scaled pyramid model with automated calibration and full 3D export.

---

## Author

**Zainab Ramsis** · Engineering Student · IMT Atlantique  
UE Pronto 2026 · zainab.ramsis@imt-atlantique.net
