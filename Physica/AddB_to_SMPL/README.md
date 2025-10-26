# AddBiomechanics to SMPL Fitting

Advanced SMPL parameter fitting for AddBiomechanics motion capture data with enhanced multi-feature optimization.

## Overview

This repository implements a state-of-the-art method for fitting SMPL body models to AddBiomechanics (.b3d) motion capture data. The v3_enhanced version achieves **39.82mm average MPJPE**, representing a **26% improvement** over the baseline approach.

## Performance

| Version | Average MPJPE | Improvement |
|---------|--------------|-------------|
| Baseline | 54.00mm | - |
| v2_bone_direction | 42.43mm | 21.4% |
| **v3_enhanced** | **39.82mm** | **26.3%** |

### Per-Subject Results (v3_enhanced)

| Subject | MPJPE | PCK@0.05m | PCK@0.10m |
|---------|-------|-----------|-----------|
| Subject7 | 39.20mm | 66.4% | 88.1% |
| Subject11 | 41.70mm | 64.7% | 86.6% |
| Subject19 | 44.29mm | 61.2% | 84.2% |
| Subject29 | 34.10mm | 71.1% | 90.0% |

## Key Features

### v3_enhanced (Recommended)

The v3_enhanced version implements 4 critical enhancements:

1. **Per-Joint Learning Rate**
   - Foot joints: 0.005 (stability-focused)
   - Knee joints: 0.01 (balanced)
   - Hip joints: 0.015 (higher DOF)
   - Gradient scaling for optimal convergence

2. **Bone Length Soft Constraint**
   - Weight: 0.1
   - Maintains temporal consistency of bone lengths
   - Prevents unrealistic skeletal deformations

3. **Multi-Stage Bone Direction Weights**
   - Early stage (0-33%): Focus on stable bones (femur, tibia) with 1.5× weight
   - Mid stage (33-66%): Balanced weights across all bones
   - Late stage (66-100%): Focus on feet (1.5×) for ground contact

4. **Contact-Aware Optimization**
   - 2.0× weight multiplier when feet contact ground
   - Automatic ground level detection (5th percentile)
   - Contact threshold: 2cm

### Four-Stage Optimization Pipeline

```
Stage 1: Initial Pose Estimation
  └─> Coarse pose with default shape
  
Stage 2: Pose-Aware Shape Optimization
  └─> Mini-batch SGD with pose consistency
  
Stage 3: Pose Refinement
  └─> Per-joint learning rates + multi-stage bone weights
  
Stage 4: Sequence-Level Enhancement
  └─> Bone length consistency + contact-aware optimization
```

## Installation

```bash
# Clone repository
git clone https://github.com/ioahKwon/addbiomechanics-to-smpl.git
cd addbiomechanics-to-smpl

# Install dependencies
pip install torch numpy nimblephysics matplotlib
```

## Usage

### Basic Usage

```bash
python addbiomechanics_to_smpl_v3_enhanced.py \
  --b3d /path/to/subject.b3d \
  --smpl_model /path/to/smpl_model.pkl \
  --out_dir ./results \
  --num_frames 64 \
  --device cpu
```

### Visualization

```bash
python visualize_single.py \
  --pred_joints results/subject/pred_joints.npy \
  --target_joints results/subject/target_joints.npy \
  --output results/subject/visualization.mp4 \
  --lower_body_only \
  --fps 30
```

## File Structure

```
addbiomechanics-to-smpl/
├── addbiomechanics_to_smpl_v3_enhanced.py  # Main implementation (RECOMMENDED)
├── addbiomechanics_to_smpl_v2_bone_direction.py  # Baseline with bone direction
├── visualize_single.py                     # Video visualization
├── models/
│   └── smpl_model.py                       # SMPL model implementation
└── README.md
```

## Method Details

### Loss Functions

The optimization combines multiple loss terms:

```python
total_loss = position_loss 
           + bone_direction_weight * bone_direction_loss
           + bone_length_soft_weight * bone_length_loss
           + pose_reg_weight * pose_regularization
           + trans_reg_weight * translation_regularization
           + temporal_smooth_weight * temporal_smoothness
```

### Bone Direction Loss

Matches bone direction vectors using cosine similarity to bypass structural mismatches between AddBiomechanics and SMPL skeletons:

```python
bone_direction_loss = 1 - cos_similarity(pred_dir, target_dir)
```

This approach is particularly effective for hip-knee mismatches (AddB: 138mm, SMPL: 106mm).

## Results & Visualizations

Sample visualizations are available in the `results_v3_enhanced/` directory for all test subjects.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{kwon2024addbiomechanics_smpl,
  author = {Kwon, Joonwoo},
  title = {Enhanced SMPL Fitting for AddBiomechanics Data},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/ioahKwon/addbiomechanics-to-smpl}
}
```

## License

MIT License - see LICENSE file for details

## Author

**Joonwoo Kwon** ([@ioahKwon](https://github.com/ioahKwon))

## Acknowledgments

- SMPL model: [Body Models](https://smpl.is.tue.mpg.de/)
- AddBiomechanics: [AddBiomechanics Project](https://addbiomechanics.org/)
- Nimble Physics: [nimblephysics](https://github.com/keenon/nimblephysics)
