# Physica

SMPL body model fitting to AddBiomechanics motion capture data with improvements from SMPL2AddBiomechanics paper.

## Features

- **Baseline SMPL Fitter (v2)**: Optimized SMPL fitting with proper centering and regularization
- **3 Priority Improvements**:
  1. Anthropometric height constraints
  2. Enhanced bone-segment weights (24 granular values)
  3. Per-joint learnable offsets

## Results (AB10_split0)

### Batch Optimization
- Baseline: 184.76 mm
- Per-Joint Offsets: 108.26 mm (41% improvement)

### Frame-by-Frame Optimization (v2-exact)
- Baseline: 217.06 mm
- Enhanced Bone Weights: 208.53 mm (4% improvement)
- Per-Joint Offsets: 134.89 mm (38% improvement)

## Files

- `addbiomechanics_to_smpl_v2.py`: Baseline SMPL fitter
- `test_improvements_simple.py`: Test all 3 improvements together
- `test_individual_improvements.py`: Test each improvement individually (batch)
- `test_individual_v2_exact.py`: Test each improvement individually (frame-by-frame)
- `visualize_comparison.py`: Visualize and compare results

## Author

Joonwoo Kwon (kwonjoon@msu.edu)
