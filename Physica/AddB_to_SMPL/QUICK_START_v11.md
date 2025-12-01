# Quick Start Guide: v11_with_physics

## TL;DR

Modified FIX 10 code to extract physics data (GRF, torques, etc.) for the EXACT sampled frames used in optimization.

## Files

- **Main script**: `addbiomechanics_to_smpl_v11_with_physics.py`
- **Documentation**: `README_v11_physics.md`
- **Example usage**: `example_load_physics_data.py`
- **Detailed changes**: `MODIFICATIONS_SUMMARY.md`

## Run It

```bash
python addbiomechanics_to_smpl_v11_with_physics.py \
    --b3d /path/to/subject.b3d \
    --smpl_model /path/to/smpl_model.pkl \
    --out_dir ./outputs \
    --num_frames 200 \
    --lower_body_only
```

## What You Get

### Standard Outputs (from v10)
- `smpl_params.npz` - SMPL parameters
- `pred_joints.npy` - Predicted joints [T, 24, 3]
- `target_joints.npy` - Target joints [T, N, 3]
- `meta.json` - Metadata

### NEW Physics Outputs (v11)
- `ground_reaction_forces.npy` - [T, 6] forces + torques
- `joint_torques.npy` - [T, N_dofs]
- `joint_velocities.npy` - [T, N_dofs]
- `joint_accelerations.npy` - [T, N_dofs]
- `com_position.npy` - [T, 3]
- `com_velocity.npy` - [T, 3]
- `com_acceleration.npy` - [T, 3]
- `residual_forces.npy` - [T, 6]
- `sampled_frame_indices.npy` - [T]

**All arrays have same T (number of sampled frames)!**

## Load and Visualize

```bash
python example_load_physics_data.py \
    --output_dir ./outputs/v11_with_physics_SUBJECT_NAME \
    --save_plot physics_viz.png
```

## Load in Python

```python
import numpy as np

# Load physics data
grf = np.load('output/run_name/ground_reaction_forces.npy')  # [T, 6]
torques = np.load('output/run_name/joint_torques.npy')  # [T, N_dofs]
indices = np.load('output/run_name/sampled_frame_indices.npy')  # [T]

# Load motion data
target = np.load('output/run_name/target_joints.npy')  # [T, N, 3]
pred = np.load('output/run_name/pred_joints.npy')  # [T, 24, 3]

print(f"Frames: {len(indices)}")
print(f"GRF shape: {grf.shape}")
print(f"Torques shape: {torques.shape}")
```

## Key Features

1. **Exact Frame Alignment**: Physics data extracted for SAME frames as motion
2. **Complete Physics**: GRF, torques, velocities, accelerations, CoM, residuals
3. **Memory Efficient**: Only extracts data for sampled frames
4. **Well Documented**: Full metadata in meta.json

## What Changed from v10?

1. Added `extract_physics_for_sampled_frames()` function
2. Extract physics after optimization (step 5/6)
3. Save 9 additional .npy files with physics data
4. Update meta.json with physics metadata
5. Change output prefix to "v11_with_physics_"

## Physics Data Explained

| File | Shape | Description |
|------|-------|-------------|
| `ground_reaction_forces.npy` | [T, 6] | Forces (N) + torques (Nm) from ground |
| `joint_torques.npy` | [T, N_dofs] | Torques at each joint (Nm) |
| `joint_velocities.npy` | [T, N_dofs] | Joint velocities (m/s or rad/s) |
| `joint_accelerations.npy` | [T, N_dofs] | Joint accelerations (m/s² or rad/s²) |
| `com_position.npy` | [T, 3] | Center of mass position (m) |
| `com_velocity.npy` | [T, 3] | Center of mass velocity (m/s) |
| `com_acceleration.npy` | [T, 3] | Center of mass acceleration (m/s²) |
| `residual_forces.npy` | [T, 6] | Residual forces/moments at root (lower = better) |
| `sampled_frame_indices.npy` | [T] | Which original frames were sampled |

## Typical Values

- **GRF**: 0-2000 N for walking, higher for running
- **Joint torques**: 0-200 Nm (hip/knee largest)
- **CoM height**: ~0.9-1.0 m for average person
- **CoM velocity**: 0.5-2.0 m/s for walking
- **Residuals**: < 50 N force, < 50 Nm moment (good physics)

## Use Cases

1. **Physics-informed training**: Train models with physics losses
2. **Motion quality**: Validate if predicted motions are physically plausible
3. **Biomechanics analysis**: Study force patterns, joint loading
4. **Controller training**: Use as GT for RL controllers
5. **Data augmentation**: Physics-based motion synthesis

## Troubleshooting

**Q: Array shapes don't match**
- Check they all have same T (number of sampled frames)
- Verify sampled_frame_indices.npy matches frame sampling

**Q: GRF values seem wrong**
- Check units (should be in Newtons)
- Verify processing_pass setting (default=0)
- GRF can be zero when not in contact with ground

**Q: Residuals very large (> 100N)**
- May indicate poor physics solve in original .b3d file
- Try different processing_pass
- Check .b3d file quality

**Q: Script crashes on physics extraction**
- Ensure nimblephysics is installed
- Check .b3d file is valid (can load with nimble)
- Verify processing_pass index is valid

## Need More Info?

- **Full details**: See `README_v11_physics.md`
- **Implementation**: See `MODIFICATIONS_SUMMARY.md`
- **Example code**: See `example_load_physics_data.py`
