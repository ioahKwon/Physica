# v11_with_physics - AddBiomechanics to SMPL with Physics Data Extraction

## Overview

This version (v11_with_physics) extends the FIX 10 code (`addbiomechanics_to_smpl_v10_foot_orient_late.py`) to extract and save AddBiomechanics physics ground truth data for the sampled frames used during optimization.

## Key Features

### New in v11: Physics Data Extraction

The code now extracts the following physics quantities from the .b3d file for the **EXACT SAME** sampled frames used for SMPL optimization:

1. **Ground Reaction Forces (GRF)**: `[T, 6]` array
   - First 3 columns: force components (fx, fy, fz) in Newtons
   - Last 3 columns: torque/moment components (tx, ty, tz) in Newton-meters
   - Sum of all contact body forces and torques

2. **Joint Torques**: `[T, N_dofs]` array
   - Torques applied at each degree of freedom
   - In Newton-meters for rotational DOFs

3. **Joint Velocities**: `[T, N_dofs]` array
   - Velocities of each degree of freedom
   - In m/s for translational DOFs, rad/s for rotational DOFs

4. **Joint Accelerations**: `[T, N_dofs]` array
   - Accelerations of each degree of freedom
   - In m/s² for translational DOFs, rad/s² for rotational DOFs

5. **Center of Mass (CoM) Data**: `[T, 3]` arrays
   - Position, velocity, and acceleration of the body's center of mass
   - In meters, m/s, and m/s² respectively

6. **Residual Forces**: `[T, 6]` array
   - Residual forces and moments at the root (pelvis)
   - Indicates how well the dynamics are balanced
   - Lower residuals = better physics consistency

7. **Sampled Frame Indices**: `[T]` array
   - Original frame indices from the .b3d file that were sampled
   - Essential for aligning physics data with motion data

## Usage

The usage is identical to v10, but with additional physics data outputs:

```bash
python addbiomechanics_to_smpl_v11_with_physics.py \
    --b3d /path/to/subject.b3d \
    --smpl_model /path/to/smpl_model.pkl \
    --out_dir /path/to/output \
    --num_frames 200 \
    --lower_body_only
```

## Output Files

In addition to the standard v10 outputs, v11 generates:

### Physics Data Files (numpy arrays)
- `ground_reaction_forces.npy`: [T, 6] GRF (force + torque)
- `joint_torques.npy`: [T, N_dofs] joint torques
- `joint_velocities.npy`: [T, N_dofs] joint velocities
- `joint_accelerations.npy`: [T, N_dofs] joint accelerations
- `com_position.npy`: [T, 3] center of mass position
- `com_velocity.npy`: [T, 3] center of mass velocity
- `com_acceleration.npy`: [T, 3] center of mass acceleration
- `residual_forces.npy`: [T, 6] root residual forces/moments
- `sampled_frame_indices.npy`: [T] original frame indices

### Metadata
The `meta.json` file now includes a `physics_data` section with:
- Number of DOFs
- Shape information for all physics arrays
- Detailed descriptions of each physics quantity

## Frame Sampling Strategy

When `--num_frames` or `--max_frames_optimize` is specified:

1. **Frame Sampling**: The code uses `adaptive_frame_sampling()` to uniformly sample frames
   - If total frames ≤ max_frames: use all frames
   - If total frames > max_frames: sample every N-th frame where N = ceil(T / max_frames)
   - Always includes first and last frame

2. **Physics Extraction**: The `extract_physics_for_sampled_frames()` function:
   - Reads the .b3d file once
   - Extracts physics data ONLY for the sampled frame indices
   - Ensures perfect alignment between motion and physics data

3. **Data Alignment**:
   - All output arrays have the same first dimension T (number of sampled frames)
   - Use `sampled_frame_indices.npy` to map back to original .b3d frame numbers

## Implementation Details

### Physics Extraction Function

The core function `extract_physics_for_sampled_frames()` (lines 270-408):

```python
def extract_physics_for_sampled_frames(
    b3d_path: str,
    sampled_frame_indices: np.ndarray,
    trial: int = 0,
    processing_pass: int = 0,
    start: int = 0
) -> dict:
    """
    Extract physics data from .b3d file for the EXACT sampled frames.

    Returns:
        dict with physics quantities aligned to sampled frames
    """
```

### Data Sources

Physics data is extracted from the `.b3d` file's `processingPasses`:

- **processingPass[0]**: KINEMATICS - kinematic reconstruction only
- **processingPass[1]**: LOW_PASS_FILTER - filtered kinematics
- **processingPass[2]**: DYNAMICS - full dynamics solution (default)

The code uses `processing_pass=0` by default (same as joint positions), but can be configured.

### Nimblephysics API Usage

The code accesses physics data through the nimblephysics API:

```python
pp = frame.processingPasses[processing_pass]

# Ground reaction forces (per contact body)
grf_raw = np.array(pp.groundContactForce, dtype=np.float32)

# Joint torques
tau = np.array(pp.tau, dtype=np.float32)

# Joint velocities and accelerations
vel = np.array(pp.vel, dtype=np.float32)
acc = np.array(pp.acc, dtype=np.float32)

# Center of mass
com_pos = np.array(pp.comPos, dtype=np.float32)
com_vel = np.array(pp.comVel, dtype=np.float32)
com_acc = np.array(pp.comAcc, dtype=np.float32)

# Residuals
residuals = np.array(pp.residualWrenchInRootFrame, dtype=np.float32)
```

## Use Cases for Physics Data

The extracted physics ground truth data can be used for:

1. **Physics-Informed Training**: Train models to predict physically plausible motions
2. **Physics Loss Functions**: Add physics-based losses (GRF, torque consistency)
3. **Motion Quality Assessment**: Evaluate how well predicted motions satisfy physics
4. **Controller Development**: Train RL controllers with physics supervision
5. **Biomechanics Analysis**: Study force patterns, joint loading, etc.

## Differences from v10

1. **New Function**: `extract_physics_for_sampled_frames()` (lines 270-408)
2. **Updated Main Flow**: Now 6 steps instead of 5 (added physics extraction)
3. **Additional Outputs**: 9 new .npy files with physics data
4. **Extended Metadata**: `meta.json` includes physics_data section
5. **Run Name Prefix**: Changed from `v10_foot_orient_late_` to `v11_with_physics_`

## Example: Loading Physics Data

```python
import numpy as np

# Load physics data
grf = np.load('output/run_name/ground_reaction_forces.npy')  # [T, 6]
torques = np.load('output/run_name/joint_torques.npy')  # [T, N_dofs]
frame_indices = np.load('output/run_name/sampled_frame_indices.npy')  # [T]

# Load corresponding motion data
target_joints = np.load('output/run_name/target_joints.npy')  # [T, N_joints, 3]
pred_joints = np.load('output/run_name/pred_joints.npy')  # [T, N_joints, 3]

print(f"Frames: {len(frame_indices)}")
print(f"DOFs: {torques.shape[1]}")
print(f"Mean GRF magnitude: {np.linalg.norm(grf[:, :3], axis=1).mean():.2f} N")
print(f"Peak GRF: {np.linalg.norm(grf[:, :3], axis=1).max():.2f} N")
```

## Validation

To verify the physics data extraction is working correctly:

1. **Check shapes**: All arrays should have same T (number of sampled frames)
2. **Check GRF magnitudes**: Should be in reasonable range (typically 0-2000N for walking)
3. **Check residuals**: Should be relatively small (< 50N force, < 50Nm moment)
4. **Check frame alignment**: `sampled_frame_indices` should match frame sampling strategy

## Performance Notes

- Physics extraction adds minimal overhead (~1-2 seconds for typical sequences)
- Data is extracted efficiently in a single pass through the .b3d file
- Only sampled frames are extracted, reducing memory and storage requirements

## Future Enhancements

Potential improvements for future versions:

1. Add contact body-specific GRF (not just summed)
2. Include contact point locations and normals
3. Add joint-specific power calculations (torque × velocity)
4. Include muscle activation data if available
5. Add option to extract physics for ALL frames (not just sampled)

## Questions?

For questions or issues, refer to:
- Original v10 documentation for SMPL fitting details
- Nimblephysics documentation for .b3d file format
- AddBiomechanics paper for physics processing details
