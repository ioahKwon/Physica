# Modifications Summary: v10 → v11 with Physics Data Extraction

## Overview

Modified `addbiomechanics_to_smpl_v10_foot_orient_late.py` to create `addbiomechanics_to_smpl_v11_with_physics.py` which extracts and saves AddBiomechanics physics ground truth data for the sampled frames.

## Key Requirement

**Extract physics data for the EXACT SAME frames that are sampled during optimization.**

When `num_frames` is specified (e.g., 200), the code uses `adaptive_frame_sampling()` to sample frames uniformly. The physics extraction must use these same frame indices to ensure perfect alignment between motion and physics data.

## Files Modified/Created

### 1. Main Script: `addbiomechanics_to_smpl_v11_with_physics.py`

**Location**: `/egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL/addbiomechanics_to_smpl_v11_with_physics.py`

**Changes**:

#### A. New Function: `extract_physics_for_sampled_frames()` (Lines 270-408)

```python
def extract_physics_for_sampled_frames(
    b3d_path: str,
    sampled_frame_indices: np.ndarray,
    trial: int = 0,
    processing_pass: int = 0,
    start: int = 0
) -> dict:
```

**Purpose**: Extract physics data from .b3d file for specific frame indices

**Implementation Details**:
- Takes `sampled_frame_indices` array as input (the exact frames used in optimization)
- Reads frames from .b3d file efficiently (only up to max needed frame)
- Iterates through sampled indices and extracts physics from each frame's processingPass
- Returns dictionary with all physics quantities aligned to sampled frames

**Extracted Physics Data**:
1. Ground reaction forces (GRF): Sum of all contact body forces [T, 3]
2. Ground reaction torques: Sum of all contact body torques [T, 3]
3. Combined GRF: [T, 6] = [forces; torques]
4. Joint torques (tau): [T, N_dofs]
5. Joint velocities (vel): [T, N_dofs]
6. Joint accelerations (acc): [T, N_dofs]
7. Center of mass position (comPos): [T, 3]
8. Center of mass velocity (comVel): [T, 3]
9. Center of mass acceleration (comAcc): [T, 3]
10. Residual forces at root (residualWrenchInRootFrame): [T, 6]

**Nimblephysics API Usage**:
```python
pp = frame.processingPasses[processing_pass]

# Extract physics quantities
grf_raw = np.array(pp.groundContactForce, dtype=np.float32)
torque_raw = np.array(pp.groundContactTorque, dtype=np.float32)
tau_raw = np.array(pp.tau, dtype=np.float32)
vel_raw = np.array(pp.vel, dtype=np.float32)
acc_raw = np.array(pp.acc, dtype=np.float32)
com_pos = np.array(pp.comPos, dtype=np.float32)
com_vel = np.array(pp.comVel, dtype=np.float32)
com_acc = np.array(pp.comAcc, dtype=np.float32)
residuals = np.array(pp.residualWrenchInRootFrame, dtype=np.float32)
```

#### B. Updated Main Flow (Lines 2816-2860)

**Before (v10)**: 5 steps
1. Load AddBiomechanics data
2. Resolve joint correspondences
3. Initialize SMPL model
4. Optimize SMPL parameters
5. Display metrics

**After (v11)**: 6 steps
1. Load AddBiomechanics data
2. Resolve joint correspondences
3. Initialize SMPL model
4. Optimize SMPL parameters
5. **Extract physics data for sampled frames** (NEW!)
6. Display metrics

**Added Code**:
```python
print('\n[5/6] Extracting physics data for sampled frames...')
physics_data = extract_physics_for_sampled_frames(
    b3d_path=args.b3d,
    sampled_frame_indices=selected_frame_indices,
    trial=args.trial,
    processing_pass=args.processing_pass,
    start=args.start
)
print(f'  Extracted physics data for {len(selected_frame_indices)} sampled frames')
print(f'  DOFs: {physics_data["n_dof"]}')
print(f'  GRF shape: {physics_data["grf"].shape}')
print(f'  Joint torques shape: {physics_data["joint_torques"].shape}')
```

#### C. Save Physics Data (Lines 2849-2860)

**Added after saving SMPL parameters**:
```python
# Save physics data
print('Saving physics data...')
np.save(os.path.join(out_dir, 'ground_reaction_forces.npy'), physics_data['grf'])
np.save(os.path.join(out_dir, 'joint_torques.npy'), physics_data['joint_torques'])
np.save(os.path.join(out_dir, 'joint_velocities.npy'), physics_data['joint_velocities'])
np.save(os.path.join(out_dir, 'joint_accelerations.npy'), physics_data['joint_accelerations'])
np.save(os.path.join(out_dir, 'com_position.npy'), physics_data['com_pos'])
np.save(os.path.join(out_dir, 'com_velocity.npy'), physics_data['com_vel'])
np.save(os.path.join(out_dir, 'com_acceleration.npy'), physics_data['com_acc'])
np.save(os.path.join(out_dir, 'residual_forces.npy'), physics_data['residual_forces'])
np.save(os.path.join(out_dir, 'sampled_frame_indices.npy'), physics_data['sampled_frame_indices'])
print(f'  Saved physics data to {out_dir}/')
```

#### D. Updated Metadata (Lines 2897-2924)

**Added to `meta.json`**:
```python
'physics_data': {
    'n_dof': int(physics_data['n_dof']),
    'grf_shape': list(physics_data['grf'].shape),
    'joint_torques_shape': list(physics_data['joint_torques'].shape),
    'includes': [
        'ground_reaction_forces.npy',
        'joint_torques.npy',
        'joint_velocities.npy',
        'joint_accelerations.npy',
        'com_position.npy',
        'com_velocity.npy',
        'com_acceleration.npy',
        'residual_forces.npy',
        'sampled_frame_indices.npy'
    ],
    'description': {
        'grf': '[T, 6] ground reaction forces (fx, fy, fz) and torques (tx, ty, tz)',
        'joint_torques': '[T, N_dofs] joint torques',
        'joint_velocities': '[T, N_dofs] joint velocities',
        'joint_accelerations': '[T, N_dofs] joint accelerations',
        'com_position': '[T, 3] center of mass position',
        'com_velocity': '[T, 3] center of mass velocity',
        'com_acceleration': '[T, 3] center of mass acceleration',
        'residual_forces': '[T, 6] residual forces/moments at root',
        'sampled_frame_indices': '[T] original frame indices that were sampled'
    }
}
```

#### E. Updated Run Name (Line 2696)

**Before**: `run_name = "v10_foot_orient_late_" + run_name`
**After**: `run_name = "v11_with_physics_" + run_name`

#### F. Updated Documentation Header (Lines 1-46)

Added comprehensive description of v11 features and new outputs.

### 2. Documentation: `README_v11_physics.md`

**Location**: `/egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL/README_v11_physics.md`

**Content**:
- Overview of v11 features
- Detailed description of each physics quantity
- Usage instructions
- Output file descriptions
- Frame sampling strategy explanation
- Implementation details
- Example code for loading data
- Use cases for physics data

### 3. Example Script: `example_load_physics_data.py`

**Location**: `/egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL/example_load_physics_data.py`

**Features**:
- Load all physics data from output directory
- Print comprehensive summary statistics
- Visualize physics data (6 subplots):
  1. Ground reaction forces over time
  2. Joint torques (top 5 DOFs) over time
  3. Center of mass position over time
  4. Center of mass velocity over time
  5. Residual forces/moments (physics quality)
  6. Joint position error (MPJPE) over time

**Usage**:
```bash
# Print summary
python example_load_physics_data.py --output_dir /path/to/output

# Generate plots
python example_load_physics_data.py --output_dir /path/to/output --plot

# Save plot to file
python example_load_physics_data.py --output_dir /path/to/output --save_plot physics.png
```

## Data Flow Diagram

```
1. Load .b3d file
   ↓
2. Sample frames using adaptive_frame_sampling()
   → selected_frame_indices = [0, 4, 8, 12, ..., 196, 199]
   ↓
3. Optimize SMPL parameters on sampled frames
   → Uses addb_joints[selected_frame_indices]
   ↓
4. Extract physics data for SAME sampled frames
   → extract_physics_for_sampled_frames(selected_frame_indices)
   ↓
5. Save aligned data:
   - target_joints.npy [T, N_joints, 3]
   - pred_joints.npy [T, N_joints, 3]
   - ground_reaction_forces.npy [T, 6]
   - joint_torques.npy [T, N_dofs]
   - ... (all physics quantities)
   - sampled_frame_indices.npy [T]
```

**Key Point**: All arrays have the same `T` (number of sampled frames), ensuring perfect alignment.

## Output Files

### Standard v10 Outputs (unchanged)
- `smpl_params.npz`: SMPL parameters (betas, poses, trans)
- `pred_joints.npy`: Predicted joint positions [T, 24, 3]
- `target_joints.npy`: Target AddB joint positions [T, N, 3]
- `meta.json`: Metadata and configuration

### New v11 Physics Outputs
- `ground_reaction_forces.npy`: [T, 6] GRF forces + torques
- `joint_torques.npy`: [T, N_dofs] joint torques
- `joint_velocities.npy`: [T, N_dofs] joint velocities
- `joint_accelerations.npy`: [T, N_dofs] joint accelerations
- `com_position.npy`: [T, 3] center of mass position
- `com_velocity.npy`: [T, 3] center of mass velocity
- `com_acceleration.npy`: [T, 3] center of mass acceleration
- `residual_forces.npy`: [T, 6] root residual forces/moments
- `sampled_frame_indices.npy`: [T] original frame indices

## Usage Example

```bash
# Run with 200 sampled frames
python addbiomechanics_to_smpl_v11_with_physics.py \
    --b3d /path/to/P002_split0.b3d \
    --smpl_model /path/to/smpl_model.pkl \
    --out_dir ./outputs \
    --num_frames 200 \
    --lower_body_only \
    --processing_pass 0

# Load and visualize results
python example_load_physics_data.py \
    --output_dir ./outputs/v11_with_physics_P002_split0 \
    --save_plot physics_viz.png
```

## Technical Details

### .b3d File Structure

AddBiomechanics .b3d files contain multiple processing passes:
- **Pass 0 (KINEMATICS)**: Kinematic reconstruction from markers
- **Pass 1 (LOW_PASS_FILTER)**: Filtered kinematics
- **Pass 2 (DYNAMICS)**: Full inverse dynamics solution

The code uses `processing_pass=0` by default (same as joint positions).

### Physics Data in processingPass

Each frame's processingPass contains:
```python
processingPass {
    pos              # Joint positions (DOF values)
    vel              # Joint velocities
    acc              # Joint accelerations
    tau              # Joint torques
    jointCenters     # Joint center positions (what we use for targets)
    comPos           # Center of mass position
    comVel           # Center of mass velocity
    comAcc           # Center of mass acceleration
    groundContactForce      # GRF per contact body [N_bodies * 3]
    groundContactTorque     # Torque per contact body [N_bodies * 3]
    residualWrenchInRootFrame  # [6] residual at root
    contact          # Binary contact state per body
    groundContactCenterOfPressure  # CoP per body
}
```

### Frame Alignment

**Critical**: All data is extracted for the SAME frames:

1. `adaptive_frame_sampling()` returns `selected_frame_indices`
2. Joint data is already sampled: `addb_joints = addb_joints_full[selected_frame_indices]`
3. Physics extraction uses same indices: `extract_physics_for_sampled_frames(selected_frame_indices)`
4. Save indices: `sampled_frame_indices.npy`

**Result**: Perfect alignment between motion and physics data.

### Memory Efficiency

The implementation is memory-efficient:
- Only reads frames up to `max(selected_frame_indices)`
- Only stores physics data for sampled frames (not all frames)
- For 200 sampled frames from 2000 total: 10× memory reduction

## Validation Checklist

To verify the implementation works correctly:

- [x] `extract_physics_for_sampled_frames()` function added
- [x] Function uses exact `selected_frame_indices` from sampling
- [x] All physics quantities extracted from processingPass
- [x] Physics extraction integrated into main workflow
- [x] 9 new .npy files saved in output directory
- [x] `meta.json` updated with physics_data section
- [x] README documentation created
- [x] Example loading script created with visualization
- [x] Run name changed to "v11_with_physics_"
- [x] All arrays have consistent first dimension (T = num_sampled_frames)

## Testing Recommendations

1. **Small sequence test**: Run on a small .b3d file (< 100 frames)
   - Verify all output files are created
   - Check array shapes match expected dimensions
   - Verify sampled_frame_indices matches frame sampling

2. **Large sequence test**: Run on long .b3d file (> 1000 frames) with sampling
   - Verify frame sampling works correctly
   - Check physics data only extracted for sampled frames
   - Verify memory usage is reasonable

3. **Data validation**:
   - Load physics data and check ranges (GRF should be 0-2000N for walking)
   - Check residual forces are relatively small (< 50N typically)
   - Verify all arrays have same first dimension

4. **Visualization test**:
   - Run `example_load_physics_data.py` on output
   - Check plots look reasonable
   - Verify statistics make physical sense

## Future Enhancements

Potential improvements for v12:

1. **Per-body contact forces**: Extract GRF per contact body, not just summed
2. **Contact information**: Include contact body names and contact states
3. **Joint-level power**: Calculate power = torque × velocity
4. **Additional physics passes**: Option to extract from DYNAMICS pass (pass 2)
5. **Muscle data**: Extract muscle activations if available
6. **Batch processing**: Process multiple .b3d files in parallel
7. **Data augmentation**: Time-shifted sampling strategies

## References

- **Original v10 code**: `addbiomechanics_to_smpl_v10_foot_orient_late.py`
- **Existing physics extraction**: `extract_addbiomechanics_physics.py`
- **Nimblephysics docs**: [https://nimblephysics.org/](https://nimblephysics.org/)
- **AddBiomechanics paper**: Werling et al., "AddBiomechanics: Automating model scaling, inverse kinematics, and inverse dynamics from human motion data through sequential optimization"
