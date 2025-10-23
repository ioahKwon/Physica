#!/usr/bin/env python3
"""
Visualize comparison between v2 and improved SMPL fitting results
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load v2 results
v2_pred = np.load('/egr/research-zijunlab/kwonjoon/out/AddB_SMPL_Full/test/No_Arm/Camargo2021_Formatted_No_Arm/AB10_split0/no_arm_camargo2021_ab10/pred_joints.npy')
v2_target = np.load('/egr/research-zijunlab/kwonjoon/out/AddB_SMPL_Full/test/No_Arm/Camargo2021_Formatted_No_Arm/AB10_split0/no_arm_camargo2021_ab10/target_joints.npy')

# Load improved results (FIXED version)
improved_pred = np.load('/egr/research-zijunlab/kwonjoon/out/AddB_SMPL_Improved_FIXED/pred_joints.npy')
improved_target = np.load('/egr/research-zijunlab/kwonjoon/out/AddB_SMPL_Improved_FIXED/target_joints.npy')

print("="*80)
print("Data Shapes:")
print("="*80)
print(f"v2 pred:        {v2_pred.shape}")
print(f"v2 target:      {v2_target.shape}")
print(f"improved pred:  {improved_pred.shape}")
print(f"improved target: {improved_target.shape}")
print()

# Check if target data is the same
print("="*80)
print("Target Data Comparison:")
print("="*80)
print(f"v2_target[0,0]:       {v2_target[0,0]}")
print(f"improved_target[0,0]: {improved_target[0,0]}")
print(f"Are targets identical? {np.allclose(v2_target, improved_target)}")
print()

# Compare predictions
print("="*80)
print("Prediction Data Comparison:")
print("="*80)
print(f"v2_pred[0,0]:       {v2_pred[0,0]}")
print(f"improved_pred[0,0]: {improved_pred[0,0]}")
print()

# Compute statistics
print("="*80)
print("Statistics:")
print("="*80)
print(f"v2_pred range:       [{v2_pred.min():.3f}, {v2_pred.max():.3f}]")
print(f"improved_pred range: [{improved_pred.min():.3f}, {improved_pred.max():.3f}]")
print(f"v2_target range:     [{v2_target.min():.3f}, {v2_target.max():.3f}]")
print()

# Visualize frame 0
frame_idx = 0
fig = plt.figure(figsize=(18, 6))

# Plot 1: v2 prediction vs target
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(v2_pred[frame_idx, :, 0], v2_pred[frame_idx, :, 1], v2_pred[frame_idx, :, 2],
           c='blue', marker='o', s=50, label='v2 pred (SMPL 24 joints)', alpha=0.6)
ax1.scatter(v2_target[frame_idx, :, 0], v2_target[frame_idx, :, 1], v2_target[frame_idx, :, 2],
           c='red', marker='x', s=100, label='target (AddB 12 joints)', alpha=0.8)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title(f'v2 Results (Frame {frame_idx})')
ax1.legend()
ax1.set_box_aspect([1,1,1])

# Plot 2: improved prediction vs target
ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(improved_pred[frame_idx, :, 0], improved_pred[frame_idx, :, 1], improved_pred[frame_idx, :, 2],
           c='green', marker='o', s=50, label='improved pred (SMPL 24 joints)', alpha=0.6)
ax2.scatter(improved_target[frame_idx, :, 0], improved_target[frame_idx, :, 1], improved_target[frame_idx, :, 2],
           c='red', marker='x', s=100, label='target (AddB 12 joints)', alpha=0.8)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title(f'Improved Results (Frame {frame_idx})')
ax2.legend()
ax2.set_box_aspect([1,1,1])

# Plot 3: overlay comparison
ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(v2_pred[frame_idx, :, 0], v2_pred[frame_idx, :, 1], v2_pred[frame_idx, :, 2],
           c='blue', marker='o', s=50, label='v2 pred', alpha=0.5)
ax3.scatter(improved_pred[frame_idx, :, 0], improved_pred[frame_idx, :, 1], improved_pred[frame_idx, :, 2],
           c='green', marker='^', s=50, label='improved pred', alpha=0.5)
ax3.scatter(v2_target[frame_idx, :, 0], v2_target[frame_idx, :, 1], v2_target[frame_idx, :, 2],
           c='red', marker='x', s=100, label='target', alpha=0.8)
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')
ax3.set_title(f'Overlay Comparison (Frame {frame_idx})')
ax3.legend()
ax3.set_box_aspect([1,1,1])

plt.tight_layout()
plt.savefig('/egr/research-zijunlab/kwonjoon/out/comparison_frame0.png', dpi=150, bbox_inches='tight')
print(f"Saved visualization to /egr/research-zijunlab/kwonjoon/out/comparison_frame0.png")
print()

# Compute MPJPE using the CORRECT centering method
print("="*80)
print("MPJPE Computation (CORRECT method - matching v2):")
print("="*80)

# Joint mapping for AB10_split0 (from previous analysis)
joint_mapping = {0: 0, 1: 2, 2: 5, 3: 8, 4: 11, 6: 1, 7: 4, 8: 7, 9: 10, 11: 3}
root_addb_idx = 0

def compute_mpjpe_correct(pred, target, joint_mapping, root_idx):
    """Compute MPJPE using v2's correct centering method"""
    errors = []
    for pred_frame, target_frame in zip(pred, target):
        # Center on root - CRITICAL: pred centers on pred[0], target centers on target[root]
        pred_centered = pred_frame.copy()
        target_centered = target_frame.copy()

        root_target = target_frame[root_idx]
        if not np.any(np.isnan(root_target)):
            pred_centered -= pred_frame[0]  # ✅ SMPL pelvis (index 0)
            target_centered -= root_target   # AddB root

        # Compute errors for mapped joints
        for addb_idx, smpl_idx in joint_mapping.items():
            if addb_idx >= target_centered.shape[0]:
                continue
            tgt = target_centered[addb_idx]
            if np.any(np.isnan(tgt)):
                continue
            diff = pred_centered[smpl_idx] - tgt
            error_m = np.linalg.norm(diff)
            errors.append(error_m * 1000.0)  # Convert to mm

    return np.mean(errors) if errors else float('nan')

v2_mpjpe = compute_mpjpe_correct(v2_pred, v2_target, joint_mapping, root_addb_idx)
improved_mpjpe = compute_mpjpe_correct(improved_pred, improved_target, joint_mapping, root_addb_idx)

print(f"v2 MPJPE:       {v2_mpjpe:.2f} mm")
print(f"improved MPJPE: {improved_mpjpe:.2f} mm")
print(f"Difference:     {improved_mpjpe - v2_mpjpe:.2f} mm ({((improved_mpjpe/v2_mpjpe - 1)*100):.1f}% {'worse' if improved_mpjpe > v2_mpjpe else 'better'})")
print()

# Analyze pelvis positions
print("="*80)
print("Pelvis Position Analysis:")
print("="*80)
print(f"v2_pred pelvis (frame 0):       {v2_pred[0, 0]}")
print(f"improved_pred pelvis (frame 0): {improved_pred[0, 0]}")
print(f"target pelvis (frame 0):        {v2_target[0, 0]}")
print()

# Check if predictions are in similar coordinate space
print("="*80)
print("Coordinate Space Check:")
print("="*80)
v2_centroid = v2_pred[0].mean(axis=0)
improved_centroid = improved_pred[0].mean(axis=0)
target_centroid = v2_target[0].mean(axis=0)
print(f"v2_pred centroid (frame 0):       {v2_centroid}")
print(f"improved_pred centroid (frame 0): {improved_centroid}")
print(f"target centroid (frame 0):        {target_centroid}")
print()

# Scale analysis
print("="*80)
print("Scale Analysis:")
print("="*80)
v2_scale = np.std(v2_pred[0])
improved_scale = np.std(improved_pred[0])
target_scale = np.std(v2_target[0])
print(f"v2_pred std (frame 0):       {v2_scale:.4f}")
print(f"improved_pred std (frame 0): {improved_scale:.4f}")
print(f"target std (frame 0):        {target_scale:.4f}")
print()

print("="*80)
print("Analysis complete!")
print("="*80)
