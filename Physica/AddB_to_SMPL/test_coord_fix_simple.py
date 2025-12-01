#!/usr/bin/env python3
"""
Simple test to verify coordinate fix - just load AddB joints and check orientation
"""

import numpy as np
import nimblephysics as nimble

# OLD conversion (INCORRECT)
def old_convert(joints):
    return np.stack([joints[:, 0], joints[:, 2], -joints[:, 1]], axis=-1)

# NEW conversion (CORRECT)
def new_convert(joints):
    return joints  # No conversion needed!

# Load AddB data
b3d_path = '/egr/research-zijunlab/kwonjoon/Dataset/AddB/train/With_Arm/Carter2023_Formatted_With_Arm/P020_split2/P020_split2.b3d'
subject = nimble.biomechanics.SubjectOnDisk(b3d_path)

frames = subject.readFrames(
    trial=0,
    startFrame=0,
    numFramesToRead=1,
    stride=1,
    includeSensorData=False,
    includeProcessingPasses=True
)

addb_joints = np.asarray(frames[0].processingPasses[0].jointCenters, dtype=np.float32).reshape(-1, 3)

print("="*80)
print("COORDINATE CONVERSION COMPARISON")
print("="*80)

print("\n[1] AddBiomechanics ORIGINAL joints (frame 0):")
print(f"    Pelvis (joint 0): X={addb_joints[0, 0]:.3f}, Y={addb_joints[0, 1]:.3f}, Z={addb_joints[0, 2]:.3f}")
print(f"    Y-axis range: [{addb_joints[:, 1].min():.3f}, {addb_joints[:, 1].max():.3f}]")
y_span = addb_joints[:, 1].max() - addb_joints[:, 1].min()
print(f"    Y-axis span: {y_span:.3f}m")
if y_span > 1.0:
    print(f"    ✓ Standing upright (Y-axis dominant)")
else:
    print(f"    ✗ NOT standing upright")

print("\n[2] OLD conversion (X, Z, -Y) - INCORRECT:")
old_joints = old_convert(addb_joints)
print(f"    Pelvis (joint 0): X={old_joints[0, 0]:.3f}, Y={old_joints[0, 1]:.3f}, Z={old_joints[0, 2]:.3f}")
print(f"    Y-axis range: [{old_joints[:, 1].min():.3f}, {old_joints[:, 1].max():.3f}]")
y_span_old = old_joints[:, 1].max() - old_joints[:, 1].min()
z_span_old = old_joints[:, 2].max() - old_joints[:, 2].min()
print(f"    Y-axis span: {y_span_old:.3f}m")
print(f"    Z-axis span: {z_span_old:.3f}m")
if z_span_old > y_span_old:
    print(f"    ✗ Z-axis dominant - skeleton lying down!")
else:
    print(f"    ✓ Y-axis dominant")

print("\n[3] NEW conversion (no change) - CORRECT:")
new_joints = new_convert(addb_joints)
print(f"    Pelvis (joint 0): X={new_joints[0, 0]:.3f}, Y={new_joints[0, 1]:.3f}, Z={new_joints[0, 2]:.3f}")
print(f"    Y-axis range: [{new_joints[:, 1].min():.3f}, {new_joints[:, 1].max():.3f}]")
y_span_new = new_joints[:, 1].max() - new_joints[:, 1].min()
print(f"    Y-axis span: {y_span_new:.3f}m")
if y_span_new > 1.0:
    print(f"    ✓ Standing upright (Y-axis dominant)")
else:
    print(f"    ✗ NOT standing upright")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
print("The OLD conversion (X, Z, -Y) rotates the skeleton 90 degrees, making it lie down.")
print("The NEW conversion (identity) keeps the skeleton upright as intended.")
print("This should fix the poor MPJPE of 79.5mm for P020!")
