#!/usr/bin/env python3
"""
Example script demonstrating how to load and visualize the physics data
extracted by addbiomechanics_to_smpl_v11_with_physics.py

Usage:
    python example_load_physics_data.py --output_dir /path/to/v11_with_physics_output
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_physics_data(output_dir: str) -> dict:
    """
    Load all physics data from a v11_with_physics output directory.

    Args:
        output_dir: Path to the output directory

    Returns:
        dict containing all physics arrays and metadata
    """
    output_dir = Path(output_dir)

    # Load metadata
    with open(output_dir / 'meta.json', 'r') as f:
        meta = json.load(f)

    # Load physics data
    data = {
        'grf': np.load(output_dir / 'ground_reaction_forces.npy'),
        'joint_torques': np.load(output_dir / 'joint_torques.npy'),
        'joint_velocities': np.load(output_dir / 'joint_velocities.npy'),
        'joint_accelerations': np.load(output_dir / 'joint_accelerations.npy'),
        'com_position': np.load(output_dir / 'com_position.npy'),
        'com_velocity': np.load(output_dir / 'com_velocity.npy'),
        'com_acceleration': np.load(output_dir / 'com_acceleration.npy'),
        'residual_forces': np.load(output_dir / 'residual_forces.npy'),
        'sampled_frame_indices': np.load(output_dir / 'sampled_frame_indices.npy'),
        'target_joints': np.load(output_dir / 'target_joints.npy'),
        'pred_joints': np.load(output_dir / 'pred_joints.npy'),
        'meta': meta
    }

    return data


def print_physics_summary(data: dict):
    """Print summary statistics of physics data."""

    print("=" * 80)
    print("PHYSICS DATA SUMMARY")
    print("=" * 80)

    # Basic info
    n_frames = len(data['sampled_frame_indices'])
    n_dof = data['joint_torques'].shape[1]
    n_joints = data['target_joints'].shape[1]
    dt = data['meta']['dt']

    print(f"\nBasic Information:")
    print(f"  Number of sampled frames: {n_frames}")
    print(f"  Number of DOFs: {n_dof}")
    print(f"  Number of joints: {n_joints}")
    print(f"  Time step: {dt:.4f} s ({1/dt:.1f} Hz)")
    print(f"  Sequence duration: {n_frames * dt:.2f} s")

    # Ground Reaction Forces
    grf_forces = data['grf'][:, :3]  # [T, 3]
    grf_torques = data['grf'][:, 3:]  # [T, 3]
    grf_magnitudes = np.linalg.norm(grf_forces, axis=1)

    print(f"\nGround Reaction Forces:")
    print(f"  Mean magnitude: {grf_magnitudes.mean():.2f} N")
    print(f"  Peak magnitude: {grf_magnitudes.max():.2f} N")
    print(f"  Min magnitude: {grf_magnitudes.min():.2f} N")
    print(f"  Frames with GRF > 50N: {(grf_magnitudes > 50).sum()} / {n_frames} ({100*(grf_magnitudes > 50).sum()/n_frames:.1f}%)")

    # Joint Torques
    torque_magnitudes = np.linalg.norm(data['joint_torques'], axis=1)

    print(f"\nJoint Torques:")
    print(f"  Mean L2 norm: {torque_magnitudes.mean():.2f} Nm")
    print(f"  Peak L2 norm: {torque_magnitudes.max():.2f} Nm")
    print(f"  Max absolute torque: {np.abs(data['joint_torques']).max():.2f} Nm")
    print(f"  Mean absolute torque: {np.abs(data['joint_torques']).mean():.2f} Nm")

    # Center of Mass
    com_height = data['com_position'][:, 1]  # Y is up
    com_vel_magnitude = np.linalg.norm(data['com_velocity'], axis=1)

    print(f"\nCenter of Mass:")
    print(f"  Mean height: {com_height.mean():.3f} m")
    print(f"  Height range: [{com_height.min():.3f}, {com_height.max():.3f}] m")
    print(f"  Mean velocity: {com_vel_magnitude.mean():.3f} m/s")
    print(f"  Peak velocity: {com_vel_magnitude.max():.3f} m/s")

    # Residuals
    residual_forces = data['residual_forces'][:, :3]
    residual_moments = data['residual_forces'][:, 3:]
    residual_force_mag = np.linalg.norm(residual_forces, axis=1)
    residual_moment_mag = np.linalg.norm(residual_moments, axis=1)

    print(f"\nResidual Forces (physics quality indicator):")
    print(f"  Mean force magnitude: {residual_force_mag.mean():.2f} N")
    print(f"  Mean moment magnitude: {residual_moment_mag.mean():.2f} Nm")
    print(f"  Peak force: {residual_force_mag.max():.2f} N")
    print(f"  Peak moment: {residual_moment_mag.max():.2f} Nm")

    # Frame Sampling
    frame_indices = data['sampled_frame_indices']
    if len(frame_indices) > 1:
        strides = np.diff(frame_indices)
        print(f"\nFrame Sampling:")
        print(f"  Original frame range: [{frame_indices[0]}, {frame_indices[-1]}]")
        print(f"  Mean stride: {strides.mean():.1f}")
        print(f"  Stride range: [{strides.min()}, {strides.max()}]")

    print("\n" + "=" * 80)


def plot_physics_data(data: dict, output_path: str = None):
    """Create visualization plots of physics data."""

    n_frames = len(data['sampled_frame_indices'])
    dt = data['meta']['dt']
    time = np.arange(n_frames) * dt

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Physics Data Visualization', fontsize=16, fontweight='bold')

    # 1. Ground Reaction Forces
    ax = axes[0, 0]
    grf = data['grf'][:, :3]
    ax.plot(time, grf[:, 0], label='Fx', alpha=0.7)
    ax.plot(time, grf[:, 1], label='Fy', alpha=0.7)
    ax.plot(time, grf[:, 2], label='Fz', alpha=0.7)
    grf_mag = np.linalg.norm(grf, axis=1)
    ax.plot(time, grf_mag, 'k--', label='Magnitude', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Force (N)')
    ax.set_title('Ground Reaction Forces')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Joint Torques (show top 5 DOFs by magnitude)
    ax = axes[0, 1]
    torques = data['joint_torques']
    mean_abs_torques = np.abs(torques).mean(axis=0)
    top_dofs = np.argsort(mean_abs_torques)[-5:]
    for dof in top_dofs:
        ax.plot(time, torques[:, dof], alpha=0.7, label=f'DOF {dof}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Torque (Nm)')
    ax.set_title('Joint Torques (Top 5 DOFs)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Center of Mass Position
    ax = axes[1, 0]
    com_pos = data['com_position']
    ax.plot(time, com_pos[:, 0], label='X', alpha=0.7)
    ax.plot(time, com_pos[:, 1], label='Y (height)', alpha=0.7, linewidth=2)
    ax.plot(time, com_pos[:, 2], label='Z', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (m)')
    ax.set_title('Center of Mass Position')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Center of Mass Velocity
    ax = axes[1, 1]
    com_vel = data['com_velocity']
    ax.plot(time, com_vel[:, 0], label='Vx', alpha=0.7)
    ax.plot(time, com_vel[:, 1], label='Vy', alpha=0.7)
    ax.plot(time, com_vel[:, 2], label='Vz', alpha=0.7)
    com_vel_mag = np.linalg.norm(com_vel, axis=1)
    ax.plot(time, com_vel_mag, 'k--', label='Speed', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Center of Mass Velocity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Residual Forces
    ax = axes[2, 0]
    residuals = data['residual_forces']
    res_force_mag = np.linalg.norm(residuals[:, :3], axis=1)
    res_moment_mag = np.linalg.norm(residuals[:, 3:], axis=1)
    ax.plot(time, res_force_mag, label='Force magnitude', linewidth=2)
    ax.plot(time, res_moment_mag, label='Moment magnitude', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Residual (N or Nm)')
    ax.set_title('Residual Forces/Moments (physics quality)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Joint Position Error (MPJPE over time)
    ax = axes[2, 1]
    target = data['target_joints']  # [T, N, 3]
    pred = data['pred_joints']  # [T, N, 3]
    errors = np.linalg.norm(target - pred, axis=2)  # [T, N]
    mpjpe_per_frame = errors.mean(axis=1) * 1000  # Convert to mm
    ax.plot(time, mpjpe_per_frame, linewidth=2, color='red')
    ax.axhline(mpjpe_per_frame.mean(), color='black', linestyle='--',
               label=f'Mean: {mpjpe_per_frame.mean():.1f} mm')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('MPJPE (mm)')
    ax.set_title('Mean Per-Joint Position Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Load and visualize physics data from v11_with_physics output'
    )
    parser.add_argument('--output_dir', required=True,
                        help='Path to v11_with_physics output directory')
    parser.add_argument('--plot', action='store_true',
                        help='Generate visualization plots')
    parser.add_argument('--save_plot', type=str, default=None,
                        help='Save plot to file (e.g., physics_plot.png)')

    args = parser.parse_args()

    print(f"\nLoading physics data from: {args.output_dir}\n")

    # Load data
    data = load_physics_data(args.output_dir)

    # Print summary
    print_physics_summary(data)

    # Plot if requested
    if args.plot or args.save_plot:
        plot_physics_data(data, output_path=args.save_plot)


if __name__ == '__main__':
    main()
