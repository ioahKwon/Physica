"""
Main AddB → SKEL conversion pipeline.

Implements the full 2-stage conversion:
1. Scale Estimation: Estimate SKEL betas from AddB bone lengths
2. Pose Optimization: Optimize SKEL poses to match AddB joint positions

Following the approaches from:
- AddBiomechanics: https://nimblephysics.org/docs/working-with-addbiomechanics-data.html
- HSMR SKELify: https://github.com/IsshikiHugh/HSMR
"""

import os
from typing import Optional, Tuple, Dict, Union
import numpy as np
import torch

from .config import OptimizationConfig, ConversionResult, DEFAULT_CONFIG
from .skel_interface import SKELInterface, create_skel_interface
from .scale_estimation import estimate_subject_scale
from .pose_optimization import optimize_poses
from .scapula_handler import ScapulaHandler
from .joint_definitions import build_direct_joint_mapping, ADDB_JOINTS, SKEL_JOINTS
from .utils.geometry import convert_addb_to_skel_coords


def convert_addb_to_skel(
    addb_joints: np.ndarray,
    gender: str = 'male',
    config: Optional[OptimizationConfig] = None,
    height_m: Optional[float] = None,
    shoulder_width_m: Optional[float] = None,
    return_vertices: bool = False,
    verbose: bool = True,
) -> ConversionResult:
    """
    Convert AddBiomechanics joint trajectories to SKEL format.

    This is the main entry point for the addb2skel pipeline.

    Args:
        addb_joints: AddB joint positions [T, 20, 3] in meters.
            Must follow the AddB joint ordering:
            0: pelvis, 1-5: right leg, 6-10: left leg, 11: torso,
            12-15: right arm, 16-19: left arm.
        gender: Subject gender ('male' or 'female').
        config: Optimization configuration. Default: DEFAULT_CONFIG.
        height_m: Optional subject height in meters (for initialization).
        shoulder_width_m: Optional shoulder width in meters (for initialization).
        return_vertices: Whether to return mesh vertices.
        verbose: Print progress.

    Returns:
        ConversionResult containing:
        - skel_joints: SKEL joint positions [T, 24, 3]
        - skel_poses: SKEL pose parameters [T, 46]
        - skel_betas: SKEL shape parameters [10]
        - skel_trans: SKEL translations [T, 3]
        - mpjpe_mm: Mean per-joint position error in mm
        - per_joint_error: Per-joint errors
        - scapula_dofs: Final scapula DOF values

    Example:
        >>> from addb2skel import convert_addb_to_skel
        >>> from addb2skel.utils.io import load_b3d
        >>>
        >>> # Load AddB data
        >>> addb_joints, joint_names, metadata = load_b3d('subject.b3d')
        >>>
        >>> # Convert to SKEL
        >>> result = convert_addb_to_skel(
        ...     addb_joints,
        ...     gender='male',
        ...     height_m=metadata['height_m'],
        ...     verbose=True
        ... )
        >>>
        >>> print(f"MPJPE: {result.mpjpe_mm:.1f} mm")
    """
    config = config or DEFAULT_CONFIG
    device = config.get_device()

    if verbose:
        print("=" * 60)
        print("AddB → SKEL Conversion Pipeline")
        print("=" * 60)
        print(f"Input: {addb_joints.shape[0]} frames, {addb_joints.shape[1]} joints")
        print(f"Gender: {gender}")
        print(f"Device: {device}")

    # Validate input shape
    T, num_joints, _ = addb_joints.shape
    if num_joints != 20:
        raise ValueError(f"Expected 20 AddB joints, got {num_joints}")

    # Convert AddB coordinates to SKEL coordinate system
    addb_joints_converted = convert_addb_to_skel_coords(addb_joints)

    # Initialize SKEL model
    if verbose:
        print("\n--- Loading SKEL Model ---")
    skel = create_skel_interface(gender=gender, device=str(device))

    # =========================================================================
    # Stage 1: Scale Estimation
    # =========================================================================
    if verbose:
        print("\n--- Stage 1: Scale Estimation ---")

    betas, scale_stats = estimate_subject_scale(
        addb_joints_converted,
        skel,
        config,
        height_m=height_m,
        shoulder_width_m=shoulder_width_m,
        verbose=verbose,
    )

    if verbose:
        print(f"  Estimated betas: {betas[:5].cpu().numpy()}")
        print(f"  Mean bone length error: {scale_stats['mean_length_error_mm']:.2f} mm")

    # =========================================================================
    # Stage 2: Pose Optimization
    # =========================================================================
    if verbose:
        print("\n--- Stage 2: Pose Optimization ---")

    poses, trans, pose_stats = optimize_poses(
        addb_joints_converted,
        betas,
        skel,
        config,
        verbose=verbose,
    )

    if verbose:
        print(f"\n  Final MPJPE: {pose_stats['mpjpe_mm']:.1f} mm")
        print(f"  Scapula DOFs (mean):")
        for k, v in pose_stats['scapula_dofs'].items():
            print(f"    {k}: {v:.3f} rad")

    # =========================================================================
    # Generate Final Outputs
    # =========================================================================
    if verbose:
        print("\n--- Generating Final Outputs ---")

    # Forward pass to get joints (and optionally vertices)
    with torch.no_grad():
        if return_vertices:
            vertices, joints, _ = skel.forward(
                betas.unsqueeze(0).expand(T, -1), poses, trans, return_skeleton=False
            )
        else:
            joints = skel.forward_kinematics(
                betas.unsqueeze(0).expand(T, -1), poses, trans
            )
            vertices = None

    # Build result
    result = ConversionResult(
        skel_joints=joints,
        skel_poses=poses,
        skel_betas=betas,
        skel_trans=trans,
        skel_vertices=vertices,
        mpjpe_mm=pose_stats['mpjpe_mm'],
        per_joint_error=pose_stats.get('per_joint_error_mm'),
        scapula_dofs=pose_stats['scapula_dofs'],
        num_frames=T,
        gender=gender,
    )

    if verbose:
        print("\n" + "=" * 60)
        print("Conversion Complete!")
        print("=" * 60)
        print(f"Output shapes:")
        print(f"  skel_joints: {result.skel_joints.shape}")
        print(f"  skel_poses:  {result.skel_poses.shape}")
        print(f"  skel_betas:  {result.skel_betas.shape}")
        print(f"  skel_trans:  {result.skel_trans.shape}")
        if result.skel_vertices is not None:
            print(f"  skel_vertices: {result.skel_vertices.shape}")

    return result


def save_conversion_result(
    result: ConversionResult,
    output_dir: str,
    save_obj: bool = True,
    save_npz: bool = True,
):
    """
    Save conversion result to files.

    Args:
        result: ConversionResult from convert_addb_to_skel().
        output_dir: Output directory.
        save_obj: Save OBJ mesh files.
        save_npz: Save NPZ parameter files.
    """
    from .utils.io import save_obj as write_obj
    from .utils.visualization import joints_to_obj

    os.makedirs(output_dir, exist_ok=True)

    # Save parameters as NPZ
    if save_npz:
        np.savez(
            os.path.join(output_dir, 'skel_params.npz'),
            poses=result.skel_poses.cpu().numpy(),
            betas=result.skel_betas.cpu().numpy(),
            trans=result.skel_trans.cpu().numpy(),
            joints=result.skel_joints.cpu().numpy(),
        )
        print(f"Saved parameters to {output_dir}/skel_params.npz")

    # Save OBJ files
    if save_obj:
        # Get SKEL interface for faces
        skel = create_skel_interface(result.gender)

        # Save first frame mesh if vertices available
        if result.skel_vertices is not None:
            write_obj(
                result.skel_vertices[0].cpu().numpy(),
                skel.faces,
                os.path.join(output_dir, 'skel_mesh.obj'),
            )
            print(f"Saved mesh to {output_dir}/skel_mesh.obj")

        # Save joints for each frame
        for t in range(min(result.num_frames, 10)):  # Limit to first 10 frames
            joints_to_obj(
                result.skel_joints[t].cpu().numpy(),
                os.path.join(output_dir, f'skel_joints_frame{t:04d}.obj'),
                parents=skel.parents,
            )

        print(f"Saved joint OBJs to {output_dir}/")


def quick_convert(
    b3d_path: str,
    output_dir: str,
    num_frames: Optional[int] = None,
    verbose: bool = True,
) -> ConversionResult:
    """
    Quick conversion from .b3d file to SKEL.

    Args:
        b3d_path: Path to AddB .b3d file.
        output_dir: Output directory for results.
        num_frames: Number of frames to process. None for all.
        verbose: Print progress.

    Returns:
        ConversionResult.
    """
    from .utils.io import load_b3d

    # Load AddB data
    if verbose:
        print(f"Loading {b3d_path}...")

    addb_joints, joint_names, metadata = load_b3d(b3d_path, num_frames=num_frames)

    if verbose:
        print(f"  Subject: height={metadata['height_m']:.2f}m, "
              f"mass={metadata['mass_kg']:.1f}kg, sex={metadata['sex']}")

    # Determine gender
    gender = 'male' if metadata['sex'] == 'male' else 'female'

    # Convert
    result = convert_addb_to_skel(
        addb_joints,
        gender=gender,
        height_m=metadata['height_m'],
        return_vertices=True,
        verbose=verbose,
    )

    # Save results
    save_conversion_result(result, output_dir, save_obj=True, save_npz=True)

    return result
