"""
I/O utilities for loading and saving data.
"""

import os
from typing import Tuple, List, Optional, Dict, Any
import numpy as np


def load_b3d(
    b3d_path: str,
    num_frames: Optional[int] = None,
    start_frame: int = 0,
    processing_pass: int = 0,
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """
    Load AddBiomechanics .b3d file.

    Args:
        b3d_path: Path to .b3d file.
        num_frames: Number of frames to load. None for all.
        start_frame: Starting frame index.
        processing_pass: Processing pass index (0 for kinematics, 1 for dynamics).

    Returns:
        joints: Joint positions [T, N_joints, 3] in meters.
        joint_names: List of joint names.
        metadata: Dictionary with subject info (height, mass, sex, etc.).
    """
    try:
        import nimblephysics as nimble
    except ImportError:
        raise ImportError(
            "nimblephysics is required to read .b3d files. "
            "Install: pip install nimblephysics"
        )

    # Load subject
    subject = nimble.biomechanics.SubjectOnDisk(b3d_path)

    # Get skeleton
    skel = subject.readSkel(0, ignoreGeometry=True)

    # Determine frame range
    trial_length = subject.getTrialLength(0)
    end_frame = start_frame + num_frames if num_frames else trial_length
    end_frame = min(end_frame, trial_length)
    frames_to_read = end_frame - start_frame

    # Get trial frames
    trial = subject.readFrames(
        trial=0,
        startFrame=start_frame,
        numFramesToRead=frames_to_read,
    )

    T = len(trial)

    # Get joint names from skeleton using correct nimblephysics API
    # Build mapping from joint centers to joint names
    pp_idx = min(processing_pass, len(trial[0].processingPasses) - 1)
    pp = trial[0].processingPasses[pp_idx]

    # Set skeleton positions
    pos = np.asarray(pp.pos, dtype=np.float32)
    if pos.size == skel.getNumDofs():
        skel.setPositions(pos)

    # Get world joint positions and names
    world_joints = []
    for i in range(skel.getNumJoints()):
        joint = skel.getJoint(i)
        world = joint.getChildBodyNode().getWorldTransform().translation()
        joint_name = joint.getName()
        world_joints.append((joint_name, world))

    # Use jointCenters from processing pass and match to joint names
    joint_centers_first = np.asarray(pp.jointCenters, dtype=np.float32).reshape(-1, 3)
    num_joints = joint_centers_first.shape[0]

    joint_names = []
    for center in joint_centers_first:
        dists = [np.linalg.norm(center - w) for _, w in world_joints]
        best = int(np.argmin(dists))
        joint_names.append(world_joints[best][0])

    # Extract joint positions for all frames using jointCenters
    joints = np.zeros((T, num_joints, 3))
    for t, frame in enumerate(trial):
        pp = frame.processingPasses[pp_idx]
        joint_centers = np.asarray(pp.jointCenters, dtype=np.float32).reshape(-1, 3)
        joints[t] = joint_centers

    # Get subject info
    metadata = {
        'height_m': subject.getHeightM(),
        'mass_kg': subject.getMassKg(),
        'sex': subject.getBiologicalSex(),
        'age': subject.getAgeYears() if hasattr(subject, 'getAgeYears') else None,
        'num_frames': T,
        'num_joints': num_joints,
    }

    return joints, joint_names, metadata


def save_npy(data: np.ndarray, path: str):
    """Save numpy array to .npy file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, data)


def load_npy(path: str) -> np.ndarray:
    """Load numpy array from .npy file."""
    return np.load(path)


def save_obj(
    vertices: np.ndarray,
    faces: np.ndarray,
    path: str,
    vertex_colors: Optional[np.ndarray] = None,
):
    """
    Save mesh to OBJ file.

    Args:
        vertices: Vertex positions [V, 3].
        faces: Face indices [F, 3] (0-indexed).
        path: Output path.
        vertex_colors: Optional vertex colors [V, 3] in [0, 1].
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w') as f:
        for i, v in enumerate(vertices):
            if vertex_colors is not None:
                c = vertex_colors[i]
                f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}\n')
            else:
                f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')

        for face in faces:
            # OBJ uses 1-indexed faces
            f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')


def save_npz(
    path: str,
    **arrays: np.ndarray,
):
    """Save multiple arrays to .npz file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **arrays)


def load_npz(path: str) -> Dict[str, np.ndarray]:
    """Load arrays from .npz file."""
    return dict(np.load(path))
