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
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """
    Load AddBiomechanics .b3d file.

    Args:
        b3d_path: Path to .b3d file.
        num_frames: Number of frames to load. None for all.
        start_frame: Starting frame index.

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

    # Get trial
    trial = subject.readFrames(
        trial=0,
        startFrame=start_frame,
        numFramesToRead=num_frames if num_frames else subject.getTrialLength(0) - start_frame,
    )

    # Extract joint positions
    T = len(trial)
    joint_names = []
    for i, joint in enumerate(skel.getJoints()):
        joint_names.append(joint.getName())

    num_joints = len(joint_names)
    joints = np.zeros((T, num_joints, 3))

    for t, frame in enumerate(trial):
        skel.setPositions(frame.pos)
        for i, joint in enumerate(skel.getJoints()):
            joints[t, i] = joint.getWorldTransform().translation()

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
