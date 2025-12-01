#!/usr/bin/env python3
"""
AddBiomechanics (.b3d) data loader.

Efficiently loads and processes .b3d files from AddBiomechanics dataset.
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, List, Tuple

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from .coordinate_systems import CoordinateConverter

try:
    import nimblephysics as nimble
    NIMBLE_AVAILABLE = True
except ImportError:
    NIMBLE_AVAILABLE = False


@dataclass
class B3DSequence:
    """Container for loaded B3D sequence data."""
    joints: np.ndarray  # [T, J, 3] joint positions
    joint_names: List[str]  # [J] joint names
    dt: float  # timestep in seconds
    fps: float  # frames per second
    num_frames: int  # T
    num_joints: int  # J
    coordinate_system: Literal["z_up", "y_up"]  # current coordinate system

    def to_tensor(self, device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """Convert joints to torch tensor."""
        return torch.from_numpy(self.joints).float().to(device)

    def convert_coordinates(self, target: Literal["z_up", "y_up"]) -> "B3DSequence":
        """
        Convert to target coordinate system.

        Args:
            target: Target coordinate system

        Returns:
            New B3DSequence with converted coordinates
        """
        if self.coordinate_system == target:
            return self

        converted_joints = CoordinateConverter.apply_conversion(
            self.joints,
            source=self.coordinate_system,
            target=target
        )

        return B3DSequence(
            joints=converted_joints,
            joint_names=self.joint_names,
            dt=self.dt,
            fps=self.fps,
            num_frames=self.num_frames,
            num_joints=self.num_joints,
            coordinate_system=target
        )


class B3DLoader:
    """
    Loader for AddBiomechanics .b3d files.

    Provides efficient loading with automatic coordinate conversion.
    """

    def __init__(self, coordinate_system: Literal["z_up", "y_up"] = "y_up"):
        """
        Initialize B3D loader.

        Args:
            coordinate_system: Target coordinate system for loaded data
        """
        if not NIMBLE_AVAILABLE:
            raise RuntimeError(
                "nimblephysics is required to read .b3d files. "
                "Install it with: pip install nimblephysics"
            )

        self.coordinate_system = coordinate_system

    def load(
        self,
        b3d_path: str,
        trial: int = 0,
        processing_pass: int = 0,
        start_frame: int = 0,
        num_frames: int = -1
    ) -> B3DSequence:
        """
        Load B3D sequence from file.

        Args:
            b3d_path: Path to .b3d file
            trial: Trial index (default: 0)
            processing_pass: Processing pass index (default: 0)
            start_frame: Starting frame index (default: 0)
            num_frames: Number of frames to load (-1 for all)

        Returns:
            B3DSequence object with loaded data
        """
        # Load joint positions
        joints, dt = self._load_joint_positions(
            b3d_path, trial, processing_pass, start_frame, num_frames
        )

        # Infer joint names
        joint_names = self._infer_joint_names(b3d_path, trial, processing_pass)

        # Create sequence object (in original Z-up coordinates)
        sequence = B3DSequence(
            joints=joints,
            joint_names=joint_names,
            dt=dt,
            fps=1.0 / dt if dt > 0 else 0.0,
            num_frames=joints.shape[0],
            num_joints=joints.shape[1],
            coordinate_system="z_up"
        )

        # Convert to target coordinate system
        return sequence.convert_coordinates(self.coordinate_system)

    def _load_joint_positions(
        self,
        b3d_path: str,
        trial: int,
        processing_pass: int,
        start: int,
        num_frames: int
    ) -> Tuple[np.ndarray, float]:
        """
        Load joint positions from .b3d file.

        Returns:
            joints: [T, J, 3] joint positions
            dt: timestep in seconds
        """
        subject = nimble.biomechanics.SubjectOnDisk(b3d_path)
        total_frames = subject.getTrialLength(trial)

        # Determine number of frames to load
        if num_frames < 0:
            count = total_frames - start
        else:
            count = min(num_frames, total_frames - start)

        # Read frames
        frames = subject.readFrames(
            trial=trial,
            startFrame=start,
            numFramesToRead=count,
            includeProcessingPasses=True,
            includeSensorData=False,
            stride=1,
            contactThreshold=1.0
        )

        if len(frames) == 0:
            raise RuntimeError(f"Failed to read frames from {b3d_path}")

        # Extract joint centers
        def get_joint_centers(frame) -> np.ndarray:
            if hasattr(frame, 'processingPasses') and len(frame.processingPasses) > 0:
                idx = min(processing_pass, len(frame.processingPasses) - 1)
                return np.asarray(frame.processingPasses[idx].jointCenters, dtype=np.float32)
            return np.asarray(frame.jointCenters, dtype=np.float32)

        # Get joint centers for first frame to determine shape
        first_centers = get_joint_centers(frames[0])
        if first_centers.ndim != 1 or first_centers.size % 3 != 0:
            raise ValueError(f"Unexpected joint center layout in {b3d_path}")

        num_joints = first_centers.size // 3

        # Load all frames
        joints = np.zeros((len(frames), num_joints, 3), dtype=np.float32)
        for i, frame in enumerate(frames):
            centers = get_joint_centers(frame)
            joints[i] = centers.reshape(-1, 3)[:num_joints]

        # Get timestep
        dt = float(subject.getTrialTimestep(trial))

        return joints, dt

    def _infer_joint_names(
        self,
        b3d_path: str,
        trial: int,
        processing_pass: int
    ) -> List[str]:
        """
        Infer joint names from skeleton structure.

        Returns:
            List of joint names
        """
        subject = nimble.biomechanics.SubjectOnDisk(b3d_path)

        # Read one frame to get joint centers
        frames = subject.readFrames(
            trial=trial,
            startFrame=0,
            numFramesToRead=1,
            includeProcessingPasses=True,
            includeSensorData=False,
            stride=1,
            contactThreshold=1.0
        )

        if len(frames) == 0:
            raise RuntimeError(f"Unable to infer joint names from {b3d_path}")

        # Get processing pass
        pp_idx = min(processing_pass, len(frames[0].processingPasses) - 1)
        pp = frames[0].processingPasses[pp_idx]

        # Load skeleton
        skel = subject.readSkel(trial)
        pos = np.asarray(pp.pos, dtype=np.float32)

        if pos.size == skel.getNumDofs():
            skel.setPositions(pos)

        # Get world positions of all joints
        world_joints = []
        for i in range(skel.getNumJoints()):
            joint = skel.getJoint(i)
            world_pos = joint.getChildBodyNode().getWorldTransform().translation()
            world_joints.append((joint.getName(), world_pos))

        # Match joint centers to skeleton joints
        joint_centers = np.asarray(pp.jointCenters, dtype=np.float32).reshape(-1, 3)
        names = []

        for center in joint_centers:
            # Find closest skeleton joint
            dists = [np.linalg.norm(center - world_pos) for _, world_pos in world_joints]
            best_idx = int(np.argmin(dists))
            names.append(world_joints[best_idx][0])

        return names

    @staticmethod
    def get_available_trials(b3d_path: str) -> int:
        """Get number of available trials in .b3d file."""
        if not NIMBLE_AVAILABLE:
            raise RuntimeError("nimblephysics is required")

        subject = nimble.biomechanics.SubjectOnDisk(b3d_path)
        return subject.getNumTrials()

    @staticmethod
    def get_trial_length(b3d_path: str, trial: int = 0) -> int:
        """Get length of a specific trial."""
        if not NIMBLE_AVAILABLE:
            raise RuntimeError("nimblephysics is required")

        subject = nimble.biomechanics.SubjectOnDisk(b3d_path)
        return subject.getTrialLength(trial)
