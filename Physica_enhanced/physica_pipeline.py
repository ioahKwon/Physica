#!/usr/bin/env python3
"""
Main Physica pipeline for AddBiomechanics → SMPL conversion.

Integrates all modules for end-to-end processing.
"""

import torch
import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path

from core import PipelineConfig, SMPLModelWrapper
from data import B3DLoader, B3DSequence
from retargeting import RetargetingPipeline, RetargetingResult
from optimization import ShapeOptimizer, PoseOptimizer


@dataclass
class PipelineResult:
    """Result of full pipeline execution."""
    betas: np.ndarray  # [10] shape parameters
    poses: np.ndarray  # [T, 24, 3] pose parameters (axis-angle)
    trans: np.ndarray  # [T, 3] global translations
    pred_joints: np.ndarray  # [T, 24, 3] predicted SMPL joints
    target_joints: np.ndarray  # [T, J_source, 3] target joints from source
    metrics: Dict[str, float]  # evaluation metrics
    retargeting_result: RetargetingResult  # retargeting information


class PhysicaPipeline:
    """
    Main pipeline for AddBiomechanics → SMPL parameter fitting.

    Workflow:
    1. Load B3D data
    2. Retarget to SMPL skeleton (with synthesis)
    3. Optimize global shape (β) over all frames
    4. Optimize poses (θ) with keyframe sampling + interpolation
    5. Evaluate and save results
    """

    def __init__(
        self,
        smpl_model_path: str,
        config: Optional[PipelineConfig] = None,
        device: Optional[str] = None
    ):
        """
        Initialize pipeline.

        Args:
            smpl_model_path: Path to SMPL model .pkl file
            config: Pipeline configuration (uses default if None)
            device: Device for computation ("cuda" or "cpu", auto-detect if None)
        """
        self.config = config or PipelineConfig.default()

        # Setup device
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device(self.config.device)

        if self.device.type == 'cuda' and not torch.cuda.is_available():
            print("[WARNING] CUDA requested but not available, falling back to CPU")
            self.device = torch.device('cpu')

        # Initialize SMPL model
        self.smpl = SMPLModelWrapper(smpl_model_path, device=self.device)

        # Apply torch.compile if enabled (requires PyTorch 2.0+)
        if self.config.use_torch_compile:
            try:
                self.smpl = torch.compile(self.smpl, mode="reduce-overhead")
                if self.config.verbose:
                    print("[INFO] torch.compile enabled for SMPL model")
            except Exception as e:
                if self.config.verbose:
                    print(f"[WARNING] torch.compile failed: {e}")

        # Initialize components
        self.loader = B3DLoader(coordinate_system=self.config.retarget.coordinate_system)
        self.retargeting = RetargetingPipeline(config=self.config.retarget)
        self.shape_optimizer = ShapeOptimizer(
            self.smpl,
            self.config.shape_opt,
            self.device,
            verbose=self.config.verbose
        )
        self.pose_optimizer = PoseOptimizer(
            self.smpl,
            self.config.pose_opt,
            self.device,
            verbose=self.config.verbose
        )

    def run(
        self,
        b3d_path: str,
        trial: int = 0,
        processing_pass: int = 0,
        start_frame: int = 0,
        num_frames: int = -1,
        mapping_overrides: Optional[dict] = None
    ) -> PipelineResult:
        """
        Run full pipeline on B3D file.

        Args:
            b3d_path: Path to .b3d file
            trial: Trial index
            processing_pass: Processing pass index
            start_frame: Starting frame
            num_frames: Number of frames (-1 for all)
            mapping_overrides: Optional manual joint mapping overrides

        Returns:
            PipelineResult with all outputs
        """
        if self.config.verbose:
            print("="*80)
            print("Physica Pipeline - AddBiomechanics → SMPL")
            print("="*80)
            print(f"Input: {b3d_path}")
            print(f"Device: {self.device}")
            print()

        # Stage 1: Load data
        if self.config.verbose:
            print("[Stage 1/4] Loading B3D data...")

        sequence = self.loader.load(
            b3d_path,
            trial=trial,
            processing_pass=processing_pass,
            start_frame=start_frame,
            num_frames=num_frames
        )

        if self.config.verbose:
            print(f"  Frames: {sequence.num_frames}")
            print(f"  Joints: {sequence.num_joints}")
            print(f"  FPS: {sequence.fps:.2f}")
            print()

        # Stage 2: Retargeting
        if self.config.verbose:
            print("[Stage 2/4] Retargeting to SMPL skeleton...")

        retarget_result = self.retargeting.retarget(
            sequence.joints,
            sequence.joint_names,
            mapping_overrides
        )

        if self.config.verbose:
            print(f"  Mapped joints: {len(retarget_result.mapped_indices)}")
            print(f"  Synthesized joints: {len(retarget_result.synthesized_indices)}")
            if len(retarget_result.unmapped_source_joints) > 0:
                print(f"  Unmapped source joints: {retarget_result.unmapped_source_joints[:5]}")
            print()

        # Convert to tensor
        target_joints = sequence.to_tensor(self.device)

        # Stage 3: Shape optimization
        if self.config.verbose:
            print("[Stage 3/4] Optimizing global shape parameters...")

        mapped_indices = (
            retarget_result.joint_mapper.get_mapped_indices()[0],
            retarget_result.joint_mapper.get_mapped_indices()[1]
        )

        betas = self.shape_optimizer.optimize(
            target_joints,
            mapped_indices
        )

        if self.config.verbose:
            print()

        # Stage 4: Pose optimization
        if self.config.verbose:
            print("[Stage 4/4] Optimizing pose parameters...")

        poses, trans = self.pose_optimizer.optimize(
            target_joints,
            mapped_indices,
            betas
        )

        if self.config.verbose:
            print()

        # Evaluation
        if self.config.verbose:
            print("[Evaluation] Computing metrics...")

        metrics, pred_joints = self._evaluate(
            betas,
            poses,
            trans,
            target_joints,
            mapped_indices
        )

        if self.config.verbose:
            print(f"  MPJPE: {metrics['mpjpe']:.2f} mm")
            print(f"  Comparisons: {metrics['num_comparisons']:.0f}")
            print()

        # Convert to numpy
        result = PipelineResult(
            betas=betas.detach().cpu().numpy(),
            poses=poses.detach().cpu().numpy(),
            trans=trans.detach().cpu().numpy(),
            pred_joints=pred_joints,
            target_joints=sequence.joints,
            metrics=metrics,
            retargeting_result=retarget_result
        )

        if self.config.verbose:
            print("="*80)
            print("Pipeline complete!")
            print("="*80)
            print()

        return result

    def _evaluate(
        self,
        betas: torch.Tensor,
        poses: torch.Tensor,
        trans: torch.Tensor,
        target_joints: torch.Tensor,
        mapped_indices: Tuple[list, list]
    ) -> Tuple[Dict[str, float], np.ndarray]:
        """Evaluate SMPL fit quality."""
        source_indices, smpl_indices = mapped_indices

        # Compute predicted joints for all frames
        T = poses.shape[0]
        pred_joints_list = []

        with torch.no_grad():
            for t in range(T):
                _, joints = self.smpl(betas, poses[t], trans[t])
                pred_joints_list.append(joints.cpu().numpy())

        pred_joints = np.stack(pred_joints_list, axis=0)  # [T, 24, 3]

        # Compute MPJPE (Mean Per-Joint Position Error)
        errors = []

        for t in range(T):
            for src_idx, smpl_idx in zip(source_indices, smpl_indices):
                target_pos = target_joints[t, src_idx].detach().cpu().numpy()
                pred_pos = pred_joints[t, smpl_idx]

                if not np.any(np.isnan(target_pos)):
                    error = np.linalg.norm(pred_pos - target_pos) * 1000.0  # mm
                    errors.append(error)

        mpjpe = float(np.mean(errors)) if len(errors) > 0 else float('nan')

        metrics = {
            'mpjpe': mpjpe,
            'num_comparisons': float(len(errors))
        }

        return metrics, pred_joints

    def save_result(
        self,
        result: PipelineResult,
        output_dir: str,
        save_visualization: bool = False
    ) -> None:
        """
        Save pipeline results to directory.

        Args:
            result: Pipeline result
            output_dir: Output directory path
            save_visualization: Whether to save visualization data
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save SMPL parameters
        np.savez(
            output_path / 'smpl_params.npz',
            betas=result.betas,
            poses=result.poses,
            trans=result.trans
        )

        # Save predicted joints
        np.save(output_path / 'pred_joints.npy', result.pred_joints)

        # Save target joints
        np.save(output_path / 'target_joints.npy', result.target_joints)

        # Save metadata
        import json
        metadata = {
            'metrics': result.metrics,
            'num_frames': int(result.poses.shape[0]),
            'num_mapped_joints': len(result.retargeting_result.mapped_indices),
            'num_synthesized_joints': len(result.retargeting_result.synthesized_indices),
            'unmapped_source_joints': result.retargeting_result.unmapped_source_joints,
        }

        with open(output_path / 'meta.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        if self.config.verbose:
            print(f"Results saved to: {output_path}")
