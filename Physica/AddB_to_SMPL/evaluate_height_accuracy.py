#!/usr/bin/env python3
"""
Evaluate SMPL Height Accuracy against AddBiomechanics Ground Truth

Compares:
- Ground truth height from .b3d files (getHeightM())
- Predicted height from SMPL mesh vertices (max_z - min_z)

Outputs:
- Height error statistics (MAE, mean, median, min, max)
- Comparison plots for No_Arm vs With_Arm
- JSON report with per-subject metrics
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import nimblephysics as nimble
from typing import Dict, List, Tuple, Optional
import argparse
import torch

# Import SMPL model
from models.smpl_model import SMPLModel


def compute_smpl_height(vertices: np.ndarray) -> float:
    """
    Compute height from SMPL mesh vertices.
    Uses the z-axis extent (max - min) as height approximation.

    This correctly measures from foot sole to skull top using all 6890 vertices,
    unlike joint-based measurement which only goes from ankle to neck base.

    Args:
        vertices: (6890, 3) array of mesh vertex positions

    Returns:
        Height in meters (z-axis extent)
    """
    min_z = np.min(vertices[:, 2])
    max_z = np.max(vertices[:, 2])
    return max_z - min_z


def load_ground_truth_height(b3d_path: str) -> float:
    """
    Load ground truth height from .b3d file.

    Args:
        b3d_path: Path to .b3d file

    Returns:
        Height in meters
    """
    try:
        subj = nimble.biomechanics.SubjectOnDisk(b3d_path)
        height = float(subj.getHeightM())
        return height
    except Exception as e:
        print(f"Warning: Could not load height from {b3d_path}: {e}")
        return None


def load_b3d_path_from_meta(subject_dir: Path) -> Optional[Path]:
    """
    Load the .b3d file path from meta.json.

    Args:
        subject_dir: Directory containing SMPL results

    Returns:
        Path to .b3d file or None
    """
    meta_file = subject_dir / "meta.json"
    if not meta_file.exists():
        return None

    try:
        with open(meta_file, 'r') as f:
            meta = json.load(f)
            b3d_path = meta.get('b3d')
            if b3d_path and Path(b3d_path).exists():
                return Path(b3d_path)
    except Exception as e:
        print(f"Warning: Could not load b3d path from {meta_file}: {e}")

    return None


def evaluate_single_subject(subject_dir: Path, smpl_model: SMPLModel, device: str = 'cpu') -> Dict:
    """
    Evaluate height accuracy for a single subject.

    Args:
        subject_dir: Directory containing SMPL results
        smpl_model: Loaded SMPL model for forward pass
        device: Device for computation ('cpu' or 'cuda')

    Returns:
        Dictionary with evaluation metrics
    """
    # Load .b3d path from metadata
    b3d_path = load_b3d_path_from_meta(subject_dir)
    if b3d_path is None:
        return None

    # Load ground truth height
    gt_height = load_ground_truth_height(str(b3d_path))
    if gt_height is None:
        return None

    # Load SMPL parameters
    smpl_params_file = subject_dir / "smpl_params.npz"
    if not smpl_params_file.exists():
        print(f"Warning: smpl_params.npz not found in {subject_dir}")
        return None

    try:
        smpl_params = np.load(smpl_params_file)
        betas = smpl_params['betas']  # (T, 10) or (10,)
        poses = smpl_params['poses']   # (T, 24, 3) or (24, 3)
        trans = smpl_params['trans']   # (T, 3) or (3,)
    except Exception as e:
        print(f"Warning: Could not load SMPL params from {smpl_params_file}: {e}")
        return None

    # Handle both single-frame and multi-frame formats
    # betas: (10,) or (T, 10)
    # poses: (24, 3) or (T, 24, 3)
    # trans: (3,) or (T, 3)

    if betas.ndim == 1:
        # Single beta for all frames - replicate it
        num_frames = len(poses) if poses.ndim == 3 else 1
        betas = np.tile(betas, (num_frames, 1))  # (T, 10)

    if poses.ndim == 2:
        poses = poses[np.newaxis, :]  # (1, 24, 3)

    if trans.ndim == 1:
        trans = trans[np.newaxis, :]  # (1, 3)

    num_frames = len(poses)

    # Compute SMPL vertices for each frame using forward pass
    smpl_heights = []
    for t in range(num_frames):
        # Prepare inputs for SMPL forward pass
        beta_t = torch.from_numpy(betas[min(t, len(betas)-1)]).float().to(device).unsqueeze(0)  # (1, 10)
        pose_t = torch.from_numpy(poses[t]).float().to(device)  # (24, 3)

        # Flatten pose to (72,) as expected by SMPL
        if pose_t.ndim == 2:
            pose_t = pose_t.reshape(-1)  # (72,)
        pose_t = pose_t.unsqueeze(0)  # (1, 72)

        # Use zero translation for height measurement
        # We only care about the relative height (max_z - min_z), not absolute position
        trans_zero = torch.zeros(1, 3).float().to(device)

        # SMPL forward pass to get vertices
        try:
            vertices, _ = smpl_model.forward(beta_t, pose_t, trans_zero)  # (1, 6890, 3)
            vertices_np = vertices[0].cpu().numpy()  # (6890, 3)

            # Compute height from vertices
            h = compute_smpl_height(vertices_np)
            smpl_heights.append(h)
        except Exception as e:
            print(f"Warning: SMPL forward pass failed for {subject_dir} frame {t}: {e}")
            continue

    if not smpl_heights:
        print(f"Warning: No valid heights computed for {subject_dir}")
        return None

    mean_smpl_height = np.mean(smpl_heights)
    std_smpl_height = np.std(smpl_heights)

    # Compute error
    height_error = abs(mean_smpl_height - gt_height)
    relative_error = (height_error / gt_height) * 100  # percentage

    return {
        'subject_name': subject_dir.name,
        'gt_height': gt_height,
        'smpl_height_mean': mean_smpl_height,
        'smpl_height_std': std_smpl_height,
        'height_error': height_error,
        'relative_error': relative_error
    }


def find_b3d_path(subject_name: str, dataset_type: str) -> Path:
    """
    Find the original .b3d file path for a subject.

    Args:
        subject_name: Subject directory name (e.g., "train_Subject7_Subject7")
        dataset_type: "No_Arm" or "With_Arm"

    Returns:
        Path to .b3d file
    """
    # Parse subject name format: {split}_{study}_{subject}
    parts = subject_name.split('_')
    if len(parts) >= 3:
        split = parts[0]  # train/test
        study = parts[1]  # Study name
        subject = parts[2]  # Subject name
    else:
        return None

    # Construct base dataset path
    if dataset_type == "No_Arm":
        base_dir = Path("/egr/research-zijunlab/kwonjoon/Dataset/AddB/train/No_Arm")
    else:
        base_dir = Path("/egr/research-zijunlab/kwonjoon/Dataset/AddB/train/With_Arm")

    # Search for .b3d file
    b3d_path = base_dir / study / f"{subject}.b3d"

    if b3d_path.exists():
        return b3d_path

    # Try alternative naming patterns
    # Pattern: {study}_Formatted_No_Arm/{subject}/{subject}.b3d
    for study_dir in base_dir.iterdir():
        if not study_dir.is_dir():
            continue
        b3d_candidate = study_dir / subject / f"{subject}.b3d"
        if b3d_candidate.exists():
            return b3d_candidate

    return None


def evaluate_dataset(results_dir: Path, smpl_model: SMPLModel, device: str = 'cpu') -> List[Dict]:
    """
    Evaluate height accuracy for all subjects in a dataset.

    Args:
        results_dir: Directory containing SMPL results
        smpl_model: Loaded SMPL model for forward pass
        device: Device for computation ('cpu' or 'cuda')

    Returns:
        List of evaluation results
    """
    results = []

    for subject_dir in sorted(results_dir.iterdir()):
        if not subject_dir.is_dir():
            continue

        # Evaluate
        result = evaluate_single_subject(subject_dir, smpl_model, device)
        if result is not None:
            results.append(result)
            print(f"Processed {subject_dir.name}: "
                  f"GT={result['gt_height']:.3f}m, "
                  f"SMPL={result['smpl_height_mean']:.3f}m, "
                  f"Error={result['height_error']*100:.1f}cm")

    return results


def compute_statistics(results: List[Dict]) -> Dict:
    """
    Compute summary statistics from evaluation results.

    Args:
        results: List of evaluation results

    Returns:
        Dictionary with summary statistics
    """
    if not results:
        return {
            'count': 0,
            'mae': 0.0,
            'mean_error': 0.0,
            'median_error': 0.0,
            'std_error': 0.0,
            'min_error': 0.0,
            'max_error': 0.0,
            'mean_relative_error': 0.0,
            'median_relative_error': 0.0
        }

    errors = [r['height_error'] for r in results]
    relative_errors = [r['relative_error'] for r in results]

    return {
        'count': len(results),
        'mae': np.mean(errors),
        'mean_error': np.mean(errors),
        'median_error': np.median(errors),
        'std_error': np.std(errors),
        'min_error': np.min(errors),
        'max_error': np.max(errors),
        'mean_relative_error': np.mean(relative_errors),
        'median_relative_error': np.median(relative_errors)
    }


def plot_comparison(no_arm_results: List[Dict],
                   with_arm_results: List[Dict],
                   output_path: str):
    """
    Create comparison plots for height accuracy.

    Args:
        no_arm_results: Results for No_Arm dataset
        with_arm_results: Results for With_Arm dataset
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('SMPL Height Accuracy Evaluation', fontsize=16, fontweight='bold')

    # Extract errors
    no_arm_errors = np.array([r['height_error'] * 100 for r in no_arm_results])  # cm
    with_arm_errors = np.array([r['height_error'] * 100 for r in with_arm_results])  # cm

    no_arm_rel = np.array([r['relative_error'] for r in no_arm_results])
    with_arm_rel = np.array([r['relative_error'] for r in with_arm_results])

    # 1. Histogram of absolute errors
    ax = axes[0, 0]
    ax.hist(no_arm_errors, bins=30, alpha=0.6, label='No_Arm', color='blue', edgecolor='black')
    ax.hist(with_arm_errors, bins=30, alpha=0.6, label='With_Arm', color='green', edgecolor='black')
    ax.set_xlabel('Height Error (cm)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Height Error Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Box plot
    ax = axes[0, 1]
    bp = ax.boxplot([no_arm_errors, with_arm_errors],
                     labels=['No_Arm', 'With_Arm'],
                     patch_artist=True,
                     showmeans=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightgreen')
    ax.set_ylabel('Height Error (cm)', fontsize=11)
    ax.set_title('Height Error Box Plot', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Relative error histogram
    ax = axes[1, 0]
    ax.hist(no_arm_rel, bins=30, alpha=0.6, label='No_Arm', color='blue', edgecolor='black')
    ax.hist(with_arm_rel, bins=30, alpha=0.6, label='With_Arm', color='green', edgecolor='black')
    ax.set_xlabel('Relative Error (%)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Relative Height Error Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Statistics table
    ax = axes[1, 1]
    ax.axis('off')

    no_arm_stats = compute_statistics(no_arm_results)
    with_arm_stats = compute_statistics(with_arm_results)

    table_data = [
        ['Metric', 'No_Arm', 'With_Arm'],
        ['Count', f"{no_arm_stats['count']}", f"{with_arm_stats['count']}"],
        ['MAE (cm)', f"{no_arm_stats['mae']*100:.2f}", f"{with_arm_stats['mae']*100:.2f}"],
        ['Mean Error (cm)', f"{no_arm_stats['mean_error']*100:.2f}", f"{with_arm_stats['mean_error']*100:.2f}"],
        ['Median Error (cm)', f"{no_arm_stats['median_error']*100:.2f}", f"{with_arm_stats['median_error']*100:.2f}"],
        ['Std Error (cm)', f"{no_arm_stats['std_error']*100:.2f}", f"{with_arm_stats['std_error']*100:.2f}"],
        ['Min Error (cm)', f"{no_arm_stats['min_error']*100:.2f}", f"{with_arm_stats['min_error']*100:.2f}"],
        ['Max Error (cm)', f"{no_arm_stats['max_error']*100:.2f}", f"{with_arm_stats['max_error']*100:.2f}"],
        ['Mean Rel Error (%)', f"{no_arm_stats['mean_relative_error']:.2f}", f"{with_arm_stats['mean_relative_error']:.2f}"],
    ]

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate SMPL height accuracy')
    parser.add_argument('--results_dir', type=str,
                       default='/egr/research-zijunlab/kwonjoon/Output/output_Physica/AddB_to_SMPL/filtered_results',
                       help='Directory containing filtered results')
    parser.add_argument('--output_dir', type=str,
                       default='/egr/research-zijunlab/kwonjoon/Output/output_Physica',
                       help='Directory to save evaluation outputs')
    parser.add_argument('--smpl_model', type=str,
                       default='/egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL/models/smpl_model.pkl',
                       help='Path to SMPL model file')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device for computation (cpu or cuda)')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SMPL Height Accuracy Evaluation (Vertex-Based)")
    print("=" * 80)

    # Load SMPL model
    print(f"\nLoading SMPL model from {args.smpl_model}...")
    device = torch.device(args.device)
    try:
        smpl_model = SMPLModel(device=device, model_path=args.smpl_model)
        print("SMPL model loaded successfully")
    except Exception as e:
        print(f"Error loading SMPL model: {e}")
        return

    # Evaluate No_Arm dataset
    print("\nEvaluating No_Arm dataset...")
    no_arm_results = evaluate_dataset(results_dir / "No_Arm", smpl_model, args.device)
    no_arm_stats = compute_statistics(no_arm_results)

    print(f"\nNo_Arm Statistics:")
    print(f"  Count: {no_arm_stats['count']}")
    print(f"  MAE: {no_arm_stats['mae']*100:.2f} cm")
    print(f"  Mean Error: {no_arm_stats['mean_error']*100:.2f} cm")
    print(f"  Median Error: {no_arm_stats['median_error']*100:.2f} cm")
    print(f"  Min/Max Error: {no_arm_stats['min_error']*100:.2f} / {no_arm_stats['max_error']*100:.2f} cm")
    print(f"  Mean Relative Error: {no_arm_stats['mean_relative_error']:.2f}%")

    # Evaluate With_Arm dataset
    print("\nEvaluating With_Arm dataset...")
    with_arm_results = evaluate_dataset(results_dir / "With_Arm", smpl_model, args.device)
    with_arm_stats = compute_statistics(with_arm_results)

    print(f"\nWith_Arm Statistics:")
    print(f"  Count: {with_arm_stats['count']}")
    print(f"  MAE: {with_arm_stats['mae']*100:.2f} cm")
    print(f"  Mean Error: {with_arm_stats['mean_error']*100:.2f} cm")
    print(f"  Median Error: {with_arm_stats['median_error']*100:.2f} cm")
    print(f"  Min/Max Error: {with_arm_stats['min_error']*100:.2f} / {with_arm_stats['max_error']*100:.2f} cm")
    print(f"  Mean Relative Error: {with_arm_stats['mean_relative_error']:.2f}%")

    # Save detailed results
    output_data = {
        'no_arm': {
            'statistics': no_arm_stats,
            'per_subject': no_arm_results
        },
        'with_arm': {
            'statistics': with_arm_stats,
            'per_subject': with_arm_results
        }
    }

    # json_path = output_dir / "height_accuracy_evaluation.json"
    # with open(json_path, 'w') as f:
        # json.dump(output_data, f, indent=2)
    # print(f"\nDetailed results saved to: {json_path}")

    # Create comparison plot
    plot_path = output_dir / "height_error_comparison.png"
    plot_comparison(no_arm_results, with_arm_results, str(plot_path))

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
