#!/usr/bin/env python3
"""
Minibatch processing for v3_enhanced SMPL fitting
Loads libraries ONCE and processes multiple subjects in batches - much faster!
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
import argparse
from typing import List, Dict

# Import ONCE at the beginning (this is the slow part!)
print("Loading libraries (this takes ~2 minutes)...", flush=True)
load_start = time.time()

import numpy as np
import torch
import nimblephysics as nimble

# Import the main processing function
sys.path.insert(0, str(Path(__file__).parent))
from addbiomechanics_to_smpl_v3_enhanced import (
    load_b3d_file,
    infer_addb_joints,
    create_joint_mapping,
    optimize_smpl_betas,
    optimize_smpl_pose_translation_with_enhancements,
    save_results,
    compute_mpjpe,
    SMPLModel,
)

load_elapsed = time.time() - load_start
print(f"✓ Libraries loaded in {load_elapsed:.1f}s", flush=True)
print()


def process_single_subject(
    b3d_file: Path,
    smpl_model: SMPLModel,
    output_dir: Path,
    num_frames: int,
    device: str
) -> Dict:
    """Process a single subject (libraries already loaded)."""

    subject_name = b3d_file.stem
    parent_name = b3d_file.parent.name

    # Determine split
    if "test" in str(b3d_file):
        split = "test"
    elif "train" in str(b3d_file):
        split = "train"
    else:
        split = "unknown"

    output_name = f"{split}_{parent_name}_{subject_name}"
    subject_output_dir = output_dir / output_name

    # Check if already processed
    meta_file = subject_output_dir / "meta.json"
    if meta_file.exists():
        try:
            with open(meta_file, 'r') as f:
                meta = json.load(f)
                if 'MPJPE' in meta:
                    return {
                        'subject': output_name,
                        'status': 'skipped',
                        'mpjpe': meta['MPJPE'],
                        'time': 0,
                    }
        except:
            pass

    # Process
    start_time = time.time()

    try:
        # Load .b3d file
        subject = load_b3d_file(str(b3d_file))

        # Get ground truth data
        addb_joints_all = subject.getJointPositions()
        total_frames = addb_joints_all.shape[1]

        if total_frames < num_frames:
            num_frames = total_frames

        # Sample frames
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        addb_joints = addb_joints_all[:, frame_indices]

        # Infer joint mapping
        addb_joint_names = infer_addb_joints(subject.getSkeleton())
        joint_mapping = create_joint_mapping(addb_joint_names)

        # Optimize shape
        betas_opt = optimize_smpl_betas(
            addb_joints=addb_joints,
            joint_mapping=joint_mapping,
            smpl_model=smpl_model,
            device=device,
            num_iters=200,
            lr=0.02,
        )

        # Optimize pose + translation
        smpl_params = optimize_smpl_pose_translation_with_enhancements(
            addb_joints=addb_joints,
            betas=betas_opt,
            joint_mapping=joint_mapping,
            smpl_model=smpl_model,
            device=device,
            num_iters=300,
            base_lr=0.01,
        )

        # Compute MPJPE
        mpjpe = compute_mpjpe(
            gt_joints=addb_joints,
            smpl_joints=smpl_params['smpl_joints'],
            joint_mapping=joint_mapping,
        )

        # Save results
        subject_output_dir.mkdir(parents=True, exist_ok=True)
        save_results(
            output_dir=str(subject_output_dir),
            smpl_params=smpl_params,
            mpjpe=mpjpe,
            num_frames=num_frames,
            total_frames=total_frames,
            frame_indices=frame_indices,
            joint_mapping=joint_mapping,
        )

        elapsed = time.time() - start_time

        return {
            'subject': output_name,
            'status': 'success',
            'mpjpe': float(mpjpe),
            'time': elapsed,
        }

    except Exception as e:
        elapsed = time.time() - start_time
        return {
            'subject': output_name,
            'status': 'failed',
            'error': str(e)[:200],
            'time': elapsed,
        }


def print_progress(result: Dict, completed: int, total: int, stats: Dict):
    """Print real-time progress."""

    # Progress bar
    progress = completed / total * 100
    bar_length = 40
    filled = int(bar_length * completed / total)
    bar = '█' * filled + '░' * (bar_length - filled)

    # Status icon
    if result['status'] == 'success':
        icon = '✓'
        mpjpe_str = f"{result['mpjpe']:.2f}mm"
        time_str = f"{result['time']:.1f}s"
    elif result['status'] == 'skipped':
        icon = '⊙'
        mpjpe_str = f"{result['mpjpe']:.2f}mm"
        time_str = "cached"
    else:
        icon = '✗'
        mpjpe_str = result.get('error', 'unknown')[:30]
        time_str = f"{result.get('time', 0):.1f}s"

    # Average MPJPE
    if stats['success'] > 0:
        avg_mpjpe = stats['total_mpjpe'] / stats['success']
        avg_str = f"Avg: {avg_mpjpe:.2f}mm"
    else:
        avg_str = "Avg: N/A"

    # Print update
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {icon} {result['subject'][:40]:40s} | {mpjpe_str:30s} | {time_str:8s}")
    print(f"          {bar} {progress:5.1f}% | {completed}/{total} | ✓{stats['success']} ✗{stats['failed']} ⊙{stats['skipped']} | {avg_str}")
    print(flush=True)


def main():
    parser = argparse.ArgumentParser(description='Minibatch processing for v3_enhanced SMPL fitting')
    parser.add_argument('--test_path', type=str,
                        default='/egr/research-zijunlab/kwonjoon/Dataset/AddB/test/No_Arm',
                        help='Path to test dataset')
    parser.add_argument('--train_path', type=str,
                        default='/egr/research-zijunlab/kwonjoon/Dataset/AddB/train/No_Arm',
                        help='Path to train dataset')
    parser.add_argument('--out_dir', type=str,
                        default='/egr/research-zijunlab/kwonjoon/Output/output_Physica/AddB_to_SMPL/results_v3_minibatch',
                        help='Output directory')
    parser.add_argument('--smpl_model', type=str,
                        default='/egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL/models/smpl_model.pkl',
                        help='Path to SMPL model')
    parser.add_argument('--num_frames', type=int, default=64,
                        help='Number of frames to process')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of subjects (for testing)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Number of subjects per batch (currently only 1 supported)')

    args = parser.parse_args()

    print("=" * 80)
    print("MINIBATCH PROCESSING - v3_enhanced")
    print("=" * 80)

    # Load SMPL model (once!)
    print("Loading SMPL model...", flush=True)
    smpl_model = SMPLModel(args.smpl_model)
    print("✓ SMPL model loaded")
    print()

    # Find all .b3d files
    print("Finding .b3d files...", flush=True)
    b3d_files = []

    test_path = Path(args.test_path)
    train_path = Path(args.train_path)

    if test_path.exists():
        b3d_files.extend(list(test_path.rglob("*.b3d")))
    if train_path.exists():
        b3d_files.extend(list(train_path.rglob("*.b3d")))

    # Apply limit if specified
    if args.limit:
        b3d_files = b3d_files[:args.limit]

    # Count test vs train
    test_count = sum(1 for f in b3d_files if "test" in str(f))
    train_count = len(b3d_files) - test_count

    print(f"Found {len(b3d_files)} subjects:")
    print(f"  Test set:  {test_count}")
    print(f"  Train set: {train_count}")
    print(f"  Total:     {len(b3d_files)}")
    print()
    print(f"Output: {args.out_dir}")
    print("=" * 80)
    print()

    # Create output directory
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Statistics
    stats = {
        'completed': 0,
        'success': 0,
        'failed': 0,
        'skipped': 0,
        'total_mpjpe': 0.0,
    }

    # Process subjects sequentially (libraries already loaded!)
    start_time = time.time()

    for b3d_file in b3d_files:
        result = process_single_subject(
            b3d_file=b3d_file,
            smpl_model=smpl_model,
            output_dir=output_dir,
            num_frames=args.num_frames,
            device=args.device,
        )

        # Update stats
        stats['completed'] += 1
        if result['status'] == 'success':
            stats['success'] += 1
            stats['total_mpjpe'] += result['mpjpe']
        elif result['status'] == 'skipped':
            stats['skipped'] += 1
            stats['total_mpjpe'] += result['mpjpe']
        else:
            stats['failed'] += 1

        # Print progress
        print_progress(result, stats['completed'], len(b3d_files), stats)

    # Final summary
    elapsed = time.time() - start_time
    print()
    print("=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total time: {elapsed/3600:.2f} hours ({elapsed/60:.1f} minutes)")
    print(f"Completed:  {stats['completed']}/{len(b3d_files)}")
    print(f"  Success:  {stats['success']}")
    print(f"  Failed:   {stats['failed']}")
    print(f"  Skipped:  {stats['skipped']}")
    if stats['success'] > 0:
        avg_mpjpe = stats['total_mpjpe'] / stats['success']
        print(f"Average MPJPE: {avg_mpjpe:.2f}mm")
    print("=" * 80)


if __name__ == "__main__":
    main()
