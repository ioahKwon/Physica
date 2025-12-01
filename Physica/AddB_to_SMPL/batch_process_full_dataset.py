#!/usr/bin/env python3
"""
TRUE MINIBATCH PROCESSING for v3_enhanced SMPL fitting - FULL BODY + PARALLEL GPU

Key differences from batch_process_v3_true_minibatch.py:
- Full body optimization: lower_body_only=False (optimizes upper + lower body)
- GPU work distribution: Each GPU processes every Nth subject (interleaved)
- Default paths: With_Arm dataset
- Loads torch, nimblephysics, and SMPL model ONCE at startup
- Processes subjects in a SINGLE Python process per GPU
- No subprocess spawning = no repeated library loading

Usage:
    # GPU 0 processes subjects 0, 4, 8, 12, ...
    CUDA_VISIBLE_DEVICES=0 python batch_process_v3_fullbody_parallel.py --device cuda --gpu_id 0 --total_gpus 4

    # GPU 1 processes subjects 1, 5, 9, 13, ...
    CUDA_VISIBLE_DEVICES=1 python batch_process_v3_fullbody_parallel.py --device cuda --gpu_id 1 --total_gpus 4

    # GPU 2 processes subjects 2, 6, 10, 14, ...
    CUDA_VISIBLE_DEVICES=2 python batch_process_v3_fullbody_parallel.py --device cuda --gpu_id 2 --total_gpus 4

    # GPU 3 processes subjects 3, 7, 11, 15, ...
    CUDA_VISIBLE_DEVICES=3 python batch_process_v3_fullbody_parallel.py --device cuda --gpu_id 3 --total_gpus 4
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
import argparse
from typing import Dict, List

# =============================================================================
# STEP 1: Load libraries ONCE (this is the slow part - ~2 minutes)
# =============================================================================
print("=" * 80)
print("TRUE MINIBATCH PROCESSING - v3_enhanced - FULL BODY + PARALLEL")
print("=" * 80)
print()
print("⏳ [1/3] Loading libraries (torch, nimblephysics)...")
print("         This takes ~2 minutes but only happens ONCE!")
print()

lib_load_start = time.time()

# Import everything we need from v3_enhanced
sys.path.insert(0, str(Path(__file__).parent))
from addbiomechanics_to_smpl_v3_enhanced import (
    process_single_b3d,
    SMPLModel,
    OptimisationConfig,
    torch,
)

lib_load_time = time.time() - lib_load_start
print(f"✓ Libraries loaded in {lib_load_time:.1f}s")
print()


def print_progress(result: Dict, stats: Dict, total: int, gpu_id: int):
    """Print real-time progress with progress bar."""
    completed = stats['completed']
    progress = completed / total * 100 if total > 0 else 0

    # Progress bar
    bar_length = 40
    filled = int(bar_length * completed / total) if total > 0 else 0
    bar = '█' * filled + '░' * (bar_length - filled)

    # Status icon and info
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
    if stats['success'] + stats['skipped'] > 0:
        avg_mpjpe = stats['total_mpjpe'] / (stats['success'] + stats['skipped'])
        avg_str = f"Avg: {avg_mpjpe:.2f}mm"
    else:
        avg_str = "Avg: N/A"

    # ETA calculation
    if stats['success'] > 0 and completed < total:
        avg_time_per_subject = stats['total_time'] / stats['success']
        remaining_subjects = total - completed
        eta_seconds = remaining_subjects * avg_time_per_subject
        eta_str = f"ETA: {eta_seconds/3600:.1f}h"
    else:
        eta_str = ""

    # Print update with GPU ID
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[GPU{gpu_id}][{timestamp}] {icon} {result['subject'][:40]:40s} | {mpjpe_str:12s} | {time_str:8s}")
    print(f"             {bar} {progress:5.1f}% | {completed}/{total} | ✓{stats['success']} ✗{stats['failed']} ⊙{stats['skipped']} | {avg_str} | {eta_str}")
    print(flush=True)


def main():
    parser = argparse.ArgumentParser(
        description='True minibatch processing - FULL BODY + PARALLEL GPU',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # GPU 0 (processes subjects 0, 4, 8, ...)
  CUDA_VISIBLE_DEVICES=0 python %(prog)s --device cuda --gpu_id 0 --total_gpus 4

  # GPU 1 (processes subjects 1, 5, 9, ...)
  CUDA_VISIBLE_DEVICES=1 python %(prog)s --device cuda --gpu_id 1 --total_gpus 4
        """
    )
    parser.add_argument('--test_path', type=str,
                        default='/egr/research-zijunlab/kwonjoon/Dataset/AddB/test/With_Arm',
                        help='Path to test dataset')
    parser.add_argument('--train_path', type=str,
                        default='/egr/research-zijunlab/kwonjoon/Dataset/AddB/train/With_Arm',
                        help='Path to train dataset')
    parser.add_argument('--out_dir', type=str,
                        default='/egr/research-zijunlab/kwonjoon/Output/output_Physica/AddB_to_SMPL/results_v3_enhanced_minibatch_with_arm_fullbody',
                        help='Output directory')
    parser.add_argument('--smpl_model', type=str,
                        default='/egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL/models/smpl_model.pkl',
                        help='Path to SMPL model')
    parser.add_argument('--num_frames', type=int, default=64,
                        help='Number of frames to process per subject')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID for work distribution (0, 1, 2, 3)')
    parser.add_argument('--total_gpus', type=int, default=4,
                        help='Total number of GPUs (for work distribution)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of subjects (for testing)')

    args = parser.parse_args()

    # =============================================================================
    # STEP 2: Load SMPL model ONCE
    # =============================================================================
    print(f"⏳ [2/3] Loading SMPL model on GPU {args.gpu_id}...")
    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('[WARN] CUDA not available, falling back to CPU')

    smpl_load_start = time.time()
    smpl_model = SMPLModel(args.smpl_model, device=device)
    smpl_load_time = time.time() - smpl_load_start
    print(f"✓ SMPL model loaded in {smpl_load_time:.1f}s")
    print()

    # =============================================================================
    # STEP 3: Find all .b3d files and distribute work
    # =============================================================================
    print("⏳ [3/3] Finding .b3d files...")
    all_b3d_files = []

    test_path = Path(args.test_path) if args.test_path else None
    train_path = Path(args.train_path) if args.train_path else None

    if test_path and test_path.exists():
        all_b3d_files.extend(list(test_path.rglob("*.b3d")))
    if train_path and train_path.exists():
        all_b3d_files.extend(list(train_path.rglob("*.b3d")))

    # Apply limit if specified (before distribution)
    if args.limit:
        all_b3d_files = all_b3d_files[:args.limit]

    # GPU work distribution: Interleaved assignment
    # GPU 0: [0, 4, 8, 12, ...]
    # GPU 1: [1, 5, 9, 13, ...]
    # GPU 2: [2, 6, 10, 14, ...]
    # GPU 3: [3, 7, 11, 15, ...]
    b3d_files = [f for i, f in enumerate(all_b3d_files) if i % args.total_gpus == args.gpu_id]

    # Count test vs train for this GPU's subset
    test_count = sum(1 for f in b3d_files if "test" in str(f))
    train_count = len(b3d_files) - test_count

    print(f"✓ GPU {args.gpu_id} assigned {len(b3d_files)}/{len(all_b3d_files)} subjects:")
    print(f"    Test:  {test_count}")
    print(f"    Train: {train_count}")
    print()
    print(f"Output directory: {args.out_dir}")
    print(f"Device: {device}")
    print(f"Num frames/subject: {args.num_frames}")
    print(f"FULL BODY OPTIMIZATION: lower_body_only=False")
    print("=" * 80)
    print()

    # Create output directory
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # =============================================================================
    # STEP 4: Process assigned subjects sequentially (libraries already loaded!)
    # =============================================================================
    print(f"Starting batch processing on GPU {args.gpu_id}...")
    print("Libraries are loaded - processing will be MUCH faster now!")
    print("=" * 80)
    print()

    # Statistics
    stats = {
        'completed': 0,
        'success': 0,
        'failed': 0,
        'skipped': 0,
        'total_mpjpe': 0.0,
        'total_time': 0.0,
    }

    # Default config (can be customized)
    config = OptimisationConfig()

    batch_start_time = time.time()

    for idx, b3d_file in enumerate(b3d_files, 1):
        # Determine split and create output name
        subject_name = b3d_file.stem
        parent_name = b3d_file.parent.name

        if "test" in str(b3d_file):
            split = "test"
        elif "train" in str(b3d_file):
            split = "train"
        else:
            split = "unknown"

        output_name = f"{split}_{parent_name}_{subject_name}"
        subject_output_dir = output_dir / output_name

        # Check if already processed (skip)
        meta_file = subject_output_dir / "meta.json"
        if meta_file.exists():
            try:
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                    if 'MPJPE' in meta:
                        result = {
                            'subject': output_name,
                            'status': 'skipped',
                            'mpjpe': meta['MPJPE'],
                            'time': 0,
                        }
                        stats['completed'] += 1
                        stats['skipped'] += 1
                        stats['total_mpjpe'] += meta['MPJPE']
                        print_progress(result, stats, len(b3d_files), args.gpu_id)
                        continue
            except:
                pass

        # Process this subject
        subject_start = time.time()

        try:
            # Call the refactored function directly (no subprocess!)
            # KEY CHANGE: lower_body_only=False for FULL BODY optimization
            process_result = process_single_b3d(
                b3d_path=str(b3d_file),
                smpl_model=smpl_model,  # Reuse the loaded model!
                out_dir=str(subject_output_dir),
                num_frames=args.num_frames,
                device=device,
                config=config,
                lower_body_only=False,  # ← FULL BODY OPTIMIZATION!
                verbose=False  # Don't print per-subject details
            )

            subject_time = time.time() - subject_start

            result = {
                'subject': output_name,
                'status': 'success',
                'mpjpe': process_result['mpjpe'],
                'time': subject_time,
            }

            stats['completed'] += 1
            stats['success'] += 1
            stats['total_mpjpe'] += process_result['mpjpe']
            stats['total_time'] += subject_time

        except Exception as e:
            subject_time = time.time() - subject_start

            # Print full error for debugging
            import traceback
            print(f"\nFull error for {output_name}:")
            print(traceback.format_exc())

            result = {
                'subject': output_name,
                'status': 'failed',
                'error': str(e)[:100],
                'time': subject_time,
            }

            stats['completed'] += 1
            stats['failed'] += 1

        # Print progress
        print_progress(result, stats, len(b3d_files), args.gpu_id)

    # =============================================================================
    # FINAL SUMMARY
    # =============================================================================
    batch_elapsed = time.time() - batch_start_time
    total_elapsed = time.time() - lib_load_start

    print()
    print("=" * 80)
    print(f"GPU {args.gpu_id} BATCH PROCESSING COMPLETE!")
    print("=" * 80)
    print(f"Library loading time:  {lib_load_time:.1f}s ({lib_load_time/60:.1f}min)")
    print(f"SMPL model load time:  {smpl_load_time:.1f}s")
    print(f"Processing time:       {batch_elapsed:.1f}s ({batch_elapsed/3600:.2f}h)")
    print(f"Total time:            {total_elapsed:.1f}s ({total_elapsed/3600:.2f}h)")
    print()
    print(f"Subjects processed:    {stats['completed']}/{len(b3d_files)}")
    print(f"  ✓ Success:           {stats['success']}")
    print(f"  ✗ Failed:            {stats['failed']}")
    print(f"  ⊙ Skipped (cached):  {stats['skipped']}")
    print()

    if stats['success'] + stats['skipped'] > 0:
        avg_mpjpe = stats['total_mpjpe'] / (stats['success'] + stats['skipped'])
        print(f"Average MPJPE:         {avg_mpjpe:.2f}mm")

    if stats['success'] > 0:
        avg_time = stats['total_time'] / stats['success']
        print(f"Avg time per subject:  {avg_time:.1f}s ({avg_time/60:.1f}min)")
        print()
        print(f"Time saved vs subprocess approach:")
        print(f"  Subprocess: ~{(stats['success'] * (lib_load_time + avg_time))/3600:.1f}h")
        print(f"  Minibatch:  ~{total_elapsed/3600:.2f}h")
        print(f"  Savings:    ~{((stats['success'] * lib_load_time))/3600:.1f}h ({((stats['success'] * lib_load_time))/(stats['success'] * (lib_load_time + avg_time))*100:.0f}% faster!)")

    print("=" * 80)
    print(f"Results saved to: {args.out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
