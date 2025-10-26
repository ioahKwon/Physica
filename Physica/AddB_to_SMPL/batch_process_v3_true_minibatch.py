#!/usr/bin/env python3
"""
TRUE MINIBATCH PROCESSING for v3_enhanced SMPL fitting

Key difference from other batch scripts:
- Loads torch, nimblephysics, and SMPL model ONCE at startup
- Processes all subjects in a SINGLE Python process
- No subprocess spawning = no repeated library loading
- Expected speedup: ~50% faster (2min loading saved per subject)

Usage:
    # Process all 674 subjects
    python batch_process_v3_true_minibatch.py

    # Test with first 10 subjects
    python batch_process_v3_true_minibatch.py --limit 10

    # Process only test set
    python batch_process_v3_true_minibatch.py --train_path ""
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
print("TRUE MINIBATCH PROCESSING - v3_enhanced")
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


def print_progress(result: Dict, stats: Dict, total: int):
    """Print real-time progress with progress bar."""
    completed = stats['completed']
    progress = completed / total * 100

    # Progress bar
    bar_length = 40
    filled = int(bar_length * completed / total)
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

    # Print update
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {icon} {result['subject'][:45]:45s} | {mpjpe_str:12s} | {time_str:8s}")
    print(f"          {bar} {progress:5.1f}% | {completed}/{total} | ✓{stats['success']} ✗{stats['failed']} ⊙{stats['skipped']} | {avg_str} | {eta_str}")
    print(flush=True)


def main():
    parser = argparse.ArgumentParser(
        description='True minibatch processing - libraries loaded once',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all subjects
  python %(prog)s

  # Test with 10 subjects
  python %(prog)s --limit 10

  # Process only test set
  python %(prog)s --train_path ""
        """
    )
    parser.add_argument('--test_path', type=str,
                        default='/egr/research-zijunlab/kwonjoon/Dataset/AddB/test/No_Arm',
                        help='Path to test dataset')
    parser.add_argument('--train_path', type=str,
                        default='/egr/research-zijunlab/kwonjoon/Dataset/AddB/train/No_Arm',
                        help='Path to train dataset')
    parser.add_argument('--out_dir', type=str,
                        default='/egr/research-zijunlab/kwonjoon/Output/output_Physica/AddB_to_SMPL/results_v3_true_minibatch',
                        help='Output directory')
    parser.add_argument('--smpl_model', type=str,
                        default='/egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL/models/smpl_model.pkl',
                        help='Path to SMPL model')
    parser.add_argument('--num_frames', type=int, default=64,
                        help='Number of frames to process per subject')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of subjects (for testing)')

    args = parser.parse_args()

    # =============================================================================
    # STEP 2: Load SMPL model ONCE
    # =============================================================================
    print("⏳ [2/3] Loading SMPL model...")
    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('[WARN] CUDA not available, falling back to CPU')

    smpl_load_start = time.time()
    smpl_model = SMPLModel(args.smpl_model, device=device)
    smpl_load_time = time.time() - smpl_load_start
    print(f"✓ SMPL model loaded in {smpl_load_time:.1f}s")
    print()

    # =============================================================================
    # STEP 3: Find all .b3d files
    # =============================================================================
    print("⏳ [3/3] Finding .b3d files...")
    b3d_files = []

    test_path = Path(args.test_path) if args.test_path else None
    train_path = Path(args.train_path) if args.train_path else None

    if test_path and test_path.exists():
        b3d_files.extend(list(test_path.rglob("*.b3d")))
    if train_path and train_path.exists():
        b3d_files.extend(list(train_path.rglob("*.b3d")))

    # Apply limit if specified
    if args.limit:
        b3d_files = b3d_files[:args.limit]

    # Count test vs train
    test_count = sum(1 for f in b3d_files if "test" in str(f))
    train_count = len(b3d_files) - test_count

    print(f"✓ Found {len(b3d_files)} subjects:")
    print(f"    Test:  {test_count}")
    print(f"    Train: {train_count}")
    print()
    print(f"Output directory: {args.out_dir}")
    print(f"Device: {device}")
    print(f"Num frames/subject: {args.num_frames}")
    print("=" * 80)
    print()

    # Create output directory
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # =============================================================================
    # STEP 4: Process all subjects sequentially (libraries already loaded!)
    # =============================================================================
    print("Starting batch processing...")
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
                        print_progress(result, stats, len(b3d_files))
                        continue
            except:
                pass

        # Process this subject
        subject_start = time.time()

        try:
            # Call the refactored function directly (no subprocess!)
            process_result = process_single_b3d(
                b3d_path=str(b3d_file),
                smpl_model=smpl_model,  # Reuse the loaded model!
                out_dir=str(subject_output_dir),
                num_frames=args.num_frames,
                device=device,
                config=config,
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
        print_progress(result, stats, len(b3d_files))

    # =============================================================================
    # FINAL SUMMARY
    # =============================================================================
    batch_elapsed = time.time() - batch_start_time
    total_elapsed = time.time() - lib_load_start

    print()
    print("=" * 80)
    print("BATCH PROCESSING COMPLETE!")
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
