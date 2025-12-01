#!/usr/bin/env python3
"""
FULL-FRAME SMPL FITTING - PILOT TEST

Key changes from batch_process_v3_fullbody_parallel.py:
- NO 64-frame truncation: processes ALL frames (num_frames=-1 default)
- Aggressive optimization: Reduced iterations for faster processing
  * shape_iters: 150 → 80
  * pose_iters: 80 → 50  
  * shape_sample_frames: adaptive (15% of total frames, max 200)
- Pilot mode: Processes only 6 selected subjects (3 No_Arm + 3 With_Arm)
- Output: /egr/research-zijunlab/kwonjoon/Output/output_full_Physica

This is a PILOT TEST to verify full-frame processing works correctly
before running on the entire dataset (1,085 subjects).

Usage:
    # GPU 0
    CUDA_VISIBLE_DEVICES=0 python batch_process_fullframes_pilot.py --device cuda --gpu_id 0 --total_gpus 4
    
    # GPU 1
    CUDA_VISIBLE_DEVICES=1 python batch_process_fullframes_pilot.py --device cuda --gpu_id 1 --total_gpus 4
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
import argparse
from typing import Dict, List

# =============================================================================
# STEP 1: Load libraries ONCE
# =============================================================================
print("=" * 80)
print("FULL-FRAME SMPL FITTING - PILOT TEST (NO 64-FRAME LIMIT)")
print("=" * 80)
print()
print("⏳ [1/4] Loading libraries (torch, nimblephysics)...")
print()

lib_load_start = time.time()

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


def load_pilot_subjects():
    """Load the 6 pilot subjects selected for testing"""
    pilot_file = Path(__file__).parent / "pilot_subjects.json"
    with open(pilot_file, 'r') as f:
        data = json.load(f)
    
    subjects = []
    for subj in data['no_arm']:
        subjects.append({
            'name': subj['subject'],
            'path': subj['path'],
            'frames': subj['frames'],
            'arm_type': 'No_Arm'
        })
    for subj in data['with_arm']:
        subjects.append({
            'name': subj['subject'],
            'path': subj['path'],
            'frames': subj['frames'],
            'arm_type': 'With_Arm'
        })
    
    return subjects


def print_progress(result: Dict, stats: Dict, total: int, gpu_id: int):
    """Print real-time progress"""
    completed = stats['completed']
    progress = completed / total * 100 if total > 0 else 0

    # Progress bar
    bar_length = 40
    filled = int(bar_length * completed / total) if total > 0 else 0
    bar = '█' * filled + '░' * (bar_length - filled)

    # Status
    if result['status'] == 'success':
        icon = '✓'
        mpjpe_str = f"{result['mpjpe']:.2f}mm"
        frames_str = f"{result.get('frames', 0)} frames"
        time_str = f"{result['time']:.1f}s"
    elif result['status'] == 'skipped':
        icon = '⊙'
        mpjpe_str = f"{result['mpjpe']:.2f}mm"
        frames_str = f"{result.get('frames', 0)} frames"
        time_str = "cached"
    else:
        icon = '✗'
        mpjpe_str = result.get('error', 'unknown')[:30]
        frames_str = ""
        time_str = f"{result.get('time', 0):.1f}s"

    # Average MPJPE
    if stats['success'] + stats['skipped'] > 0:
        avg_mpjpe = stats['total_mpjpe'] / (stats['success'] + stats['skipped'])
        avg_str = f"Avg: {avg_mpjpe:.2f}mm"
    else:
        avg_str = "Avg: N/A"

    # Print update
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[GPU{gpu_id}][{timestamp}] {icon} {result['subject'][:35]:35s} | {frames_str:12s} | {mpjpe_str:12s} | {time_str:8s}")
    print(f"             {bar} {progress:5.1f}% | {completed}/{total} | ✓{stats['success']} ✗{stats['failed']} ⊙{stats['skipped']} | {avg_str}")
    print(flush=True)


def main():
    parser = argparse.ArgumentParser(
        description='Full-frame SMPL fitting - PILOT TEST (6 subjects)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--out_dir', type=str,
                        default='/egr/research-zijunlab/kwonjoon/Output/output_full_Physica/pilot_results',
                        help='Output directory')
    parser.add_argument('--smpl_model', type=str,
                        default='/egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL/models/smpl_model.pkl',
                        help='Path to SMPL model')
    parser.add_argument('--num_frames', type=int, default=-1,
                        help='Number of frames (-1 = ALL frames, no truncation!)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID for work distribution (0, 1, 2, 3)')
    parser.add_argument('--total_gpus', type=int, default=4,
                        help='Total number of GPUs')

    args = parser.parse_args()

    # =============================================================================
    # STEP 2: Load SMPL model ONCE
    # =============================================================================
    print(f"⏳ [2/4] Loading SMPL model on GPU {args.gpu_id}...")
    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    
    smpl_load_start = time.time()
    smpl_model = SMPLModel(args.smpl_model, device=device)
    smpl_load_time = time.time() - smpl_load_start
    print(f"✓ SMPL model loaded in {smpl_load_time:.1f}s")
    print()

    # =============================================================================
    # STEP 3: Load pilot subjects and distribute to GPUs
    # =============================================================================
    print("⏳ [3/4] Loading pilot subjects...")
    all_subjects = load_pilot_subjects()
    
    # GPU work distribution: Interleaved
    my_subjects = [s for i, s in enumerate(all_subjects) if i % args.total_gpus == args.gpu_id]
    
    print(f"✓ GPU {args.gpu_id} assigned {len(my_subjects)}/{len(all_subjects)} subjects:")
    for subj in my_subjects:
        print(f"    {subj['name']}: {subj['frames']} frames ({subj['arm_type']})")
    print()
    print(f"Output directory: {args.out_dir}")
    print(f"Device: {device}")
    print(f"Num frames: {args.num_frames} (ALL FRAMES - NO TRUNCATION!)")
    print(f"AGGRESSIVE OPTIMIZATION: shape=80 iter, pose=50 iter")
    print("=" * 80)
    print()

    # Create output directory
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # =============================================================================
    # STEP 4: Process subjects with AGGRESSIVE optimization
    # =============================================================================
    print(f"⏳ [4/4] Starting full-frame processing on GPU {args.gpu_id}...")
    print("=" * 80)
    print()

    stats = {
        'completed': 0,
        'success': 0,
        'failed': 0,
        'skipped': 0,
        'total_mpjpe': 0.0,
        'total_time': 0.0,
        'total_frames': 0,
    }

    # AGGRESSIVE config for speed
    config = OptimisationConfig(
        shape_iters=80,         # 150 → 80 (47% faster)
        pose_iters=50,          # 80 → 50 (38% faster)
        pose_subsample_rate=48,  # Stage 3: 48x speedup (subsample every 48th frame)
        sequence_enhancement_iters=25,  # Stage 4: 50 → 25 iterations (50% faster)
        sequence_subsample_rate=4,  # Stage 4: subsample every 4th frame (75% faster)
        sequence_batch_size=200,  # Stage 4: mini-batch SGD (60% faster)
        shape_sample_frames=200  # Max 200 frames for shape (will be adjusted per subject)
    )

    batch_start_time = time.time()

    for idx, subject in enumerate(my_subjects, 1):
        b3d_path = Path(subject['path'])
        subject_name = subject['name']
        expected_frames = subject['frames']
        
        output_name = f"{subject['arm_type']}_{subject_name}"
        subject_output_dir = output_dir / output_name

        # Check if already processed
        meta_file = subject_output_dir / "meta.json"
        if meta_file.exists():
            try:
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                    if 'MPJPE' in meta and 'num_frames' in meta:
                        result = {
                            'subject': output_name,
                            'status': 'skipped',
                            'mpjpe': meta['MPJPE'],
                            'frames': meta['num_frames'],
                            'time': 0,
                        }
                        stats['completed'] += 1
                        stats['skipped'] += 1
                        stats['total_mpjpe'] += meta['MPJPE']
                        stats['total_frames'] += meta['num_frames']
                        print_progress(result, stats, len(my_subjects), args.gpu_id)
                        continue
            except:
                pass

        # Process this subject
        subject_start = time.time()

        try:
            # Adaptive shape sampling: 15% of frames, max 200
            adaptive_config = OptimisationConfig(
                shape_iters=80,
                pose_iters=50,
                pose_subsample_rate=48,  # Stage 3: 48x speedup
                sequence_enhancement_iters=25,  # Stage 4: 25 iterations
                sequence_subsample_rate=4,  # Stage 4: subsample rate 4
                sequence_batch_size=200,  # Stage 4: batch size 200
                shape_sample_frames=min(200, max(20, int(expected_frames * 0.15)))
            )
            
            process_result = process_single_b3d(
                b3d_path=str(b3d_path),
                smpl_model=smpl_model,
                out_dir=str(subject_output_dir),
                num_frames=args.num_frames,  # -1 = ALL frames!
                device=device,
                config=adaptive_config,
                lower_body_only=False,  # Full body
                verbose=False
            )

            subject_time = time.time() - subject_start
            actual_frames = process_result.get('num_frames', expected_frames)

            result = {
                'subject': output_name,
                'status': 'success',
                'mpjpe': process_result['mpjpe'],
                'frames': actual_frames,
                'time': subject_time,
            }

            stats['completed'] += 1
            stats['success'] += 1
            stats['total_mpjpe'] += process_result['mpjpe']
            stats['total_time'] += subject_time
            stats['total_frames'] += actual_frames

        except Exception as e:
            subject_time = time.time() - subject_start

            import traceback
            print(f"\nError for {output_name}:")
            print(traceback.format_exc())

            result = {
                'subject': output_name,
                'status': 'failed',
                'error': str(e)[:100],
                'time': subject_time,
            }

            stats['completed'] += 1
            stats['failed'] += 1

        print_progress(result, stats, len(my_subjects), args.gpu_id)

    # =============================================================================
    # FINAL SUMMARY
    # =============================================================================
    batch_elapsed = time.time() - batch_start_time
    total_elapsed = time.time() - lib_load_start

    print()
    print("=" * 80)
    print(f"GPU {args.gpu_id} PILOT TEST COMPLETE!")
    print("=" * 80)
    print(f"Library loading:       {lib_load_time:.1f}s")
    print(f"SMPL model loading:    {smpl_load_time:.1f}s")
    print(f"Processing time:       {batch_elapsed:.1f}s ({batch_elapsed/3600:.2f}h)")
    print(f"Total time:            {total_elapsed:.1f}s ({total_elapsed/3600:.2f}h)")
    print()
    print(f"Subjects processed:    {stats['completed']}/{len(my_subjects)}")
    print(f"  ✓ Success:           {stats['success']}")
    print(f"  ✗ Failed:            {stats['failed']}")
    print(f"  ⊙ Skipped:           {stats['skipped']}")
    print()
    print(f"Total frames processed: {stats['total_frames']:,}")

    if stats['success'] + stats['skipped'] > 0:
        avg_mpjpe = stats['total_mpjpe'] / (stats['success'] + stats['skipped'])
        avg_frames = stats['total_frames'] / (stats['success'] + stats['skipped'])
        print(f"Average MPJPE:         {avg_mpjpe:.2f}mm")
        print(f"Average frames/subj:   {avg_frames:.0f}")

    if stats['success'] > 0:
        avg_time = stats['total_time'] / stats['success']
        print(f"Avg time/subject:      {avg_time:.1f}s ({avg_time/60:.1f}min)")

    print("=" * 80)
    print(f"Results saved to: {args.out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
