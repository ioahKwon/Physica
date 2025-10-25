#!/usr/bin/env python3
"""
Parallel batch processing for AddBiomechanics to SMPL conversion
Uses multiple GPUs to process files in parallel
"""
import os
import subprocess
import sys
from pathlib import Path
import time
from multiprocessing import Pool, Manager
import queue

def find_b3d_files(dataset_dir):
    """Find all .b3d files recursively"""
    b3d_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.b3d'):
                b3d_files.append(os.path.join(root, file))
    return sorted(b3d_files)

def is_already_processed(output_dir, b3d_file, dataset_dir):
    """Check if a .b3d file has already been processed"""
    rel_path = os.path.relpath(os.path.dirname(b3d_file), dataset_dir)
    subject_name = os.path.splitext(os.path.basename(b3d_file))[0]
    output_path = os.path.join(output_dir, rel_path, subject_name)
    smpl_params_file = os.path.join(output_path, 'smpl_params.npz')
    return os.path.exists(smpl_params_file)

def process_file(args):
    """Process a single .b3d file on a specific GPU or CPU"""
    b3d_file, dataset_dir, output_dir, smpl_model, device, worker_id, file_num, total_files, gpu_ids = args

    rel_path = os.path.relpath(os.path.dirname(b3d_file), dataset_dir)
    subject_name = os.path.splitext(os.path.basename(b3d_file))[0]
    output_path = os.path.join(output_dir, rel_path)

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Prepare environment and command based on device
    env = os.environ.copy()

    if device == 'cuda':
        # For GPU: set CUDA_VISIBLE_DEVICES to specific GPU
        gpu_id = gpu_ids[worker_id % len(gpu_ids)]
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        worker_label = f"GPU{gpu_id}"
    else:
        # For CPU: no special environment needed
        worker_label = f"CPU{worker_id}"

    # Run optimization
    cmd = [
        'python', '-u', os.path.join(script_dir, 'addbiomechanics_to_smpl_v2.py'),
        '--b3d', b3d_file,
        '--smpl_model', smpl_model,
        '--out_dir', output_path,
        '--device', device,
        '--shape_lr', '0.005',
        '--shape_iters', '150',
        '--pose_lr', '0.01',
        '--pose_iters', '100'
    ]

    start_time = time.time()

    try:
        # Print start message
        print(f"[{file_num}/{total_files}] 🔄 {worker_label}: Starting {os.path.relpath(b3d_file, dataset_dir)}", flush=True)

        # Run with timeout (no capture_output for real-time display)
        result = subprocess.run(
            cmd,
            env=env,
            check=True,
            timeout=1800  # 30 minute timeout per file
        )

        elapsed = time.time() - start_time

        print(f"[{file_num}/{total_files}] ✅ {worker_label}: {os.path.relpath(b3d_file, dataset_dir)} ({elapsed:.1f}s)", flush=True)
        return (True, b3d_file, elapsed, None)

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        error_msg = f"Timeout after {elapsed:.1f}s"
        print(f"[{file_num}/{total_files}] ⏱️  {worker_label}: {os.path.relpath(b3d_file, dataset_dir)} - {error_msg}", flush=True)
        return (False, b3d_file, elapsed, error_msg)

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        error_msg = f"Error: {e.returncode}"
        print(f"[{file_num}/{total_files}] ❌ {worker_label}: {os.path.relpath(b3d_file, dataset_dir)} - {error_msg}", flush=True)
        return (False, b3d_file, elapsed, error_msg)

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = str(e)
        print(f"[{file_num}/{total_files}] ❌ {worker_label}: {os.path.relpath(b3d_file, dataset_dir)} - {error_msg}", flush=True)
        return (False, b3d_file, elapsed, error_msg)

def main():
    if len(sys.argv) < 3:
        print("Usage: python batch_process_parallel.py <dataset_dir> <output_dir> <device> [num_workers] [gpu_ids]")
        print("")
        print("Arguments:")
        print("  device       : 'cuda' or 'cpu'")
        print("  num_workers  : Number of parallel workers (default: 2 for cuda, 32 for cpu)")
        print("  gpu_ids      : Comma-separated GPU IDs (e.g., '1,2') (only for cuda, default: '1,2')")
        print("")
        print("Examples:")
        print("  # GPU mode with 2 GPUs (GPU 1 and 2)")
        print("  python batch_process_parallel.py /path/to/AddB /path/to/output cuda 2 1,2")
        print("")
        print("  # CPU mode with 32 workers")
        print("  python batch_process_parallel.py /path/to/AddB /path/to/output cpu 32")
        sys.exit(1)

    dataset_dir = sys.argv[1]
    output_dir = sys.argv[2]
    device = sys.argv[3] if len(sys.argv) > 3 else 'cuda'

    # Set defaults based on device
    if device == 'cuda':
        default_workers = 2
        default_gpu_ids = [1, 2]
    else:
        default_workers = 32
        default_gpu_ids = []

    num_workers = int(sys.argv[4]) if len(sys.argv) > 4 else default_workers

    # Parse GPU IDs if provided
    if device == 'cuda' and len(sys.argv) > 5:
        gpu_ids = [int(x.strip()) for x in sys.argv[5].split(',')]
    else:
        gpu_ids = default_gpu_ids

    smpl_model = '/egr/research-zijunlab/kwonjoon/PhysPT/models/SMPL_NEUTRAL.pkl'

    print("="*80)
    print("Parallel Batch Processing - AddBiomechanics to SMPL")
    print("="*80)
    print(f"Dataset Dir: {dataset_dir}")
    print(f"Output Dir: {output_dir}")
    print(f"Device: {device.upper()}")
    if device == 'cuda':
        print(f"GPU IDs: {gpu_ids}")
        print(f"Num Workers: {num_workers} (using {len(gpu_ids)} GPUs)")
    else:
        print(f"Num Workers: {num_workers} (CPU cores)")
    print("="*80)
    print("")

    # Find all .b3d files
    all_b3d_files = find_b3d_files(dataset_dir)
    total_files = len(all_b3d_files)

    print(f"Found {total_files} .b3d files")

    if total_files == 0:
        print("No .b3d files found!")
        sys.exit(1)

    # Filter out already processed files
    b3d_files = []
    skipped = 0
    for b3d_file in all_b3d_files:
        if is_already_processed(output_dir, b3d_file, dataset_dir):
            skipped += 1
        else:
            b3d_files.append(b3d_file)

    remaining = len(b3d_files)

    print(f"Already processed: {skipped}")
    print(f"Remaining to process: {remaining}")
    print("")

    if remaining == 0:
        print("All files already processed!")
        sys.exit(0)

    # Prepare arguments for parallel processing
    args_list = []
    for i, b3d_file in enumerate(b3d_files):
        worker_id = i % num_workers
        file_num = i + 1 + skipped  # Actual position in total list
        args_list.append((b3d_file, dataset_dir, output_dir, smpl_model, device, worker_id, file_num, total_files, gpu_ids))

    # Process in parallel
    start_time = time.time()
    processed = 0
    failed = 0
    processing_times = []

    if device == 'cuda':
        print(f"Starting parallel processing with {num_workers} workers using GPUs {gpu_ids}...")
    else:
        print(f"Starting parallel processing with {num_workers} CPU workers...")
    print("")

    with Pool(processes=num_workers) as pool:
        results = pool.map(process_file, args_list)

    # Analyze results
    for success, b3d_file, elapsed, error in results:
        if success:
            processed += 1
            processing_times.append(elapsed)
        else:
            failed += 1

    total_time = time.time() - start_time

    print("")
    print("="*80)
    print("Parallel Batch Processing Complete!")
    print("="*80)
    print(f"Total files: {total_files}")
    print(f"Already processed (skipped): {skipped}")
    print(f"Newly processed: {processed}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time/3600:.2f} hours")

    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        print(f"Average processing time: {avg_time:.1f}s per file")
        print(f"Effective speedup: {len(processing_times) / (total_time / avg_time):.1f}x")

    print("="*80)

if __name__ == '__main__':
    main()
