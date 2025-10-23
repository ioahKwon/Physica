#!/usr/bin/env python3
"""
Batch process all .b3d files in a dataset directory
"""
import os
import subprocess
import sys
from pathlib import Path
import time

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

def process_file(b3d_file, dataset_dir, output_dir, smpl_model, device):
    """Process a single .b3d file"""
    rel_path = os.path.relpath(os.path.dirname(b3d_file), dataset_dir)
    subject_name = os.path.splitext(os.path.basename(b3d_file))[0]
    output_path = os.path.join(output_dir, rel_path)

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Run optimization
    cmd = [
        'python', '-u', 'addbiomechanics_to_smpl_v2.py',
        '--b3d', b3d_file,
        '--smpl_model', smpl_model,
        '--out_dir', output_path,
        '--device', device,
        '--shape_lr', '0.005',
        '--shape_iters', '150',
        '--pose_lr', '0.01',
        '--pose_iters', '100'
    ]

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR processing {b3d_file}: {e}")
        return False

def main():
    if len(sys.argv) < 3:
        print("Usage: python batch_process_all.py <dataset_dir> <output_dir> [device]")
        print("")
        print("Example:")
        print("  python batch_process_all.py /path/to/AddB /path/to/output cuda")
        sys.exit(1)

    dataset_dir = sys.argv[1]
    output_dir = sys.argv[2]
    device = sys.argv[3] if len(sys.argv) > 3 else 'cuda'
    smpl_model = '/egr/research-zijunlab/kwonjoon/PhysPT/models/SMPL_NEUTRAL.pkl'

    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("="*80)
    print("Batch Processing - AddBiomechanics to SMPL")
    print("="*80)
    print(f"Dataset Dir: {dataset_dir}")
    print(f"Output Dir: {output_dir}")
    print(f"Device: {device}")
    print("="*80)
    print("")

    # Find all .b3d files
    b3d_files = find_b3d_files(dataset_dir)
    total_files = len(b3d_files)

    print(f"Found {total_files} .b3d files")
    print("")

    if total_files == 0:
        print("No .b3d files found!")
        sys.exit(1)

    # Process each file
    processed = 0
    skipped = 0
    failed = 0

    start_time = time.time()

    for i, b3d_file in enumerate(b3d_files, 1):
        rel_path = os.path.relpath(b3d_file, dataset_dir)

        print(f"\n[{i}/{total_files}] Processing: {rel_path}")

        # Check if already processed
        if is_already_processed(output_dir, b3d_file, dataset_dir):
            print(f"  ⏭️  Skipping (already processed)")
            skipped += 1
            continue

        # Process file
        success = process_file(b3d_file, dataset_dir, output_dir, smpl_model, device)

        if success:
            processed += 1
            print(f"  ✅ Completed ({processed} done, {total_files - i} remaining)")
        else:
            failed += 1
            print(f"  ❌ Failed")

        # Print progress
        elapsed = time.time() - start_time
        avg_time = elapsed / (processed + failed) if (processed + failed) > 0 else 0
        remaining = (total_files - i) * avg_time

        print(f"  ⏱️  Elapsed: {elapsed/3600:.1f}h, Est. remaining: {remaining/3600:.1f}h")

    print("")
    print("="*80)
    print("Batch Processing Complete!")
    print("="*80)
    print(f"Total files: {total_files}")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print(f"Total time: {(time.time() - start_time)/3600:.2f} hours")
    print("="*80)

if __name__ == '__main__':
    main()
