#!/usr/bin/env python3
"""
Run all SMPL evaluation scripts and generate a summary report.
"""

import subprocess
import sys
from pathlib import Path

def run_evaluation(script_name: str) -> bool:
    """Run an evaluation script and return success status."""
    print(f"\n{'='*80}")
    print(f"Running {script_name}...")
    print('='*80)

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=Path(__file__).parent,
            check=True,
            capture_output=False
        )
        print(f"\n✓ {script_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {script_name} failed with error code {e.returncode}")
        return False

def main():
    print("="*80)
    print("SMPL Evaluation Suite")
    print("="*80)

    scripts = [
        'evaluate_height_accuracy.py',
        'evaluate_shape_consistency.py',
    ]

    results = {}
    for script in scripts:
        results[script] = run_evaluation(script)

    print("\n" + "="*80)
    print("Evaluation Summary")
    print("="*80)
    for script, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status}: {script}")

    print("\n" + "="*80)
    print("All evaluations complete!")
    print("="*80)

    # Exit with error if any failed
    if not all(results.values()):
        sys.exit(1)

if __name__ == "__main__":
    main()
