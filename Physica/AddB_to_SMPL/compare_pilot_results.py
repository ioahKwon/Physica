#!/usr/bin/env python3
"""
Compare baseline vs physics pilot test results
"""

import json
from pathlib import Path

baseline_dir = Path("/egr/research-zijunlab/kwonjoon/Output/output_Physica/baseline_pilot_64frames")
physics_dir = Path("/egr/research-zijunlab/kwonjoon/Output/output_Physica/physics_pilot_64frames")

# Collect baseline results from meta.json files
baseline_results = {}
for subject_dir in baseline_dir.iterdir():
    if subject_dir.is_dir():
        meta_file = subject_dir / "meta.json"
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                meta = json.load(f)
                subject_name = subject_dir.name
                baseline_results[subject_name] = {
                    'mpjpe': meta.get('MPJPE'),
                    'time': meta.get('optimization_time_seconds')
                }

# Collect physics results from meta.json files
physics_results = {}
for subject_dir in physics_dir.iterdir():
    if subject_dir.is_dir():
        meta_file = subject_dir / "meta.json"
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                meta = json.load(f)
                subject_name = subject_dir.name
                physics_results[subject_name] = {
                    'mpjpe': meta.get('MPJPE'),
                    'time': meta.get('optimization_time_seconds')
                }

print("=" * 80)
print("64-FRAME PILOT TEST: BASELINE vs PHYSICS COMPARISON")
print("=" * 80)
print()

# Per-subject comparison
print("Per-Subject Results:")
print(f"{'Subject':30s}  {'Baseline MPJPE':>12s}  {'Physics MPJPE':>12s}  {'Δ MPJPE':>10s}  {'% Change':>10s}")
print("-" * 85)

improvements = []
degradations = []
all_baseline = []
all_physics = []

for subject in sorted(baseline_results.keys()):
    if subject in physics_results:
        baseline_mpjpe = baseline_results[subject]['mpjpe']
        physics_mpjpe = physics_results[subject]['mpjpe']

        if baseline_mpjpe is not None and physics_mpjpe is not None:
            delta = physics_mpjpe - baseline_mpjpe
            pct_change = (delta / baseline_mpjpe) * 100

            all_baseline.append(baseline_mpjpe)
            all_physics.append(physics_mpjpe)

            if delta < 0:
                improvements.append(abs(delta))
                status = "✓"
            else:
                degradations.append(delta)
                status = "✗"

            print(f"{subject:30s}  {baseline_mpjpe:10.2f} mm  {physics_mpjpe:10.2f} mm  "
                  f"{delta:+9.2f} mm  {pct_change:+8.1f}%  {status}")

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

if all_baseline and all_physics:
    avg_baseline = sum(all_baseline) / len(all_baseline)
    avg_physics = sum(all_physics) / len(all_physics)
    avg_delta = avg_physics - avg_baseline
    avg_pct = (avg_delta / avg_baseline) * 100

    print(f"Average Baseline MPJPE: {avg_baseline:.2f} mm")
    print(f"Average Physics MPJPE:  {avg_physics:.2f} mm")
    print(f"Average Δ:              {avg_delta:+.2f} mm ({avg_pct:+.1f}%)")
    print()

    print(f"Subjects improved:  {len(improvements)} (avg improvement: {sum(improvements)/len(improvements):.2f} mm)" if improvements else "Subjects improved:  0")
    print(f"Subjects degraded:  {len(degradations)} (avg degradation: {sum(degradations)/len(degradations):.2f} mm)" if degradations else "Subjects degraded:  0")
    print()

    # Time comparison
    baseline_times = [baseline_results[s]['time'] for s in baseline_results if baseline_results[s]['time'] is not None]
    physics_times = [physics_results[s]['time'] for s in physics_results if physics_results[s]['time'] is not None]

    if baseline_times and physics_times:
        avg_baseline_time = sum(baseline_times) / len(baseline_times)
        avg_physics_time = sum(physics_times) / len(physics_times)
        time_delta = avg_physics_time - avg_baseline_time
        time_pct = (time_delta / avg_baseline_time) * 100

        print(f"Average Baseline Time: {avg_baseline_time:.1f}s")
        print(f"Average Physics Time:  {avg_physics_time:.1f}s")
        print(f"Time Overhead:         {time_delta:+.1f}s ({time_pct:+.1f}%)")
        print()

    # Analysis
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()

    if avg_delta > 0:
        print("⚠ Physics loss DEGRADED performance on average")
        print()
        print("Possible reasons:")
        print("  1. Physics loss weights may be too high, pulling away from joint alignment")
        print("  2. Physics data from AddBiomechanics may have noise/errors")
        print("  3. SMPL kinematic constraints may conflict with physics constraints")
        print("  4. Contact detection threshold (5cm) may be suboptimal")
        print("  5. CoM approximation (mean of joints) may be too simplistic")
        print()
        print("Recommendations:")
        print("  - Reduce physics loss weights (try 0.01, 0.005, 0.002)")
        print("  - Use physics loss only in Stage 4 (sequence enhancement)")
        print("  - Improve CoM calculation using SMPL body part masses")
        print("  - Add GRF magnitude matching instead of just contact consistency")
    else:
        print("✓ Physics loss IMPROVED performance!")
        print()
        print("The physics constraints helped align SMPL poses better with")
        print("the biomechanical ground truth data.")
