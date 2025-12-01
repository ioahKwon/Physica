#!/usr/bin/env python3
"""Quick test on 3 subjects."""

import sys
import os
import time

# Set up path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
# Add original Physica directory for SMPL model
sys.path.insert(0, "/egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL")

# Now import
from physica_pipeline import PhysicaPipeline
from core.config import PipelineConfig

print("="*80)
print("Physica Enhanced - Quick 3-Subject Test")
print("="*80)

# Configuration
smpl_model_path = "/egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL/models/smpl_model.pkl"
config = PipelineConfig.fast()
config.verbose = True  # Verbose for debugging

# Initialize pipeline once (reuse for all subjects)
print("\nInitializing pipeline...")
pipeline = PhysicaPipeline(
    smpl_model_path=smpl_model_path,
    config=config,
    device="cuda"
)
print("✓ Pipeline initialized")

# Test subjects - using existing files
test_subjects = [
    {
        "name": "With_Arm_Subject25",
        "b3d": "/egr/research-zijunlab/kwonjoon/Dataset/AddB/train/With_Arm/Tiziana2019_Formatted_With_Arm/Subject25/Subject25.b3d",
        "output": "/egr/research-zijunlab/kwonjoon/Output/output_Physica_enhanced/With_Arm_Subject25",
        "num_frames": 100
    },
    {
        "name": "With_Arm_Subject17",
        "b3d": "/egr/research-zijunlab/kwonjoon/Dataset/AddB/train/With_Arm/Tiziana2019_Formatted_With_Arm/Subject17/Subject17.b3d",
        "output": "/egr/research-zijunlab/kwonjoon/Output/output_Physica_enhanced/With_Arm_Subject17",
        "num_frames": 100
    },
    {
        "name": "No_Arm_Subject7",
        "b3d": "/egr/research-zijunlab/kwonjoon/Dataset/AddB/train/No_Arm/Tiziana2019_Formatted_No_Arm/Subject7/Subject7.b3d",
        "output": "/egr/research-zijunlab/kwonjoon/Output/output_Physica_enhanced/No_Arm_Subject7",
        "num_frames": 100
    }
]

results = []

print("\n" + "="*80)
print("Running tests on 3 subjects...")
print("="*80)

for i, subject in enumerate(test_subjects, 1):
    print(f"\n[{i}/3] {subject['name']}")
    print(f"  Input: {os.path.basename(subject['b3d'])}")
    print(f"  Frames: {subject['num_frames']}")

    try:
        start_time = time.time()

        result = pipeline.run(
            b3d_path=subject['b3d'],
            num_frames=subject['num_frames']
        )

        elapsed = time.time() - start_time

        pipeline.save_result(result, subject['output'])

        results.append({
            'name': subject['name'],
            'mpjpe': result.metrics['mpjpe'],
            'time': elapsed,
            'success': True,
            'mapped_joints': len(result.retargeting_result.mapped_indices),
            'synthesized_joints': len(result.retargeting_result.synthesized_indices)
        })

        print(f"  ✓ MPJPE: {result.metrics['mpjpe']:.2f} mm")
        print(f"  ✓ Time: {elapsed:.2f}s")
        print(f"  ✓ Mapped joints: {len(result.retargeting_result.mapped_indices)}")
        print(f"  ✓ Synthesized joints: {len(result.retargeting_result.synthesized_indices)}")
        print(f"  ✓ Output: {subject['output']}")

    except Exception as e:
        import traceback
        import sys
        print(f"  ✗ Failed: {e}")
        print(f"  Full traceback:")
        traceback.print_exception(*sys.exc_info())
        results.append({
            'name': subject['name'],
            'success': False,
            'error': str(e)
        })

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

for r in results:
    if r['success']:
        print(f"\n✓ {r['name']}")
        print(f"  MPJPE: {r['mpjpe']:.2f} mm")
        print(f"  Time: {r['time']:.2f}s")
        print(f"  Mapped: {r['mapped_joints']}, Synthesized: {r['synthesized_joints']}")
    else:
        print(f"\n✗ {r['name']}")
        print(f"  Error: {r['error']}")

successful = sum(1 for r in results if r['success'])
print(f"\n{'='*80}")
print(f"Results: {successful}/{len(results)} successful")
print(f"{'='*80}")

if successful == len(results):
    print("\n✓ All tests passed!")
else:
    print(f"\n⚠ {len(results) - successful} test(s) failed")

print()
