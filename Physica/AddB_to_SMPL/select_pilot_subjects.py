#!/usr/bin/env python3
"""
Select pilot subjects for full-frame SMPL fitting test.
Selects 3 No_Arm + 3 With_Arm subjects with varying frame counts.
"""

import json
from pathlib import Path

# Load the survey results
survey_file = Path("/tmp/b3d_survey/b3d_frame_counts.json")
with open(survey_file, 'r') as f:
    data = json.load(f)

all_files = data['all_files']

# Separate by dataset
no_arm_files = [f for f in all_files if f['arm_type'] == 'No_Arm' and f['split'] == 'train']
with_arm_files = [f for f in all_files if f['arm_type'] == 'With_Arm' and f['split'] == 'train']

# Sort by frame count
no_arm_files.sort(key=lambda x: x['original_frames'])
with_arm_files.sort(key=lambda x: x['original_frames'])

print("=" * 80)
print("PILOT SUBJECT SELECTION")
print("=" * 80)
print()

# Select diverse subjects: short, medium, long
def select_diverse(files, count=3):
    """Select subjects with short, medium, and long sequences"""
    if len(files) < count:
        return files[:count]
    
    # Find subjects around target frame counts
    targets = [64, 500, 2000]
    selected = []
    
    for target in targets:
        # Find closest to target
        closest = min(files, key=lambda x: abs(x['original_frames'] - target))
        if closest not in selected:
            selected.append(closest)
        files = [f for f in files if f != closest]
    
    return selected[:count]

# Select No_Arm subjects
print("NO_ARM SUBJECTS (3 selected):")
print("-" * 80)
no_arm_selected = select_diverse(no_arm_files, 3)
for i, subj in enumerate(no_arm_selected, 1):
    print(f"{i}. {subj['subject']}")
    print(f"   Frames: {subj['original_frames']}")
    print(f"   Path: {subj['file_path']}")
    print()

# Select With_Arm subjects
print("WITH_ARM SUBJECTS (3 selected):")
print("-" * 80)
with_arm_selected = select_diverse(with_arm_files, 3)
for i, subj in enumerate(with_arm_selected, 1):
    print(f"{i}. {subj['subject']}")
    print(f"   Frames: {subj['original_frames']}")
    print(f"   Path: {subj['file_path']}")
    print()

# Save selection
output = {
    'no_arm': [
        {
            'subject': s['subject'],
            'frames': s['original_frames'],
            'path': s['file_path']
        } for s in no_arm_selected
    ],
    'with_arm': [
        {
            'subject': s['subject'],
            'frames': s['original_frames'],
            'path': s['file_path']
        } for s in with_arm_selected
    ]
}

output_file = Path("/egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL/pilot_subjects.json")
with open(output_file, 'w') as f:
    json.dump(output, f, indent=2)

print("=" * 80)
print(f"✓ Selection saved to: {output_file}")
print()
print("SUMMARY:")
print(f"  No_Arm: {len(no_arm_selected)} subjects, avg {sum(s['original_frames'] for s in no_arm_selected)/len(no_arm_selected):.0f} frames")
print(f"  With_Arm: {len(with_arm_selected)} subjects, avg {sum(s['original_frames'] for s in with_arm_selected)/len(with_arm_selected):.0f} frames")
print(f"  Total: 6 subjects")
print()
