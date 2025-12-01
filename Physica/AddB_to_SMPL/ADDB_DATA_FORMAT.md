# AddBiomechanics (AddB) Data Format - Detailed Documentation

## Overview

AddBiomechanics provides biomechanical motion capture data in `.b3d` format with two skeletal configurations:
- **No_Arm**: 637 subjects (lower body + spine, ~12 joints)
- **With_Arm**: 435 subjects (full body, ~20 joints)

**Current work focuses on With_Arm configuration for full-body SMPL fitting.**

---

## 1. Joint/Marker Information (With_Arm Configuration)

### Total Joints: 20

AddBiomechanics provides **20 joints per subject** representing key anatomical landmarks:

```
Index  Joint Name          Body Part        Description
─────────────────────────────────────────────────────────────────
0      ground_pelvis       Pelvis           Root joint (ground contact)
1      hip_r               Right leg        Right hip joint
2      walker_knee_r       Right leg        Right knee joint
3      ankle_r             Right leg        Right ankle joint
4      subtalar_r          Right leg        Right subtalar joint (foot)
5      mtp_r               Right leg        Right metatarsophalangeal (toes)
6      hip_l               Left leg         Left hip joint
7      walker_knee_l       Left leg         Left knee joint
8      ankle_l             Left leg         Left ankle joint
9      subtalar_l          Left leg         Left subtalar joint (foot)
10     mtp_l               Left leg         Left metatarsophalangeal (toes)
11     back                Spine            Lower spine/torso
12     acromial_r          Right arm        Right shoulder (acromion)
13     elbow_r             Right arm        Right elbow joint
14     radioulnar_r        Right arm        Right forearm rotation joint
15     radius_hand_r       Right arm        Right wrist (radial side)
16     acromial_l          Left arm         Left shoulder (acromion)
17     elbow_l             Left arm         Left elbow joint
18     radioulnar_l        Left arm         Left forearm rotation joint
19     radius_hand_l       Left arm         Left wrist (radial side)
```

### Body Part Distribution

**Lower body (12 joints)** - same as No_Arm configuration:
- 1× Pelvis (root)
- 2× Hips (left/right)
- 2× Knees (left/right)
- 2× Ankles (left/right)
- 2× Feet (subtalar joints)
- 2× Toes (metatarsophalangeal)
- 1× Spine (back/torso)

**Upper body (8 joints)** - additional in With_Arm:
- 2× Shoulders (acromial joints)
- 2× Elbows
- 2× Forearm rotation (radioulnar)
- 2× Wrists (radius_hand)

**Note**: Some datasets may include additional hand joints (fingers, thumb), but these are not consistently available and excluded from SMPL mapping.

---

## 2. Joint Mapping to SMPL

**Key Point**: Not all AddB joints are mapped to SMPL!

- **AddBiomechanics**: 20 joints provided
- **SMPL**: 24 joints in model
- **Mapped**: **16 joints** used for optimization

### Mapping Table

```
AddB Index  AddB Joint Name      → SMPL Index  SMPL Joint Name
────────────────────────────────────────────────────────────────
0           ground_pelvis        → 0            pelvis
1           hip_r                → 2            right_hip
2           walker_knee_r        → 5            right_knee
3           ankle_r              → 8            right_ankle
4           subtalar_r           → 11           right_foot
6           hip_l                → 1            left_hip
7           walker_knee_l        → 4            left_knee
8           ankle_l              → 7            left_ankle
9           subtalar_l           → 10           left_foot
11          back                 → 3            spine1
12          acromial_r           → 17           right_shoulder
13          elbow_r              → 19           right_elbow
15          radius_hand_r        → 21           right_wrist
16          acromial_l           → 16           left_shoulder
17          elbow_l              → 18           left_elbow
19          radius_hand_l        → 20           left_wrist
```

### Unmapped Joints

**AddB joints NOT mapped to SMPL** (4 joints):
- `mtp_r` (index 5) - Right toes
- `mtp_l` (index 10) - Left toes
- `radioulnar_r` (index 14) - Right forearm rotation (intermediate joint)
- `radioulnar_l` (index 18) - Left forearm rotation (intermediate joint)

**Reason**: SMPL doesn't have detailed toe joints or intermediate forearm rotation joints.

**SMPL joints NOT present in AddB** (8 joints):
- Neck (SMPL index 12)
- Head (SMPL index 15)
- Left/Right hands (SMPL indices 22, 23)
- Left/Right toes (SMPL indices 14, 13)
- Additional spine segments

**Reason**: AddB skeleton has different hierarchy and granularity.

---

## 3. MPJPE Computation

**IMPORTANT**: MPJPE (Mean Per Joint Position Error) is computed **only on the 16 mapped joints**, not all 20 AddB joints or all 24 SMPL joints.

```python
# Only compute error on mapped joints
pred_subset = smpl_joints[mapped_smpl_indices]     # Shape: [16, 3]
target_subset = addb_joints[mapped_addb_indices]   # Shape: [16, 3]

# L2 distance per joint
errors = ||pred_subset - target_subset||_2          # Shape: [16]

# Mean over joints
MPJPE = mean(errors)  # Scalar in millimeters
```

This ensures fair comparison by only evaluating **biomechanically meaningful matched joints**.

---

## 4. Dataset Composition (With_Arm)

### Studies and Subject Counts

| Dataset | Subjects | Motion Type | Frames Range |
|---------|----------|-------------|--------------|
| **Carter2023** | 315 | Treadmill walking (various speeds) | 2000 |
| **Tiziana2019** | 49 | Movement analysis | 500-1000 |
| **Han2023** | 29 | Treadmill running | 2000 |
| **Moore2015** | 11 | Walking variations | 400-800 |
| **Falisse2017** | 10 | Treadmill walking | 250-300 |
| **Hammer2013** | 9 | Overground walking | 200-300 |
| **vanderZee2022** | 9 | Walking patterns | 400-600 |
| **Fregly2012** | 3 | Gait analysis | 500-650 |
| **Total** | **435** | | |

### Motion Types Summary

**Primary motions in With_Arm dataset**:
1. **Walking** (~360 subjects, 83%)
   - Treadmill walking: Carter2023 (315), Falisse2017 (10)
   - Overground walking: Hammer2013 (9), Moore2015 (11), vanderZee2022 (9)
   - Gait analysis: Fregly2012 (3)

2. **Running** (~29 subjects, 7%)
   - Treadmill running: Han2023 (29)

3. **Other movements** (~46 subjects, 10%)
   - General movement analysis: Tiziana2019 (49)

**Characteristics**:
- Predominantly **locomotion data** (walking, running)
- Variety of speeds (slow walk to running)
- Both treadmill and overground conditions
- Focus on lower-limb biomechanics with upper-body context

---

## 5. Sequence Characteristics

### Frame Counts

**Distribution**:
- **Short sequences** (200-500 frames): ~100 subjects
  - Hammer2013, Falisse2017
  - ~1-2 seconds at 200Hz sampling

- **Medium sequences** (500-1000 frames): ~60 subjects
  - Fregly2012, Moore2015, vanderZee2022, Tiziana2019
  - ~2-5 seconds

- **Long sequences** (2000 frames): ~275 subjects
  - Carter2023, Han2023
  - ~8-10 seconds at 250Hz sampling

**Most common**: 2000 frames (63% of subjects)

### Sampling Rates

- **Typical**: 200-250 Hz
- **Frame duration**: 0.004-0.005 seconds per frame
- **Sequence duration**: 1-10 seconds of motion

---

## 6. Data Format Details

### File Structure

```
.b3d file format (binary HDF5-based):
- Joint positions [T, N, 3] where T=frames, N=20 joints
- Joint velocities [T, N, 3]
- Joint accelerations [T, N, 3]
- Ground reaction forces (if available)
- Marker positions (raw motion capture)
- Body segment parameters
- Metadata (subject demographics, trial info)
```

### Coordinate System

- **Units**: Meters (converted to millimeters for MPJPE)
- **Origin**: Ground plane (pelvis is "ground_pelvis")
- **Axes**:
  - X: forward/backward
  - Y: up/down (vertical)
  - Z: left/right (mediolateral)

---

## 7. Data Quality

### Joint Position Quality

**High quality** (well-tracked):
- Pelvis, hips, knees, ankles: Core locomotion joints
- Shoulders, elbows: Upper body major joints

**Variable quality** (depending on dataset):
- Feet (subtalar): Sometimes affected by ground contact
- Wrists: Can have noise during arm swing
- Toes (mtp): Often low quality, hence unmapped

**Missing data handling**:
- Some frames may have NaN (missing) values for certain joints
- Our code filters out NaN frames during optimization

---

## 8. Comparison: No_Arm vs With_Arm

| Feature | No_Arm | With_Arm |
|---------|--------|----------|
| **Subjects** | 637 | 435 |
| **Joints** | ~12 | ~20 |
| **Body parts** | Lower body + spine | Full body |
| **Mapped to SMPL** | ~10 joints | ~16 joints |
| **Primary use** | Gait analysis | Full-body motion |
| **Datasets** | Lower-limb focused studies | Studies with upper-body data |

**Why two configurations?**
- Some biomechanics studies only track lower body (gait labs)
- Upper body tracking adds complexity and cost
- Many gait analysis applications don't need arms

---

## 9. Technical Specifications

### Per-Subject Data

```python
subject = SubjectOnDisk("subject.b3d")
skeleton = subject.readSkel(0)

# Properties:
- skeleton.getNumJoints() → 20 joints
- skeleton.getNumFrames() → 200-2000 frames
- skeleton.getJointPositions(frame) → [20, 3] array
```

### Joint Hierarchy

```
ground_pelvis (root)
├── back (spine)
├── hip_r
│   └── walker_knee_r
│       └── ankle_r
│           ├── subtalar_r (foot)
│           └── mtp_r (toes)
├── hip_l
│   └── walker_knee_l
│       └── ankle_l
│           ├── subtalar_l (foot)
│           └── mtp_l (toes)
├── acromial_r (shoulder)
│   └── elbow_r
│       └── radioulnar_r (forearm)
│           └── radius_hand_r (wrist)
└── acromial_l (shoulder)
    └── elbow_l
        └── radioulnar_l (forearm)
            └── radius_hand_l (wrist)
```

---

## 10. Usage in Our Work

### Current Processing

**Dataset**: With_Arm (435 subjects)
**Objective**: Fit SMPL body model to AddB joint positions
**Optimization**: Minimize L2 distance on 16 mapped joints
**Output**: SMPL parameters (shape betas, pose parameters, translation)

### Processing Strategy

1. **Fast batch** (91 subjects, non-Carter/Han):
   - Datasets: Hammer, Falisse, Fregly, Moore, vanderZee, Tiziana
   - Frame range: 200-1000 frames
   - Expected time: 3-5 hours
   - Purpose: Quick validation of optimized parameters

2. **Slow batch** (344 subjects, Carter/Han):
   - Datasets: Carter2023, Han2023
   - Frame range: 2000 frames (long sequences)
   - Expected time: 15-20 hours
   - Purpose: Process majority of dataset

### Performance Targets

- **MPJPE**: 35-45mm (current results: 28-82mm, avg ~42mm)
- **Speed**: <15 min/subject (optimized parameters)
- **Completion**: 24 hours for all 435 subjects

---

## Summary

**Question**: What is the AddB data format?

**Answer**:
- **20 joints per subject** in With_Arm configuration
- **Body part distribution**: 12 lower-body joints + 8 upper-body joints
- **Mapped to SMPL**: 16 out of 20 joints used for optimization
- **Motions**: Predominantly walking (83%), with running (7%) and other movements
- **Datasets**: 8 studies with 435 subjects total
- **Sequence lengths**: Mostly 2000 frames (~8 seconds), some 200-1000 frames
- **Primary focus**: Locomotion biomechanics with full-body kinematics

This provides comprehensive biomechanical motion data suitable for SMPL body model fitting and human motion analysis.

---

**Last Updated**: 2025-10-30
**Data Source**: AddBiomechanics Dataset v3.0
**Configuration**: With_Arm (full body)
