# Joint Mapping Documentation: SMPL ↔ AddBiomechanics

## 1. SMPL Joint Names (24 joints)

SMPL uses a hierarchical skeleton with 24 joints:

| Index | Joint Name | Parent | Description |
|-------|------------|--------|-------------|
| 0 | pelvis | - | Root joint |
| 1 | left_hip | 0 | Left hip |
| 2 | right_hip | 0 | Right hip |
| 3 | spine1 | 0 | Lower spine |
| 4 | left_knee | 1 | Left knee |
| 5 | right_knee | 2 | Right knee |
| 6 | spine2 | 3 | Mid spine |
| 7 | left_ankle | 4 | Left ankle |
| 8 | right_ankle | 5 | Right ankle |
| 9 | spine3 | 6 | Upper spine |
| 10 | left_foot | 7 | Left foot |
| 11 | right_foot | 8 | Right foot |
| 12 | neck | 9 | Neck base |
| 13 | left_collar | 9 | Left collar |
| 14 | right_collar | 9 | Right collar |
| 15 | head | 12 | Head |
| 16 | left_shoulder | 13 | Left shoulder |
| 17 | right_shoulder | 14 | Right shoulder |
| 18 | left_elbow | 16 | Left elbow |
| 19 | right_elbow | 17 | Right elbow |
| 20 | left_wrist | 18 | Left wrist |
| 21 | right_wrist | 19 | Right wrist |
| 22 | left_hand | 20 | Left hand |
| 23 | right_hand | 21 | Right hand |

### SMPL Spine Hierarchy:
```
pelvis (0)
  └─ spine1 (3)
      └─ spine2 (6)
          └─ spine3 (9)
              └─ neck (12)
                  └─ head (15)
```

**Key Feature**: SMPL has **5 spine/torso joints** (spine1, spine2, spine3, neck, head)


## 2. AddBiomechanics Joint Names (20 joints)

AddBiomechanics data from .b3d files contains 20 joints:

| Index | Joint Name | Description |
|-------|------------|-------------|
| 0 | ground_pelvis | Pelvis (root) |
| 1 | hip_r | Right hip |
| 2 | walker_knee_r | Right knee |
| 3 | ankle_r | Right ankle |
| 4 | subtalar_r | Right subtalar (ankle sub-joint) |
| 5 | mtp_r | Right metatarsophalangeal (toe) |
| 6 | hip_l | Left hip |
| 7 | walker_knee_l | Left knee |
| 8 | ankle_l | Left ankle |
| 9 | subtalar_l | Left subtalar (ankle sub-joint) |
| 10 | mtp_l | Left metatarsophalangeal (toe) |
| 11 | back | **Single torso/spine joint** |
| 12 | acromial_r | Right shoulder (acromial joint) |
| 13 | elbow_r | Right elbow |
| 14 | radius_hand_r | Right wrist (radial) |
| 15 | hand_r | Right hand |
| 16 | acromial_l | Left shoulder (acromial joint) |
| 17 | elbow_l | Left elbow |
| 18 | radius_hand_l | Left wrist (radial) |
| 19 | hand_l | Left hand |

**Key Feature**: AddBiomechanics has **only 1 torso/spine joint** called 'back' (index 11)


## 3. Joint Name Aliases in AddBiomechanics

Some joint names in .b3d files have aliases that refer to the **same physical joint**:

| Primary Name | Aliases | Physical Joint |
|--------------|---------|----------------|
| back | torso, spine | Joint index 11 |

**Important**: 'back', 'torso', and 'spine' are **three different names** for the **same single joint** in AddBiomechanics data.


## 4. Current Joint Mapping (FIX 8)

### FIX 8 spine2 version:
```python
JOINT_MAPPING = {
    # Pelvis
    'ground_pelvis': 'pelvis',           # AddB[0] → SMPL[0]

    # Right Leg
    'hip_r': 'right_hip',                # AddB[1] → SMPL[2]
    'walker_knee_r': 'right_knee',       # AddB[2] → SMPL[5]
    'ankle_r': 'right_ankle',            # AddB[3] → SMPL[8]

    # Left Leg
    'hip_l': 'left_hip',                 # AddB[6] → SMPL[1]
    'walker_knee_l': 'left_knee',        # AddB[7] → SMPL[4]
    'ankle_l': 'left_ankle',             # AddB[8] → SMPL[7]

    # Spine (RESTORED in FIX 8)
    'back': 'spine2',                    # AddB[11] → SMPL[6]
    'torso': 'spine2',                   # AddB[11] → SMPL[6] (alias)
    'spine': 'spine2',                   # AddB[11] → SMPL[6] (alias)

    # Right Arm
    'acromial_r': 'right_shoulder',      # AddB[12] → SMPL[17]
    'elbow_r': 'right_elbow',            # AddB[13] → SMPL[19]
    'radius_hand_r': 'right_wrist',      # AddB[14] → SMPL[21]

    # Left Arm
    'acromial_l': 'left_shoulder',       # AddB[16] → SMPL[16]
    'elbow_l': 'left_elbow',             # AddB[17] → SMPL[18]
    'radius_hand_l': 'left_wrist',       # AddB[18] → SMPL[20]
}
```

### FIX 8 spine3 version (alternative):
Same as above, except:
```python
    # Spine (alternative mapping)
    'back': 'spine3',                    # AddB[11] → SMPL[9]
    'torso': 'spine3',                   # AddB[11] → SMPL[9] (alias)
    'spine': 'spine3',                   # AddB[11] → SMPL[9] (alias)
```


## 5. Unmapped Joints

### SMPL joints with NO AddBiomechanics correspondence:
- **spine1** (SMPL[3]) - Lower spine, no AddB equivalent
- **spine3** (SMPL[9]) - Upper spine, unmapped in spine2 version
- **spine2** (SMPL[6]) - Mid spine, unmapped in spine3 version
- **left_foot** (SMPL[10]) - Left foot end, no AddB equivalent
- **right_foot** (SMPL[11]) - Right foot end, no AddB equivalent
- **neck** (SMPL[12]) - Neck, no AddB equivalent
- **left_collar** (SMPL[13]) - Left collar, no AddB equivalent
- **right_collar** (SMPL[14]) - Right collar, no AddB equivalent
- **head** (SMPL[15]) - Head, no AddB equivalent
- **left_hand** (SMPL[22]) - Left hand end, no AddB equivalent
- **right_hand** (SMPL[23]) - Right hand end, no AddB equivalent

These joints are estimated through SMPL's kinematic model and optimization, not directly supervised by AddBiomechanics data.

### AddBiomechanics joints NOT mapped to SMPL:
- **subtalar_r** (AddB[4]) - Right subtalar, no SMPL equivalent
- **mtp_r** (AddB[5]) - Right toe, no SMPL equivalent
- **subtalar_l** (AddB[9]) - Left subtalar, no SMPL equivalent
- **mtp_l** (AddB[10]) - Left toe, no SMPL equivalent
- **hand_r** (AddB[15]) - Right hand end, not used (wrist is enough)
- **hand_l** (AddB[19]) - Left hand end, not used (wrist is enough)


## 6. Key Insights from Analysis

### Spine Mapping Investigation (FIX 7 → FIX 8):

**Problem in FIX 7**:
- Incorrectly assumed AddB 'back' joint was below pelvis
- Removed all spine mapping
- Result: 21.7° spine forward lean, MPJPE = 19.10mm

**Analysis Results** (P020_split5, 200 frames):
```
AddB 'back' joint position relative to pelvis:
  Mean Y offset: +0.106m (10.6cm ABOVE pelvis, not below!)

3D distance to SMPL spine joints:
  spine1: 24.8cm (closest in 16% of frames)
  spine2: 16.2cm (closest in 61% of frames) ← BEST MATCH
  spine3: 20.6cm (closest in 23% of frames)
```

**Conclusion**:
- AddB 'back' joint is closest to SMPL **spine2** (mid-back)
- Provides crucial upper body supervision to prevent forward lean
- FIX 8 tests both spine2 and spine3 mappings


## 7. Mapping Strategy

### Why Not Subdivide AddB's Single Spine Joint?

The user asked: "back과 torso,spine을 spine2에 하나로 매핑하는게 맞을까? 더 세분화해야하는거 아냐?"
(Is it right to map back/torso/spine all to spine2? Should we subdivide?)

**Answer**: We **cannot** subdivide because:
1. 'back', 'torso', 'spine' are **aliases** - they refer to the **same physical joint** (index 11)
2. AddBiomechanics only captures **1 torso position**, not multiple spine positions
3. SMPL's 5 spine joints must be inferred from this single supervision point

### Optimization Approach:
1. **Direct supervision**: AddB 'back' joint → SMPL spine2 (or spine3)
2. **Indirect inference**: spine1, spine3, neck, head estimated through:
   - Kinematic constraints (parent-child relationships)
   - Shape parameters (betas)
   - Pose parameters (joint rotations)
   - Physics-based losses (foot contact, symmetry)


## 8. Visual Comparison

### SMPL Skeleton:
```
          head (15)
            |
          neck (12)
            |
         spine3 (9) ─┬─ left_collar (13) ─ left_shoulder (16) ─ ...
            |        └─ right_collar (14) ─ right_shoulder (17) ─ ...
         spine2 (6)  ← [FIX 8 spine2: AddB 'back' maps here]
            |
         spine1 (3)
            |
         pelvis (0) ─┬─ left_hip (1) ─ ...
                     └─ right_hip (2) ─ ...
```

### AddBiomechanics Skeleton:
```
          (no head)
            |
          back (11)  ← [Single torso joint]
         /    |    \
    acromial  |  acromial
     (L/R)    |   (L/R)
              |
      ground_pelvis (0)
           /     \
        hip_l   hip_r
        (6)      (1)
```


## 9. Testing in Progress

**Current Status**: Testing two spine mapping alternatives on P020_split5:

| Version | Mapping | Output Directory | GPU |
|---------|---------|------------------|-----|
| v8_spine2 | back → spine2 | `/tmp/v8_spine2_p020_split5/` | GPU 0 |
| v8_spine3 | back → spine3 | `/tmp/v8_spine3_p020_split5/` | GPU 1 |

**Expected Outcomes**:
- Both should eliminate the 21.7° forward lean seen in FIX 7
- spine2 mapping likely better (closest 3D distance)
- spine3 mapping may provide better shoulder alignment
- Will compare MPJPE, spine angle, and visual quality


## 10. Historical Context

| Version | Spine Mapping | Result |
|---------|---------------|--------|
| FIX 7 (v7) | ❌ REMOVED (wrong assumption) | MPJPE: 19.10mm, **21.7° forward lean** |
| FIX 8 spine2 (v8) | ✅ back → spine2 | Testing... |
| FIX 8 spine3 (v8) | ✅ back → spine3 | Testing... |

---

**Created**: 2025-11-06
**Purpose**: Document joint name mappings for SMPL ↔ AddBiomechanics optimization
**Related Files**:
- `addbiomechanics_to_smpl_v8_spine2.py`
- `addbiomechanics_to_smpl_v8_spine3.py`
- `addbiomechanics_to_smpl_v7_no_spine.py`
