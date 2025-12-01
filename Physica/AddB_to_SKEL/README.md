# AddB to SKEL Shoulder Correction

Shoulder joint definition correction for AddBiomechanics → SKEL optimization pipeline.

## Problem

AddBiomechanics and SKEL have different shoulder joint definitions:

| Model | Shoulder Joint | Location |
|-------|---------------|----------|
| **AddBiomechanics** | `acromial_r/l` (12, 16) | Acromion - outermost shoulder tip |
| **SKEL** | `scapula_r/l` (14, 19) | Scapula - close to spine (medial) |
| | `humerus_r/l` (15, 20) | Shoulder-arm joint (GH joint) |

Current mapping `acromial → humerus` causes SKEL to appear "narrow-shouldered" because humerus is more medial than acromial.

## Solution: 3-Step Shoulder Correction

### Step 1: Virtual Acromial Joint (Experimental)
Compute virtual acromial from mesh vertices during optimization.
- No model modification required
- Good for initial experiments

### Step 2: Extended J_regressor (Production)
Add 2 new measurement joints (acromial_r, acromial_l) to SKEL's joint regressor.
- New indices: 24 (acromial_r), 25 (acromial_l)
- These are "measurement-only" joints (not part of FK chain)

### Step 3: Width/Direction Constraints
Additional losses to ensure shoulder alignment:
- `L_width`: Match shoulder width between models
- `L_direction`: Match thorax→acromial direction

## Files

| File | Description |
|------|-------------|
| `shoulder_correction.py` | All loss functions (virtual acromial, width, direction) |
| `create_acromial_regressor.py` | Script to extend J_regressor from 24→26 joints |
| `integration_example.py` | Example integration with optimization loop |

## Acromion Vertex Indices

Standard SMPL/SKEL acromion landmarks (used by SMPLify-X, ExPose):

```python
# Left acromion cluster
left = [3321, 3325, 3290, 3340]  # main: 3321

# Right acromion cluster
right = [5624, 5630, 5690, 5700]  # main: 5624
```

## Usage

### Step 1: Virtual Joint Approach

```python
from shoulder_correction import compute_shoulder_losses

# In your optimization loop:
shoulder_losses = compute_shoulder_losses(
    skel_vertices=skel_vertices,  # [B, 6890, 3]
    skel_joints=skel_joints,      # [B, 24, 3]
    addb_joints=addb_joints,      # [B, 20, 3]
    lambda_acromial=1.0,
    lambda_width=1.0,
    lambda_direction=0.5,
)

loss_total = (
    loss_joint
    + loss_bone_length
    + shoulder_losses['loss_total']  # Add this
)
```

### Step 2: Extended Regressor Approach

```bash
# 1. Create extended regressor
python create_acromial_regressor.py \
    --skel_path /path/to/skel_models \
    --output ./J_regressor_osim_acromial.pkl \
    --verify
```

```python
# 2. Load SKEL with extended regressor
from skel import SKEL

skel = SKEL(
    model_path='/path/to/skel_models',
    custom_joint_reg_path='./J_regressor_osim_acromial.pkl'
)

# Now skel_joints has shape [B, 26, 3] with:
# - Index 24: acromial_r
# - Index 25: acromial_l
```

```python
# 3. Use extended losses
from shoulder_correction import compute_shoulder_losses_extended

shoulder_losses = compute_shoulder_losses_extended(
    skel_joints=skel_joints,  # [B, 26, 3]
    addb_joints=addb_joints,  # [B, 20, 3]
)
```

## Loss Functions

### L_acromial
```
L_acromial = MSE(virtual_acromial_skel, acromial_addb)
```
Matches the SKEL virtual acromial to AddB acromial position.

### L_width
```
L_width = (||skel_acr_R - skel_acr_L|| - ||addb_acr_R - addb_acr_L||)²
```
Ensures shoulder width matches between models.

### L_direction
```
L_direction = (1 - cos_sim(skel_dir, addb_dir))
```
Ensures thorax→acromial direction matches AddB back→acromial direction.

## Joint Indices Reference

### AddBiomechanics (20 joints)
```
12: acromial_r (right shoulder)
16: acromial_l (left shoulder)
11: torso (back)
```

### SKEL Original (24 joints)
```
12: thorax
14: scapula_r
15: humerus_r
19: scapula_l
20: humerus_l
```

### SKEL Extended (26 joints)
```
24: acromial_r (NEW - measurement only)
25: acromial_l (NEW - measurement only)
```

## Recommended Lambda Values

```python
lambda_acromial = 1.0    # Primary: acromial position matching
lambda_width = 1.0       # Secondary: shoulder width
lambda_direction = 0.5   # Tertiary: direction constraint
```

Start with these values and adjust based on results.
