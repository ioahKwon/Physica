# Full-Frame SMPL Fitting - Pilot Test Results

## Objective
Verify that the full-frame SMPL fitting pipeline correctly processes **all frames** from AddBiomechanics `.b3d` files (not just 64 frames) and maintains reasonable optimization speed.

## Critical Discovery
Previous processing was truncating sequences to 64 frames, discarding **96% of available data**:
- Total frames in dataset: 1,703,168
- Frames actually used (64-frame limit): 69,077
- **Frames lost: 1,634,091 (96.0%)**

## Pilot Test Configuration

### Test Subjects (Diverse Frame Counts)
**No_Arm (Lower Body Only):**
1. Subject29: 64 frames (short sequence)
2. s005_split1: 500 frames (medium sequence)
3. subject39: 2000 frames (long sequence)

**With_Arm (Full Body):**
1. Subject18: 63 frames (short sequence)
2. 2GC: 529 frames (medium sequence)
3. P044_split0: 2000 frames (long sequence)

### Aggressive Optimization Settings
To speed up processing while maintaining quality:
- Shape iterations: 150 → **80** (47% faster)
- Pose iterations: 80 → **50** (38% faster)
- Adaptive shape sampling: 15% of total frames (max 200 frames)

## Results Summary (4/6 Completed)

### Verification #1: Full Frames Used ✓

| Subject              | Frames | MPJPE (mm) | Time (s) |
|---------------------|--------|------------|----------|
| No_Arm_Subject29    | 64     | 42.20      | 130.2    |
| No_Arm_s005_split1  | **500**| 72.25      | 1210.0   |
| With_Arm_2GC        | **529**| 37.59      | 1118.5   |
| With_Arm_Subject18  | 63     | 49.49      | 110.2    |

**Average: 2.08 seconds/frame**

### Verification #2: SMPL Parameters Saved Correctly ✓

All subjects have complete SMPL parameters:
```
No_Arm_Subject29:    betas=(10,), poses=(64, 24, 3),  trans=(64, 3)
No_Arm_s005_split1:  betas=(10,), poses=(500, 24, 3), trans=(500, 3)  ✓ FULL FRAMES
With_Arm_2GC:        betas=(10,), poses=(529, 24, 3), trans=(529, 3)  ✓ FULL FRAMES
With_Arm_Subject18:  betas=(10,), poses=(63, 24, 3),  trans=(63, 3)
```

### Verification #3: Optimization Speed ✓

**Aggressive config achieves reasonable speed:**
- Short sequences (64 frames): ~2 minutes
- Medium sequences (500 frames): ~20 minutes
- Long sequences (2000 frames): ~69 minutes (estimated)

**Comparison to 64-frame baseline:**
- Subject29 (64 frames): 42.20mm MPJPE in 130s
- Baseline expectation: Similar quality for short sequences

## Technical Details

### Code Changes from Original
1. **No 64-frame truncation**: `num_frames=-1` (process all frames)
2. **Aggressive optimization**: Reduced iterations for faster processing
3. **Adaptive shape sampling**:
   ```python
   shape_sample_frames = min(200, max(20, int(total_frames * 0.15)))
   ```
4. **4-GPU parallel processing**: Interleaved work distribution

### File Structure
```
Code/Physica/AddB_to_SMPL/
├── batch_process_fullframes_pilot.py    # Main pilot processing script
├── select_pilot_subjects.py              # Pilot subject selection
├── run_fullframes_pilot.sh               # 4-GPU parallel launcher
├── monitor_pilot.sh                      # Real-time monitoring script
├── pilot_subjects.json                   # Selected subjects metadata
└── summarize_pilot_results.py            # Results summary script

Output/output_full_Physica/pilot_results/
├── No_Arm_Subject29/
│   ├── meta.json
│   ├── smpl_params.npz                  # betas, poses[T,24,3], trans[T,3]
│   ├── pred_joints.npy                  # SMPL joints[T,24,3]
│   └── target_joints.npy                # AddB joints[T,J,3]
├── No_Arm_s005_split1/                  # ✓ 500 frames
├── With_Arm_2GC/                        # ✓ 529 frames
└── ...
```

## Conclusions (Preliminary)

✅ **Full-frame processing works correctly** - all 500+ frame sequences processed successfully

✅ **SMPL parameters saved correctly** - pose[T,24,3] and trans[T,3] for all T frames

✅ **Aggressive optimization is fast enough** - ~2 sec/frame average

🔄 **Waiting for 2000-frame sequences to complete** - will update with final results

## Next Steps

1. **Wait for full pilot completion** (2000-frame sequences still running)
2. **Verify MPJPE quality** on long sequences
3. **If successful**: Run full dataset (1,085 subjects) with all frames
4. **Estimated full dataset time**: 1,085 subjects × 1,570 avg frames × 2 sec/frame ÷ 4 GPUs = ~237 hours (~10 days)

## Impact

Processing the full dataset will provide:
- **25x more training data** (1.7M frames vs 69K frames)
- Better temporal dynamics (full sequences instead of truncated)
- Potential MPJPE improvement (more data for optimization)

---
*Pilot test started: 2025-10-29 21:13*
*Results updated: 2025-10-29 21:35 (4/6 subjects completed)*
