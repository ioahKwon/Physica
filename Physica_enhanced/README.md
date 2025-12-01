# Physica Enhanced

**Fast, modular pipeline for AddBiomechanics → SMPL parameter fitting**

## 🚀 Key Features

- **Shape-first optimization**: Optimize global β over all frames first, then refine poses
- **Keyframe sampling + interpolation**: 5-10x faster pose optimization using SLERP interpolation
- **Vectorized operations**: Fully batched PyTorch implementation
- **Mixed precision**: FP16 support for 2x speedup on modern GPUs
- **torch.compile**: Optimized kernel fusion (PyTorch 2.0+)
- **Modular design**: Clean separation of concerns following Physica's style
- **Comprehensive retargeting**: Automatic joint mapping with synthesis rules

## 📦 Installation

```bash
# Install dependencies
pip install torch numpy nimblephysics

# Optional: For best performance
pip install torch --upgrade  # PyTorch 2.0+ for torch.compile
```

## 🏃 Quick Start

### CLI Usage

```bash
python -m physica_enhanced.cli.main \
    --b3d /path/to/data.b3d \
    --smpl_model /path/to/smpl_model.pkl \
    --out_dir /path/to/output \
    --preset fast
```

### Library Usage

```python
from physica_enhanced import PhysicaPipeline, PipelineConfig

# Initialize pipeline
pipeline = PhysicaPipeline(
    smpl_model_path="/path/to/smpl_model.pkl",
    config=PipelineConfig.fast(),
    device="cuda"
)

# Run on B3D file
result = pipeline.run("/path/to/data.b3d")

# Access results
print(f"MPJPE: {result.metrics['mpjpe']:.2f} mm")
print(f"Shape: {result.betas.shape}")
print(f"Poses: {result.poses.shape}")

# Save results
pipeline.save_result(result, "/path/to/output")
```

## 📁 Project Structure

```
physica_enhanced/
├── core/
│   ├── config.py              # Configuration dataclasses
│   └── smpl_model.py          # SMPL model wrapper
├── data/
│   ├── b3d_loader.py          # AddBiomechanics loader
│   └── coordinate_systems.py  # Z-up ↔ Y-up conversion
├── retargeting/
│   ├── joint_mapping.py       # Automatic joint mapping
│   ├── joint_synthesis.py     # Missing joint synthesis
│   └── retargeting_pipeline.py
├── utils/
│   ├── rotation.py            # Rotation utilities (Euler/quat→axis-angle)
│   └── interpolation.py       # Keyframe interpolation (SLERP)
├── optimization/
│   ├── losses.py              # Loss functions
│   ├── shape_optimizer.py     # Shape-first optimization
│   └── pose_optimizer.py      # Fast pose optimization
├── cli/
│   └── main.py                # Command-line interface
├── physica_pipeline.py        # Main pipeline
└── example.py                 # Usage examples
```

## ⚙️ Configuration

### Presets

- **`default`**: Balanced quality and speed (50 shape iters, 10 pose iters)
- **`fast`**: Quick iterations (30 shape iters, 5 pose iters)

### Custom Configuration

```python
from physica_enhanced.core import PipelineConfig, ShapeOptConfig, PoseOptConfig

config = PipelineConfig(
    shape_opt=ShapeOptConfig(
        max_iters=100,
        sample_frames=200,
        batch_size=64,
        use_mixed_precision=True
    ),
    pose_opt=PoseOptConfig(
        max_iters=15,
        keyframe_ratio=0.25,  # Use 25% of frames as keyframes
        interpolation_method="slerp",  # or "linear", "cubic"
        use_mixed_precision=True
    ),
    use_torch_compile=True,
    device="cuda"
)
```

## 🔬 Pipeline Stages

1. **Data Loading**: Load B3D file with automatic coordinate conversion (Z-up → Y-up)
2. **Retargeting**: Map source joints to SMPL skeleton + synthesize missing joints
3. **Shape Optimization**: Optimize global β over sampled keyframes (mini-batch SGD)
4. **Pose Optimization**: Optimize θ for keyframes only, then interpolate with SLERP
5. **Evaluation**: Compute MPJPE and other metrics

## 🎯 Joint Synthesis Rules

For missing SMPL joints not present in source data:

- **spine2**: Midpoint between spine1 and spine3
- **spine3**: Proportional extension from spine1 towards head
- **neck**: Midpoint between head and spine3
- **clavicles**: Lateral offset from spine3 towards shoulders
- **toes**: Forward extension from ankles

## 📊 Output Format

Results saved to `<out_dir>/`:

- `smpl_params.npz`: Shape (β), poses (θ), translations
- `pred_joints.npy`: Predicted SMPL joint positions [T, 24, 3]
- `target_joints.npy`: Target joint positions from source
- `meta.json`: Metrics and metadata

## 🚄 Performance Tips

1. **Use GPU**: `--device cuda` for 10-50x speedup
2. **Mixed precision**: Enabled by default on CUDA, 2x faster
3. **torch.compile**: Enable with `--use_torch_compile` (PyTorch 2.0+)
4. **Keyframe ratio**: Lower = faster (try 0.15-0.20)
5. **Batch size**: Increase if GPU memory allows

## 📝 Examples

See `example.py` for comprehensive examples:
- Basic library usage
- Batch processing multiple files
- Custom configuration
- CLI usage

## 🔧 Advanced Usage

### Custom Joint Mapping

```python
from physica_enhanced.retargeting import JointMapper

# Define manual mapping overrides
mapping_overrides = {
    0: 0,   # source joint 0 → SMPL pelvis
    5: 4,   # source joint 5 → SMPL left_knee
    # ...
}

result = pipeline.run(
    b3d_path="/path/to/data.b3d",
    mapping_overrides=mapping_overrides
)
```

### Access Intermediate Results

```python
# Access retargeting information
retarget = result.retargeting_result
print(f"Mapped joints: {retarget.mapped_indices}")
print(f"Synthesized joints: {retarget.synthesized_indices}")
print(f"Unmapped: {retarget.unmapped_source_joints}")
```

## 🐛 Troubleshooting

**CUDA out of memory:**
- Reduce batch size: `config.shape_opt.batch_size = 16`
- Reduce sample frames: `config.shape_opt.sample_frames = 50`
- Disable mixed precision: `config.shape_opt.use_mixed_precision = False`

**Slow optimization:**
- Use `--preset fast`
- Lower keyframe ratio: `--keyframe_ratio 0.15`
- Reduce iterations: `--shape_iters 30 --pose_iters 5`

**Poor fitting quality:**
- Increase iterations: `--shape_iters 100 --pose_iters 15`
- Increase keyframe ratio: `--keyframe_ratio 0.30`
- Check joint mapping (unmapped joints in meta.json)

## 📜 License

Same as original Physica repository.

## 🙏 Acknowledgments

- Original SMPL model and implementation
- AddBiomechanics dataset and nimblephysics library
- Physica project for architecture inspiration
