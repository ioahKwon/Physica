# AddBiomechanics to SMPL Pipeline

이 파이프라인은 AddBiomechanics (.b3d) 데이터를 SMPL 파라미터로 변환합니다.

## 📋 개요

- **입력**: AddBiomechanics .b3d 파일 (subject-specific biomechanical data)
- **출력**: SMPL 파라미터 (betas, poses, trans) + 평가 메트릭 + 시각화
- **성능**: MPJPE ~44.89mm (hammer2013_subject19 기준)

## 🚀 빠른 시작

### 1. 단일 Subject 처리

```bash
cd /egr/research-zijunlab/kwonjoon/PhysPT/AddB_to_SMPL

# 기본 사용법
./run_single_subject.sh <b3d_file> <output_name> [device]

# 예시
./run_single_subject.sh \
    /egr/research-zijunlab/kwonjoon/dataset/AddB/train/With_Arm/Hammer2013_Formatted_With_Arm/subject19/subject19.b3d \
    with_arm_hammer2013_subject19 \
    cuda
```

### 2. 전체 데이터셋 배치 처리

```bash
# 기본 사용법
./run_batch_dataset.sh <dataset_dir> <output_base_dir> [device] [parallel_jobs]

# 예시: 순차 처리
./run_batch_dataset.sh \
    /egr/research-zijunlab/kwonjoon/dataset/AddB/train \
    /egr/research-zijunlab/kwonjoon/out/AddB_SMPL \
    cuda \
    1

# 예시: 병렬 처리 (4개 작업 동시 실행)
./run_batch_dataset.sh \
    /egr/research-zijunlab/kwonjoon/dataset/AddB/train \
    /egr/research-zijunlab/kwonjoon/out/AddB_SMPL \
    cuda \
    4
```

## 📂 출력 구조

```
output_dir/
├── subject_name/
│   ├── smpl_params.npz          # SMPL 파라미터 (betas, poses, trans)
│   ├── pred_joints.npy          # 예측된 joint 위치
│   ├── target_joints.npy        # Ground truth joint 위치
│   ├── meta.json                # 메타데이터 및 평가 메트릭
│   ├── optimization.log         # Optimization 로그
│   ├── visualization.log        # Visualization 로그
│   └── visualizations/          # 시각화 결과
│       ├── joint_comparison.png
│       ├── skeleton_overlay.png
│       ├── temporal_analysis.png
│       └── motion_video.mp4
```

## ⚙️ 주요 파라미터

최적화된 하이퍼파라미터 (실험을 통해 검증됨):

```python
# Shape optimization
SHAPE_LR = 0.005          # Learning rate for shape (betas)
SHAPE_ITERS = 150         # Iterations for shape optimization

# Pose optimization
POSE_LR = 0.01            # Learning rate for pose
POSE_ITERS = 100          # Iterations per frame

# Joint weights (problematic joints에 더 높은 가중치)
ANKLE_WEIGHT = 3.0        # Weight for ankle joints
SPINE_WEIGHT = 2.0        # Weight for spine joints
```

## 📊 성능 메트릭

최적의 결과 (hammer2013_subject19):
- **MPJPE**: 44.89 mm
- **PCK@50mm**: 62.48%
- **PCK@100mm**: 94.37%
- **PCK@150mm**: 100.0%

## 🔧 고급 사용법

### Python API로 직접 사용

```python
from addbiomechanics_to_smpl_v2 import AddBiomechanicsToSMPL, OptimizerConfig

# 설정
config = OptimizerConfig(
    shape_lr=0.005,
    shape_iters=150,
    pose_lr=0.01,
    pose_iters=100,
    ankle_weight=3.0,
    spine_weight=2.0
)

# 실행
fitter = AddBiomechanicsToSMPL(
    b3d_path="path/to/subject.b3d",
    smpl_model_path="path/to/SMPL_NEUTRAL.pkl",
    device="cuda",
    config=config
)

results = fitter.run()
```

### 커스텀 시각화

```python
from visualize_results import visualize_smpl_results

visualize_smpl_results(
    result_dir="path/to/results",
    output_dir="path/to/visualizations",
    generate_video=True,
    fps=30
)
```

## 🛠️ 트러블슈팅

### 1. CUDA Out of Memory
```bash
# CPU로 실행
./run_single_subject.sh input.b3d output cpu
```

### 2. 병렬 처리 실패
```bash
# GNU parallel 설치
conda install -c conda-forge parallel

# 또는 순차 처리로 변경
./run_batch_dataset.sh dataset_dir output_dir cuda 1
```

### 3. 이미 처리된 파일 스킵
배치 처리 스크립트는 `smpl_params.npz`가 존재하면 자동으로 스킵합니다.

### 4. 로그 확인
```bash
# Optimization 로그
cat output_dir/subject_name/optimization.log

# Visualization 로그
cat output_dir/subject_name/visualization.log
```

## 📝 주요 파일

- `addbiomechanics_to_smpl_v2.py` - 메인 optimization 스크립트
- `visualize_results.py` - 시각화 생성 스크립트
- `run_single_subject.sh` - 단일 subject 파이프라인
- `run_batch_dataset.sh` - 배치 처리 파이프라인
- `refine_bad_joints.py` - (실험용) Per-joint refinement
- `visualize_refined.py` - (실험용) Refinement 시각화

## 🧪 실험 히스토리

### 시도한 방법들:
1. ✅ **Original optimization** (MPJPE: 44.89mm) - **최종 채택**
2. ❌ Per-joint refinement (MPJPE: 47.74mm) - Temporal smoothness 손상
3. ❌ Temporal smoothness regularization (MPJPE: 56.54mm) - Fitting 정확도 감소

### 결론:
Original optimization이 가장 좋은 성능을 보입니다:
- 낮은 MPJPE
- 우수한 temporal smoothness (velocity correlation: 0.9379)
- 안정적인 motion quality

## 📚 참고사항

- SMPL 모델 경로: `/egr/research-zijunlab/kwonjoon/PhysPT/models/SMPL_NEUTRAL.pkl`
- AddBiomechanics 데이터: `/egr/research-zijunlab/kwonjoon/dataset/AddB/`
- 출력 디렉토리: `/egr/research-zijunlab/kwonjoon/out/AddB_SMPL/`

## 📄 License

이 프로젝트는 연구 목적으로만 사용됩니다.
