# SMPL and AddBiomechanics OBJ Export for MeshLab Visualization

이 문서는 SMPL 메쉬와 AddBiomechanics OpenSIM 모델을 OBJ 파일로 내보내서 MeshLab에서 동시에 시각화하는 방법을 설명합니다.

## 개요

네 가지 스크립트를 제공합니다:

1. **`export_smpl_to_obj.py`**: SMPL 메쉬만 OBJ로 내보내기
2. **`export_addb_to_obj.py`**: AddBiomechanics 스켈레톤만 OBJ로 내보내기
3. **`export_comparison_to_obj.py`**: 두 모델을 동시에 내보내기 (geometric primitives)
4. **`export_comparison_with_rajagopal.py`**: SMPL + Rajagopal 해부학적 메쉬 내보내기 (✨ 추천!)

## 필요한 환경

```bash
# physpt conda 환경 활성화 (nimblephysics 필요)
conda activate physpt

# 필요한 라이브러리들은 이미 설치되어 있습니다:
# - torch
# - numpy
# - trimesh
# - nimblephysics
```

## 사용법

### 방법 1: Rajagopal 해부학적 메쉬 사용 (✨ 최고의 시각화!)

SMPL2AddBiomechanics 논문처럼 실제 해부학적 메쉬를 사용하여 시각화합니다.

```bash
cd /egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL

python export_comparison_with_rajagopal.py \
    --b3d /path/to/subject.b3d \
    --smpl_params /path/to/smpl_params.npz \
    --smpl_model models/smpl_model.pkl \
    --output_dir /path/to/output/meshlab_rajagopal \
    --start_frame 0 \
    --end_frame 10
```

**특징:**
- Rajagopal OpenSIM 해부학적 메쉬 (85,670 vertices)
- 실제 인체 형상으로 시각화
- SMPL과 직접 비교 가능
- **98.4% 이상의 정점이 올바르게 변환됨**

**결과 예시:**
- 파일 위치: `/egr/research-zijunlab/kwonjoon/meshlab_viz_rajagopal_subject29/`
- README.md 파일 포함 (자세한 기술 정보)

### 방법 2: 통합 비교 스크립트 사용 (geometric primitives)

두 모델을 동시에 내보내서 MeshLab에서 바로 비교할 수 있습니다.

```bash
cd /egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL

python export_comparison_to_obj.py \
    --b3d /path/to/subject.b3d \
    --smpl_params /path/to/smpl_params.npz \
    --smpl_model models/smpl_model.pkl \
    --output_dir /path/to/output/meshlab_comparison \
    --start_frame 0 \
    --end_frame 100
```

#### 예시: 특정 결과물 시각화

```bash
# 예시 데이터 경로
B3D_FILE="/egr/research-zijunlab/kwonjoon/Dataset/AddB/train/With_Arm/Tiziana2019_Formatted_With_Arm/Subject29/Subject29.b3d"
SMPL_PARAMS="/egr/research-zijunlab/kwonjoon/Output/output_Physica/AddB_to_SMPL/results_FIXED11_clean/no_arm_tiziana2019_subject29/smpl_params.npz"
OUTPUT_DIR="/tmp/meshlab_viz_subject29"

python export_comparison_to_obj.py \
    --b3d $B3D_FILE \
    --smpl_params $SMPL_PARAMS \
    --smpl_model models/smpl_model.pkl \
    --output_dir $OUTPUT_DIR \
    --start_frame 0 \
    --end_frame 50
```

**출력 결과:**
```
meshlab_viz_subject29/
├── smpl_frame_0000.obj  (SMPL 메쉬 - 프레임 0)
├── addb_frame_0000.obj  (AddB 스켈레톤 - 프레임 0)
├── smpl_frame_0001.obj  (SMPL 메쉬 - 프레임 1)
├── addb_frame_0001.obj  (AddB 스켈레톤 - 프레임 1)
├── ...
├── smpl_frame_0049.obj
└── addb_frame_0049.obj
```

### 방법 2: SMPL만 내보내기

```bash
python export_smpl_to_obj.py \
    --smpl_params /path/to/smpl_params.npz \
    --smpl_model models/smpl_model.pkl \
    --output_dir /path/to/output/smpl_only \
    --start_frame 0 \
    --end_frame 100 \
    --prefix smpl
```

### 방법 3: AddBiomechanics만 내보내기

```bash
python export_addb_to_obj.py \
    --b3d /path/to/subject.b3d \
    --output_dir /path/to/output/addb_only \
    --start_frame 0 \
    --end_frame 100 \
    --prefix addb \
    --joint_radius 0.015 \
    --bone_radius 0.007
```

## MeshLab에서 시각화하기

### 단일 프레임 비교

1. **MeshLab 실행**
   ```bash
   meshlab
   ```

2. **두 모델 불러오기**
   - `File` → `Import Mesh` → 다음 두 파일 선택:
     - `smpl_frame_0000.obj` (SMPL 메쉬)
     - `addb_frame_0000.obj` (AddB 스켈레톤)

3. **레이어 토글**
   - `View` → `Show Layer Dialog` (또는 `Alt+L`)
   - 각 레이어의 체크박스로 가시성 조절

4. **투명도 조절**
   - 레이어 선택 후 우클릭 → `Render Mode`
   - `Transparency` 조절

5. **정렬 확인**
   - 두 모델이 같은 좌표계에 있으므로 자동으로 정렬됨
   - 필요시 `Filters` → `Align` 메뉴 사용

### 시퀀스 애니메이션

MeshLab은 기본적으로 애니메이션을 지원하지 않으므로, 프레임별로 수동으로 불러와야 합니다.

**대안:**
- Blender 사용: OBJ 시퀀스를 애니메이션으로 불러올 수 있음
- Python 스크립트로 비디오 생성: 기존 `visualize_*.py` 스크립트 참조

## 파라미터 설명

### 공통 파라미터

- `--start_frame`: 시작 프레임 인덱스 (기본값: 0)
- `--end_frame`: 종료 프레임 인덱스 (기본값: 모든 프레임)
- `--output_dir`: OBJ 파일을 저장할 디렉토리 (필수)

### SMPL 관련 파라미터

- `--smpl_params`: `smpl_params.npz` 파일 경로 (필수)
- `--smpl_model`: SMPL 모델 파일 경로 (기본값: `models/smpl_model.pkl`)
- `--prefix`: 출력 파일 접두어 (기본값: `smpl`)

### AddBiomechanics 관련 파라미터

- `--b3d`: AddBiomechanics `.b3d` 파일 경로 (필수)
- `--joint_radius`: 관절 구의 반지름 (기본값: 0.015m)
- `--bone_radius`: 뼈 실린더의 반지름 (기본값: 0.007m)
- `--prefix`: 출력 파일 접두어 (기본값: `addb`)

## 시각화 특징

### SMPL 메쉬
- **색상**: 연한 파란색 반투명 (RGBA: 0.7, 0.8, 1.0, 0.5)
- **형태**: 6890개의 정점으로 구성된 삼각형 메쉬
- **내용**: 전신 인체 표면 메쉬

### AddBiomechanics 스켈레톤
- **관절**: 빨간색 구 (icosphere)
- **뼈**: 파란색 실린더
- **내용**: OpenSIM 모델의 관절 위치와 연결 구조

## 팁

### 프레임 수 최적화

전체 시퀀스를 내보내면 많은 파일이 생성되므로, 필요한 프레임만 선택하세요:

```bash
# 처음 50 프레임만
python export_comparison_to_obj.py ... --start_frame 0 --end_frame 50

# 특정 구간 (100-150 프레임)
python export_comparison_to_obj.py ... --start_frame 100 --end_frame 150
```

### 디스크 공간

- SMPL OBJ 파일: 약 1-2MB per frame
- AddB OBJ 파일: 약 100-500KB per frame
- 100 프레임 = 약 150-250MB

### 빠른 미리보기

먼저 소수의 프레임만 내보내서 확인:

```bash
# 처음 5 프레임만 테스트
python export_comparison_to_obj.py ... --end_frame 5
```

## 문제 해결

### 오류: "SMPL model does not contain face information"

SMPL 모델 파일이 올바르지 않거나 faces 정보가 없습니다.
- 확인: `models/smpl_model.pkl` 파일이 올바른 SMPL 모델인지 확인
- 해결: `/egr/research-zijunlab/kwonjoon/Code/Physica/models/SMPL_NEUTRAL.pkl` 사용

### 오류: "nimblephysics is required"

nimblephysics가 설치되지 않았습니다.
- 해결: `conda activate physpt` 환경 활성화

### 두 모델이 정렬되지 않음

프레임 인덱스가 일치하지 않을 수 있습니다.
- 확인: 두 모델의 프레임 수와 시작 인덱스 확인
- 해결: `--start_frame`과 `--end_frame`을 동일하게 설정

### MeshLab에서 메쉬가 보이지 않음

스케일이 너무 크거나 작을 수 있습니다.
- 해결: MeshLab에서 `View` → `View All` (단축키: `Ctrl+H`)

## 예시 워크플로우

### 1. 결과물 찾기

```bash
# 사용 가능한 결과물 확인
ls /egr/research-zijunlab/kwonjoon/Output/output_Physica/AddB_to_SMPL/results_*/
```

### 2. 해당 .b3d 파일 찾기

결과물 디렉토리 이름에서 subject 이름 확인 후 원본 .b3d 파일 찾기

### 3. OBJ 내보내기

```bash
python export_comparison_to_obj.py \
    --b3d <원본_b3d_경로> \
    --smpl_params <결과물_디렉토리>/smpl_params.npz \
    --output_dir /tmp/meshlab_viz \
    --end_frame 50
```

### 4. MeshLab에서 확인

```bash
meshlab /tmp/meshlab_viz/smpl_frame_0000.obj /tmp/meshlab_viz/addb_frame_0000.obj
```

## 추가 리소스

- **기존 시각화 스크립트**: `visualize_smpl_simple.py`, `visualize_worst_case.py`
- **SMPL 모델 문서**: `/egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL/models/smpl_model.py`
- **변환 알고리즘**: `addbiomechanics_to_smpl_v11_with_physics.py`

## 업데이트 로그

### 2025-01-17: Rajagopal 해부학적 메쉬 변환 수정 ✅

**문제:**
- Rajagopal 메쉬가 T-pose 상태로 원점에 표시됨
- 스켈레톤의 포즈를 따라가지 않음

**원인:**
- OSSO 뼈 분할(bone segmentation)의 뼈 이름이 AddBiomechanics 스켈레톤의 body node 이름과 불일치
  - 예: `thorax`, `lumbar_body` → `torso`
  - `scapula_l`, `humerus_l` 등 팔 뼈들이 No_Arm 스켈레톤에 없음

**해결:**
- 포괄적인 뼈 이름 매핑 추가:
  ```python
  bone_name_map = {
      'thorax': 'torso',
      'lumbar_body': 'torso',
      'head': 'torso',
      'scapula_l': 'torso',
      'scapula_r': 'torso',
      'humerus_l': 'torso',  # No_Arm용
      'radius_l': 'torso',
      'ulna_l': 'torso',
      'hand_l': 'torso',
      'humerus_r': 'torso',
      'radius_r': 'torso',
      'ulna_r': 'torso',
      'hand_r': 'torso',
  }
  ```

**결과:**
- ✅ 84,321 / 85,670 정점 변환 (98.4%)
- ✅ 골반이 `[-0.10, -0.03, -0.00]`에서 `[0.46, 0.82, 0.22]`로 올바르게 변환
- ✅ 모든 주요 신체 부위(골반, 다리, 상체, 팔)가 스켈레톤 포즈를 따름
- ✅ SMPL과 Rajagopal 메쉬가 동일한 좌표계와 포즈에 정렬됨

**예시 결과:**
- `/egr/research-zijunlab/kwonjoon/meshlab_viz_rajagopal_subject29/`
- README.md 파일에 자세한 기술 정보 포함

**변경된 파일:**
- `export_comparison_with_rajagopal.py` - 뼈 이름 매핑 및 디버그 기능 추가

## 연락처

문제가 있거나 개선 사항이 있으면 프로젝트 관리자에게 연락하세요.
