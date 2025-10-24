# 전체 데이터셋 배치 처리 정보

## 📊 데이터셋 구성

- **총 파일 개수**: 1,137개 (.b3d 파일)
  - Train: 1,072개
    - With_Arm: 435개
    - No_Arm: 637개
  - Test: 65개

## ⏱️ 예상 소요 시간

- **파일당 처리 시간**: 약 1-2분
- **전체 예상 시간**: 19-38시간

## 🚀 실행 정보

### 시작 시간
2025-10-23 08:46 KST (대략)

### 실행 명령
```bash
cd /egr/research-zijunlab/kwonjoon/PhysPT/AddB_to_SMPL
./run_batch_dataset.sh \
    /egr/research-zijunlab/kwonjoon/dataset/AddB \
    /egr/research-zijunlab/kwonjoon/out/AddB_SMPL_Full \
    cuda \
    1
```

### 백그라운드 프로세스
- **PID**: 3516280
- **로그 파일**: `/tmp/batch_processing.log`
- **출력 디렉토리**: `/egr/research-zijunlab/kwonjoon/out/AddB_SMPL_Full`

## 📈 모니터링 방법

### 1. 실시간 모니터링 스크립트 (30초마다 업데이트)
```bash
/egr/research-zijunlab/kwonjoon/PhysPT/AddB_to_SMPL/monitor_batch.sh
```

### 2. 진행 상황 체크
```bash
cd /egr/research-zijunlab/kwonjoon/PhysPT/AddB_to_SMPL
source ~/miniforge3/etc/profile.d/conda.sh
conda activate physpt
python check_progress.py /egr/research-zijunlab/kwonjoon/out/AddB_SMPL_Full
```

### 3. 로그 파일 확인
```bash
# 실시간 로그 보기
tail -f /tmp/batch_processing.log

# 최근 100줄 보기 (Warning 제외)
tail -100 /tmp/batch_processing.log | grep -v "Warning"

# 완료된 파일만 보기
grep "DONE" /tmp/batch_processing.log
```

### 4. 프로세스 상태 확인
```bash
# 프로세스가 살아있는지 확인
ps aux | grep run_batch_dataset.sh | grep -v grep

# 완료된 파일 개수 확인
find /egr/research-zijunlab/kwonjoon/out/AddB_SMPL_Full -name "smpl_params.npz" | wc -l
```

## 🗂️ 출력 구조

```
/egr/research-zijunlab/kwonjoon/out/AddB_SMPL_Full/
├── train/
│   ├── With_Arm/
│   │   ├── Tiziana2019_Formatted_With_Arm/
│   │   │   └── Subject25/
│   │   │       └── with_arm_tiziana2019_subject25/
│   │   │           ├── smpl_params.npz
│   │   │           ├── pred_joints.npy
│   │   │           ├── target_joints.npy
│   │   │           ├── meta.json
│   │   │           └── visualizations/
│   │   └── ...
│   └── No_Arm/
│       └── ...
└── test/
    └── ...
```

## ⚠️ 중요 사항

1. **프로세스가 중단되지 않도록 주의**
   - SSH 연결이 끊어져도 `nohup`으로 실행 중이므로 계속 진행됨
   - 하지만 서버 재부팅이나 강제 종료되면 중단됨

2. **이미 처리된 파일은 자동으로 스킵**
   - `smpl_params.npz`가 존재하면 스킵
   - 재시작해도 이어서 처리 가능

3. **디스크 공간 확인**
   ```bash
   df -h /egr/research-zijunlab/kwonjoon/out
   ```

4. **GPU 메모리 확인**
   ```bash
   nvidia-smi
   ```

## 📊 결과 분석 (완료 후 실행)

완료 후 다음 명령으로 전체 결과를 분석할 수 있습니다:

```bash
cd /egr/research-zijunlab/kwonjoon/PhysPT/AddB_to_SMPL
source ~/miniforge3/etc/profile.d/conda.sh
conda activate physpt

# 전체 통계
python check_progress.py /egr/research-zijunlab/kwonjoon/out/AddB_SMPL_Full

# 카테고리별 분석 (추가 분석 스크립트 필요)
python analyze_results.py /egr/research-zijunlab/kwonjoon/out/AddB_SMPL_Full
```

## 🛑 프로세스 종료 (필요시)

```bash
# PID로 종료
kill $(cat /tmp/batch_processing.pid)

# 또는 프로세스 이름으로 종료
pkill -f run_batch_dataset.sh
pkill -f addbiomechanics_to_smpl_v2.py
```

## 📝 참고

- 모든 Warning 메시지는 정상입니다 (Geometry 파일 없음)
- 각 subject마다 자동으로 visualization도 생성됩니다
- 오류 발생 시 해당 subject는 스킵되고 계속 진행됩니다
