#!/bin/bash
#
# SMPL vs SKEL 비교 최적화 실행 스크립트
#
# Usage:
#   ./run_compare.sh <b3d_path> <output_dir> [num_frames] [num_iters]
#
# Examples:
#   # 기본 실행 (10 frames, 100 iters)
#   ./run_compare.sh /path/to/data.b3d /path/to/output
#
#   # 50 프레임, 200 iterations
#   ./run_compare.sh /path/to/data.b3d /path/to/output 50 200
#
#   # 테스트 데이터로 빠른 실행
#   ./run_compare.sh test
#

set -e

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/egr/research-zijunlab/kwonjoon/98_Dependency/01_Conda/envs/physpt/bin/python"

# Default test data
DEFAULT_B3D="/egr/research-zijunlab/kwonjoon/02_Dataset/AddB/train/With_Arm/Tiziana2019_Formatted_With_Arm/Subject1/Subject1.b3d"
DEFAULT_OUT="/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare"

# Parse arguments
if [ "$1" == "test" ]; then
    B3D_PATH="$DEFAULT_B3D"
    OUT_DIR="$DEFAULT_OUT/test_$(date +%Y%m%d_%H%M%S)"
    NUM_FRAMES="${2:-10}"
    NUM_ITERS="${3:-100}"
elif [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <b3d_path> <output_dir> [num_frames] [num_iters]"
    echo "   or: $0 test [num_frames] [num_iters]"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/data.b3d /path/to/output"
    echo "  $0 /path/to/data.b3d /path/to/output 50 200"
    echo "  $0 test                    # Quick test with default data"
    echo "  $0 test 5 50               # Very quick test"
    exit 1
else
    B3D_PATH="$1"
    OUT_DIR="$2"
    NUM_FRAMES="${3:-10}"
    NUM_ITERS="${4:-100}"
fi

# Check if b3d file exists
if [ ! -f "$B3D_PATH" ]; then
    echo "Error: B3D file not found: $B3D_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$OUT_DIR"

echo "=============================================="
echo "SMPL vs SKEL Comparison"
echo "=============================================="
echo "Input:       $B3D_PATH"
echo "Output:      $OUT_DIR"
echo "Frames:      $NUM_FRAMES"
echo "Iterations:  $NUM_ITERS"
echo "=============================================="
echo ""

# Run comparison
cd "$SCRIPT_DIR"
$PYTHON_BIN compare_smpl_skel.py \
    --b3d "$B3D_PATH" \
    --out_dir "$OUT_DIR" \
    --num_frames "$NUM_FRAMES" \
    --num_iters "$NUM_ITERS" \
    --device cuda \
    --save_every 1

echo ""
echo "=============================================="
echo "Done! Results saved to: $OUT_DIR"
echo ""
echo "Output files:"
ls -la "$OUT_DIR"
echo ""
echo "SMPL meshes:"
ls "$OUT_DIR/smpl/"*.obj 2>/dev/null | head -5 || echo "  (no OBJ files)"
echo ""
echo "SKEL meshes:"
ls "$OUT_DIR/skel/"*.obj 2>/dev/null | head -5 || echo "  (no OBJ files)"
echo "=============================================="
