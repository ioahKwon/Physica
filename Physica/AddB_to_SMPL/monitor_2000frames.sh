#!/bin/bash
# 2000프레임 시퀀스 실시간 모니터링 (상세 버전)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

clear
echo "================================================================================"
echo "🔍 2000프레임 시퀀스 실시간 모니터링"
echo "================================================================================"
echo ""
echo "GPU 1: subject39 (No_Arm, 2000프레임)"
echo "GPU 2: P044_split0 (With_Arm, 2000프레임)"
echo ""
echo "업데이트: 5초마다 | Ctrl+C로 종료"
echo "================================================================================"
echo ""

while true; do
    clear
    echo "================================================================================"
    echo "🔍 2000프레임 시퀀스 실시간 모니터링 - $(date '+%H:%M:%S')"
    echo "================================================================================"
    echo ""

    # GPU 1 상태 (subject39)
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 GPU 1: subject39 (No_Arm, 2000프레임)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [ -f "$SCRIPT_DIR/gpu1_fullframes_pilot.log" ]; then
        # 현재 단계 확인
        if grep -q "STAGE 1" "$SCRIPT_DIR/gpu1_fullframes_pilot.log"; then
            echo "현재 상태: ✓ 데이터 로딩 완료"

            # 각 STAGE 완료 여부 확인
            if grep -q "STAGE 4" "$SCRIPT_DIR/gpu1_fullframes_pilot.log"; then
                echo "진행 단계: [✓✓✓✓] Stage 4 (최종 단계)"
                # Stage 4의 Iteration 진행상황
                stage4_iter=$(grep "Iter.*Loss" "$SCRIPT_DIR/gpu1_fullframes_pilot.log" | tail -1)
                echo "최적화: $stage4_iter"
            elif grep -q "STAGE 3" "$SCRIPT_DIR/gpu1_fullframes_pilot.log"; then
                echo "진행 단계: [✓✓✓·] Stage 3 (Pose Refinement)"
            elif grep -q "STAGE 2" "$SCRIPT_DIR/gpu1_fullframes_pilot.log"; then
                echo "진행 단계: [✓✓··] Stage 2 (Shape Optimization)"
                # Iteration 진행상황
                iter=$(grep "Iteration" "$SCRIPT_DIR/gpu1_fullframes_pilot.log" | tail -1)
                echo "최적화: $iter"
            elif grep -q "STAGE 1" "$SCRIPT_DIR/gpu1_fullframes_pilot.log"; then
                echo "진행 단계: [✓···] Stage 1 (Initial Pose)"
            fi

            # MPJPE 진행상황
            echo ""
            echo "MPJPE 진행:"
            grep "MPJPE" "$SCRIPT_DIR/gpu1_fullframes_pilot.log" | tail -5 | sed 's/^/  /'

        else
            echo "현재 상태: ⏳ 데이터 로딩 중..."
            lines=$(wc -l < "$SCRIPT_DIR/gpu1_fullframes_pilot.log")
            echo "로그 라인: $lines (geometry 파일 로딩중)"
        fi
    else
        echo "⚠️  로그 파일 없음"
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 GPU 2: P044_split0 (With_Arm, 2000프레임)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [ -f "$SCRIPT_DIR/gpu2_fullframes_pilot.log" ]; then
        # 현재 단계 확인
        if grep -q "STAGE 1" "$SCRIPT_DIR/gpu2_fullframes_pilot.log"; then
            echo "현재 상태: ✓ 데이터 로딩 완료"

            # 각 STAGE 완료 여부 확인
            if grep -q "STAGE 4" "$SCRIPT_DIR/gpu2_fullframes_pilot.log"; then
                echo "진행 단계: [✓✓✓✓] Stage 4 (최종 단계)"
                # Stage 4의 Iteration 진행상황
                stage4_iter=$(grep "Iter.*Loss" "$SCRIPT_DIR/gpu2_fullframes_pilot.log" | tail -1)
                echo "최적화: $stage4_iter"
            elif grep -q "STAGE 3" "$SCRIPT_DIR/gpu2_fullframes_pilot.log"; then
                echo "진행 단계: [✓✓✓·] Stage 3 (Pose Refinement)"
            elif grep -q "STAGE 2" "$SCRIPT_DIR/gpu2_fullframes_pilot.log"; then
                echo "진행 단계: [✓✓··] Stage 2 (Shape Optimization)"
                # Iteration 진행상황
                iter=$(grep "Iteration" "$SCRIPT_DIR/gpu2_fullframes_pilot.log" | tail -1)
                echo "최적화: $iter"
            elif grep -q "STAGE 1" "$SCRIPT_DIR/gpu2_fullframes_pilot.log"; then
                echo "진행 단계: [✓···] Stage 1 (Initial Pose)"
            fi

            # MPJPE 진행상황
            echo ""
            echo "MPJPE 진행:"
            grep "MPJPE" "$SCRIPT_DIR/gpu2_fullframes_pilot.log" | tail -5 | sed 's/^/  /'

        else
            echo "현재 상태: ⏳ 데이터 로딩 중..."
            lines=$(wc -l < "$SCRIPT_DIR/gpu2_fullframes_pilot.log")
            echo "로그 라인: $lines (geometry 파일 로딩중)"
        fi
    else
        echo "⚠️  로그 파일 없음"
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📈 완료된 피험자 현황"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    completed=$(ls /egr/research-zijunlab/kwonjoon/Output/output_full_Physica/pilot_results/*/meta.json 2>/dev/null | wc -l)
    echo "완료: $completed/6 피험자"

    if [ $completed -gt 0 ]; then
        echo ""
        printf "%-25s | %10s | %12s\n" "피험자" "프레임" "MPJPE (mm)"
        echo "───────────────────────────────────────────────────────────"
        for meta in /egr/research-zijunlab/kwonjoon/Output/output_full_Physica/pilot_results/*/meta.json; do
            if [ -f "$meta" ]; then
                name=$(basename $(dirname "$meta"))
                frames=$(grep '"num_frames"' "$meta" | head -1 | grep -oE '[0-9]+')
                mpjpe=$(grep '"MPJPE"' "$meta" | head -1 | grep -oE '[0-9]+\.[0-9]+')
                printf "%-25s | %10s | %12s\n" "$name" "$frames" "$mpjpe"
            fi
        done
    fi

    echo ""
    echo "================================================================================"
    echo "⏱️  시작 시간: 21:13 | 현재: $(date '+%H:%M:%S') | 5초마다 업데이트"
    echo "================================================================================"

    sleep 5
done
