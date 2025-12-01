#!/bin/bash
#
# Analyze MPJPE results from With_Arm 500-frame processing
# Shows statistics, distribution, best/worst cases
#

OUTPUT_DIR="/egr/research-zijunlab/kwonjoon/Output/output_full_Physica/With_Arm_500Frames"

echo "================================================================================"
echo "WITH_ARM 500-FRAME PROCESSING - MPJPE ANALYSIS"
echo "================================================================================"
echo ""

# Check if output directory exists
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "❌ Output directory not found: $OUTPUT_DIR"
    exit 1
fi

# Count completed subjects
TOTAL_COMPLETED=$(find "$OUTPUT_DIR" -name "meta.json" 2>/dev/null | wc -l)

if [ $TOTAL_COMPLETED -eq 0 ]; then
    echo "❌ No completed subjects found. Run the processing first!"
    exit 1
fi

echo "📊 Found $TOTAL_COMPLETED completed subjects"
echo ""

# Detailed statistics and analysis
python3 << 'EOF'
import json
from pathlib import Path
import numpy as np
from collections import defaultdict

output_dir = Path("/egr/research-zijunlab/kwonjoon/Output/output_full_Physica/With_Arm_500Frames")

# Collect all data
data = []
study_stats = defaultdict(list)

for meta_file in output_dir.glob("*/meta.json"):
    try:
        with open(meta_file) as f:
            meta = json.load(f)
            mpjpe = meta.get('metrics', {}).get('MPJPE')
            num_frames_orig = meta.get('num_frames_original', meta.get('num_frames', 0))
            num_frames_opt = meta.get('num_frames', 0)
            subject = meta_file.parent.name

            # Extract study name
            study = subject.rsplit('_', 1)[0] if '_' in subject else 'Unknown'

            if mpjpe is not None:
                data.append({
                    'subject': subject,
                    'study': study,
                    'mpjpe': mpjpe,
                    'frames_original': num_frames_orig,
                    'frames_optimized': num_frames_opt
                })
                study_stats[study].append(mpjpe)
    except Exception as e:
        pass

if not data:
    print("❌ No valid MPJPE data found")
    exit(1)

mpjpe_values = np.array([d['mpjpe'] for d in data])

# Overall statistics
print("="*80)
print("📈 OVERALL STATISTICS")
print("="*80)
print(f"Total subjects: {len(data)}")
print(f"Mean MPJPE: {np.mean(mpjpe_values):.2f} mm")
print(f"Median MPJPE: {np.median(mpjpe_values):.2f} mm")
print(f"Std Dev: {np.std(mpjpe_values):.2f} mm")
print(f"Min MPJPE: {np.min(mpjpe_values):.2f} mm")
print(f"Max MPJPE: {np.max(mpjpe_values):.2f} mm")
print(f"25th percentile: {np.percentile(mpjpe_values, 25):.2f} mm")
print(f"75th percentile: {np.percentile(mpjpe_values, 75):.2f} mm")
print()

# Distribution
print("="*80)
print("📊 MPJPE DISTRIBUTION")
print("="*80)
bins = [0, 20, 30, 40, 50, 75, 100, 150, 200, float('inf')]
labels = ['0-20mm', '20-30mm', '30-40mm', '40-50mm', '50-75mm', '75-100mm', '100-150mm', '150-200mm', '>200mm']

for i in range(len(bins)-1):
    count = sum((mpjpe_values >= bins[i]) & (mpjpe_values < bins[i+1]))
    percent = count * 100 / len(data)
    bar = '█' * int(percent / 2)  # Scale bar to 50 chars max
    print(f"{labels[i]:>12s}: {count:4d} ({percent:5.1f}%) {bar}")
print()

# Study-wise statistics
print("="*80)
print("📚 PER-STUDY STATISTICS")
print("="*80)
print(f"{'Study':<50s} {'Count':>6s} {'Mean':>8s} {'Median':>8s} {'Min':>8s} {'Max':>8s}")
print("-"*80)
for study, mpjpe_list in sorted(study_stats.items(), key=lambda x: np.mean(x[1])):
    values = np.array(mpjpe_list)
    print(f"{study:<50s} {len(values):>6d} {np.mean(values):>7.1f}  {np.median(values):>7.1f}  {np.min(values):>7.1f}  {np.max(values):>7.1f}")
print()

# Best cases (lowest MPJPE)
print("="*80)
print("🏆 TOP 10 BEST CASES (Lowest MPJPE)")
print("="*80)
sorted_data = sorted(data, key=lambda x: x['mpjpe'])
print(f"{'Subject':<60s} {'MPJPE':>10s} {'Frames':>15s}")
print("-"*80)
for i, d in enumerate(sorted_data[:10], 1):
    frames_info = f"{d['frames_original']}→{d['frames_optimized']}" if d['frames_original'] != d['frames_optimized'] else str(d['frames_optimized'])
    print(f"{i:2d}. {d['subject']:<57s} {d['mpjpe']:>7.2f} mm {frames_info:>15s}")
print()

# Worst cases (highest MPJPE)
print("="*80)
print("⚠️  TOP 10 WORST CASES (Highest MPJPE)")
print("="*80)
print(f"{'Subject':<60s} {'MPJPE':>10s} {'Frames':>15s}")
print("-"*80)
for i, d in enumerate(sorted_data[-10:][::-1], 1):
    frames_info = f"{d['frames_original']}→{d['frames_optimized']}" if d['frames_original'] != d['frames_optimized'] else str(d['frames_optimized'])
    print(f"{i:2d}. {d['subject']:<57s} {d['mpjpe']:>7.2f} mm {frames_info:>15s}")
print()

# Frame sampling statistics
print("="*80)
print("🔍 FRAME SAMPLING ANALYSIS")
print("="*80)
sampled = [d for d in data if d['frames_original'] > d['frames_optimized']]
not_sampled = [d for d in data if d['frames_original'] == d['frames_optimized']]

print(f"Sampled subjects (>500 frames): {len(sampled)}")
print(f"  Average original frames: {np.mean([d['frames_original'] for d in sampled]):.0f}")
print(f"  Average optimized frames: {np.mean([d['frames_optimized'] for d in sampled]):.0f}")
print(f"  Average MPJPE: {np.mean([d['mpjpe'] for d in sampled]):.2f} mm")
print()
print(f"Not sampled subjects (≤500 frames): {len(not_sampled)}")
print(f"  Average frames: {np.mean([d['frames_original'] for d in not_sampled]):.0f}")
print(f"  Average MPJPE: {np.mean([d['mpjpe'] for d in not_sampled]):.2f} mm")
print()

# Quality summary
print("="*80)
print("✅ QUALITY SUMMARY")
print("="*80)
excellent = sum(mpjpe_values < 30)
good = sum((mpjpe_values >= 30) & (mpjpe_values < 50))
acceptable = sum((mpjpe_values >= 50) & (mpjpe_values < 100))
poor = sum(mpjpe_values >= 100)

print(f"Excellent (<30mm):   {excellent:4d} ({excellent*100/len(data):5.1f}%)")
print(f"Good (30-50mm):      {good:4d} ({good*100/len(data):5.1f}%)")
print(f"Acceptable (50-100mm): {acceptable:4d} ({acceptable*100/len(data):5.1f}%)")
print(f"Poor (≥100mm):       {poor:4d} ({poor*100/len(data):5.1f}%)")
print()

print("="*80)
EOF

echo ""
echo "================================================================================"
echo "Analysis complete!"
echo ""
echo "💡 Tips:"
echo "  - Subjects with MPJPE >100mm may need manual inspection"
echo "  - Use --disable_frame_sampling for worst cases if needed"
echo "  - Check study-wise statistics to identify problematic datasets"
echo "================================================================================"
