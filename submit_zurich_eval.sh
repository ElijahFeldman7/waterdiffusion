#!/bin/bash
#SBATCH --job-name=zurich_watermark_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=80GB
#SBATCH --time=08:00:00
#SBATCH --output=zurich_watermark_%j.log
#SBATCH --error=zurich_watermark_%j.err

# Zurich Watermark Evaluation on LLADA-8B-Instruct
# Runs on cluster with 300 samples per delta value

set -e

WORK_DIR="/scratch/efeldma5/waterdiffusion/waterdiffusion"
cd "$WORK_DIR"

# Configuration
DATASET="${WORK_DIR}/zurich/data/WaterBench/2-1_longform_qa.jsonl"
OUTPUT_DIR="${WORK_DIR}/results"
OUTPUT_FILE="${OUTPUT_DIR}/watermark_results_eli5_300samples.jsonl"
NUM_SAMPLES=300
DELTA_VALUES="0.5 1.0 2.0 4.0"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Zurich Watermark Evaluation"
echo "=========================================="
echo "Working directory: $WORK_DIR"
echo "Dataset: $DATASET"
echo "Output file: $OUTPUT_FILE"
echo "Number of samples per delta: $NUM_SAMPLES"
echo "Delta values: $DELTA_VALUES"
echo "=========================================="

# Run the evaluation
python3 run_zurich_simple.py \
  --dataset "$DATASET" \
  --output "$OUTPUT_FILE" \
  --num_samples $NUM_SAMPLES \
  --delta_values $DELTA_VALUES

echo ""
echo "=========================================="
echo "✓ Evaluation complete!"
echo "Results saved to: $OUTPUT_FILE"
echo "=========================================="

# Summary statistics
echo ""
echo "Results Summary:"
python3 << 'EOF'
import pandas as pd
import sys

try:
    output_file = sys.argv[1] if len(sys.argv) > 1 else "results/watermark_results_eli5_300samples.jsonl"
    df = pd.read_json(output_file, lines=True)
    
    print(f"\nTotal rows: {len(df)}")
    print(f"\nSamples per delta:")
    print(df.groupby("delta").size())
    print(f"\nDetection metrics by delta:")
    summary = df.groupby("delta").agg({
        "det_wm_z_score": ["mean", "std"],
        "det_no_wm_z_score": ["mean", "std"],
        "tpr_at_1pct": "first"
    })
    print(summary)
except Exception as e:
    print(f"Could not generate summary: {e}")
EOF "$OUTPUT_FILE"
