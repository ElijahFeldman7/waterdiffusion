#!/bin/bash
#SBATCH --job-name=zurich_eval_same_prompts
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=80GB
#SBATCH --time=08:00:00
#SBATCH --output=zurich_same_prompts_%j.log
#SBATCH --error=zurich_same_prompts_%j.err

set -euo pipefail

WORK_DIR="/scratch/efeldma5/waterdiffusion/waterdiffusion"
cd "$WORK_DIR"

SOURCE_RUN_DIR="${WORK_DIR}/runs/remask_eval_6937532"
SOURCE_JSONL="${SOURCE_RUN_DIR}/results_merged.jsonl"

RUN_NAME="zurich_eval_same_prompts_${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${WORK_DIR}/runs/${RUN_NAME}"
PROMPTS_JSONL="${OUTPUT_DIR}/prompts_from_remask_eval_6937532.jsonl"
OUTPUT_FILE="${OUTPUT_DIR}/watermark_results_same_prompts.jsonl"
DELTA_VALUES="0.5 1.0 2.0 4.0"

mkdir -p "$OUTPUT_DIR"

if [[ ! -f "$SOURCE_JSONL" ]]; then
  echo "Error: source file not found: $SOURCE_JSONL"
  exit 1
fi

echo "Building prompt dataset from: $SOURCE_JSONL"
python3 - << EOF
import json
from pathlib import Path

src = Path("$SOURCE_JSONL")
out = Path("$PROMPTS_JSONL")
seen = set()
rows = 0

with src.open() as fin, out.open("w") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        prompt = obj.get("prompt", "")
        if not prompt or prompt in seen:
            continue
        seen.add(prompt)
        fout.write(json.dumps({"input": prompt, "source_file": "remask_eval_6937532"}) + "\n")
        rows += 1

print(f"Wrote {rows} unique prompts to {out}")
EOF

NUM_SAMPLES=$(wc -l < "$PROMPTS_JSONL" | tr -d ' ')

echo "=========================================="
echo "Zurich Eval On Same Prompts"
echo "=========================================="
echo "Source run: $SOURCE_RUN_DIR"
echo "Prompt dataset: $PROMPTS_JSONL"
echo "Output: $OUTPUT_FILE"
echo "Num samples: $NUM_SAMPLES"
echo "Delta values: $DELTA_VALUES"
echo "=========================================="

python3 run_zurich_simple.py \
  --datasets "$PROMPTS_JSONL" \
  --output "$OUTPUT_FILE" \
  --num_samples "$NUM_SAMPLES" \
  --delta_values $DELTA_VALUES

echo ""
echo "=========================================="
echo "Evaluation complete"
echo "Results saved to: $OUTPUT_FILE"
echo "=========================================="

python3 << EOF
import pandas as pd

output_file = "$OUTPUT_FILE"
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
EOF
