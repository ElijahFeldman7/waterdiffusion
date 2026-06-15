#!/bin/bash
#SBATCH --job-name=ogt-watermark
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --time=72:00:00
#SBATCH --array=0-6

module load hosts/hopper gnu/12.3.0
module load cuda/12.4.0
module load ollama/0.20.3
module load python/3.12.1-33

BASE_DIR="/scratch/efeldma5/waterdiffusion"
PROJECT_DIR="${BASE_DIR}/waterdiffusion"
SCRIPT_PATH="${PROJECT_DIR}/scripts/waterbench_shared.py"
WATERMARK_FILE="watermarks/proposal_5_ogt.py"

# Deltas (KL budgets) used for KL-budgeted algorithms like Prop 3 and Prop 5
DELTAS=(0.02 0.05 0.10 0.20 0.40 0.80 1.50)
DELTA=${DELTAS[$SLURM_ARRAY_TASK_ID]}
DELTA_DIR="delta_${DELTA}"

GEN_LENGTH=300
NUM_SAMPLES=300

if [ -n "$SLURM_ARRAY_JOB_ID" ]; then
    PARENT_DIR="${PROJECT_DIR}/runs/prompt5_${SLURM_ARRAY_JOB_ID}"
else
    PARENT_DIR="${PROJECT_DIR}/runs/prompt5_$(date +'%m_%d_%H_%M')"
fi

RUN_DIR="${PARENT_DIR}/${DELTA_DIR}"
mkdir -p "$RUN_DIR"
mkdir -p slurm_logs

echo "Job started on $(date)"
echo "Array task ${SLURM_ARRAY_TASK_ID} | Delta (KL): ${DELTA} | Dir: ${RUN_DIR}"

source "${BASE_DIR}/watervenv/bin/activate"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

# 1. Generate text and run GPT-2 PPL / Detection
echo "Running Generation..."
python "$SCRIPT_PATH" \
    --shard 0 \
    --num_shards 1 \
    --run_dir "$RUN_DIR" \
    --watermark_path "${PROJECT_DIR}/${WATERMARK_FILE}" \
    --delta "$DELTA" \
    --gen_length "$GEN_LENGTH" \
    --num_samples "$NUM_SAMPLES"

# Merge (just cleans up and creates results_merged.jsonl)
python "$SCRIPT_PATH" \
    --run_dir "$RUN_DIR" \
    --merge_only \
    --watermark_path "${PROJECT_DIR}/${WATERMARK_FILE}"

# 2. Re-evaluate PPL with Qwen 32B on the generated output
echo "Running Qwen-32B PPL Eval..."
QWEN_SCRIPT="${PROJECT_DIR}/scripts/reevaluate_ppl_qwen.py"
python "$QWEN_SCRIPT" \
    --input "$RUN_DIR/results_merged.jsonl" \
    --batch-size 4 \
    --model "Qwen/Qwen2.5-32B-Instruct"

echo "Job finished on $(date)"
