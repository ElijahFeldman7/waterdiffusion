#!/bin/bash
#SBATCH --job-name=water-diffusion
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --time=72:00:00
#SBATCH --array=0-2

BASE_DIR="/scratch/efeldma5/waterdiffusion"
PROJECT_DIR="${BASE_DIR}/waterdiffusion"
SCRIPT_PATH="${PROJECT_DIR}/scripts/waterbench_shared.py"

WATERMARK_FILE="watermarks/4_9_segmented_DWT_watermark.py"
DELTA=16.0
GEN_LENGTH=300
NUM_SAMPLES=300
NUM_SHARDS=3

RUN_NAME="remask_eval_$(date +'%m_%d_%H_%M')"
RUN_DIR="${PROJECT_DIR}/runs/${RUN_NAME}"

if [ -n "$SLURM_ARRAY_JOB_ID" ]; then
    RUN_DIR="${PROJECT_DIR}/runs/remask_eval_${SLURM_ARRAY_JOB_ID}"
fi

mkdir -p "$RUN_DIR"
mkdir -p slurm_logs

echo "Job started on $(date)"
source "${BASE_DIR}/watervenv/bin/activate"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

echo "Running Shard $SLURM_ARRAY_TASK_ID of $NUM_SHARDS"
python "$SCRIPT_PATH" \
    --shard "$SLURM_ARRAY_TASK_ID" \
    --num_shards "$NUM_SHARDS" \
    --run_dir "$RUN_DIR" \
    --watermark_path "${PROJECT_DIR}/${WATERMARK_FILE}" \
    --delta "$DELTA" \
    --gen_length "$GEN_LENGTH" \
    --num_samples "$NUM_SAMPLES"

if [ "$SLURM_ARRAY_TASK_ID" -eq $((NUM_SHARDS - 1)) ]; then
    echo "Waiting for other shards to finish before merging..."
    sleep 60 
    python "$SCRIPT_PATH" \
        --run_dir "$RUN_DIR" \
        --merge_only \
        --watermark_path "${PROJECT_DIR}/${WATERMARK_FILE}"
fi

echo "Job finished on $(date)"
