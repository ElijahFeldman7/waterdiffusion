#!/bin/bash
#SBATCH --job-name=water-diffusion-zurich
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

set -euo pipefail

BASE_DIR="${BASE_DIR:-/scratch/efeldma5/waterdiffusion}"
PROJECT_DIR="${PROJECT_DIR:-${BASE_DIR}/waterdiffusion}"
SCRIPT_PATH="${PROJECT_DIR}/scripts/waterbench_shared.py"

WATERMARK_FILE="watermarks/zurich_diffusion_adapter.py"
DELTA="${DELTA:-2.0}"
GEN_LENGTH="${GEN_LENGTH:-300}"
GEN_STEPS="${GEN_STEPS:-300}"
BLOCK_LENGTH="${BLOCK_LENGTH:-25}"
NUM_SAMPLES="${NUM_SAMPLES:-300}"
NUM_SHARDS="${NUM_SHARDS:-3}"

RUN_DIR="${PROJECT_DIR}/runs/remask_eval_${SLURM_ARRAY_JOB_ID}"

mkdir -p "${RUN_DIR}"
mkdir -p "${PROJECT_DIR}/slurm_logs"

echo "Job started on $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Host: $(hostname)"

source "${BASE_DIR}/watervenv/bin/activate"
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/zurich/src:${PYTHONPATH:-}"

echo "Generating JSONL shard ${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS} with Zurich diffusion watermark"
python "${SCRIPT_PATH}" \
    --shard "${SLURM_ARRAY_TASK_ID}" \
    --num_shards "${NUM_SHARDS}" \
    --run_dir "${RUN_DIR}" \
    --watermark_path "${PROJECT_DIR}/${WATERMARK_FILE}" \
    --delta "${DELTA}" \
    --gen_length "${GEN_LENGTH}" \
    --gen_steps "${GEN_STEPS}" \
    --block_length "${BLOCK_LENGTH}" \
    --num_samples "${NUM_SAMPLES}"

if [ "${SLURM_ARRAY_TASK_ID}" -eq $((NUM_SHARDS - 1)) ]; then
    echo "Waiting briefly before merge..."
    sleep 60
    python "${SCRIPT_PATH}" \
        --run_dir "${RUN_DIR}" \
        --merge_only \
        --watermark_path "${PROJECT_DIR}/${WATERMARK_FILE}"
fi

echo "Job finished on $(date)"
