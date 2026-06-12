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
#SBATCH --array=0-6

BASE_DIR="/scratch/efeldma5/waterdiffusion"
PROJECT_DIR="${BASE_DIR}/waterdiffusion"
SCRIPT_PATH="${PROJECT_DIR}/scripts/waterbench_shared.py"
WATERMARK_FILE="watermarks/4_9_segmented_DWT_watermark.py"

DELTAS=(0.5 1.0 2.0 4.0 6.0 8.0 10.0)
DELTA=${DELTAS[$SLURM_ARRAY_TASK_ID]}
DELTA_DIR="delta_${DELTA}"

GEN_LENGTH=300
NUM_SAMPLES=300

if [ -n "$SLURM_ARRAY_JOB_ID" ]; then
    PARENT_DIR="${PROJECT_DIR}/runs/prompt1_${SLURM_ARRAY_JOB_ID}"
else
    PARENT_DIR="${PROJECT_DIR}/runs/prompt1_$(date +'%m_%d_%H_%M')"
fi

RUN_DIR="${PARENT_DIR}/${DELTA_DIR}"
mkdir -p "$RUN_DIR"
mkdir -p slurm_logs

echo "Job started on $(date)"
echo "Array task ${SLURM_ARRAY_TASK_ID} | Delta: ${DELTA} | Dir: ${RUN_DIR}"

source "${BASE_DIR}/watervenv/bin/activate"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

python "$SCRIPT_PATH" \
    --shard 0 \
    --num_shards 1 \
    --run_dir "$RUN_DIR" \
    --watermark_path "${PROJECT_DIR}/${WATERMARK_FILE}" \
    --delta "$DELTA" \
    --gen_length "$GEN_LENGTH" \
    --num_samples "$NUM_SAMPLES"

python "$SCRIPT_PATH" \
    --run_dir "$RUN_DIR" \
    --merge_only \
    --watermark_path "${PROJECT_DIR}/${WATERMARK_FILE}"

echo "Job finished on $(date)"