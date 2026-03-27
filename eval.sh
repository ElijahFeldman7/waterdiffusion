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

WATERMARK_FILE="watermarks/3_19_watermark.py"
DELTA=4.0
GEN_LENGTH=300
NUM_SAMPLES=300
NUM_SHARDS=3

RUN_NAME="remask_eval_$(date +'%m_%d_%H_%M')"
RUN_DIR="/scratch/efeldma5/waterdiffusion/runs/${RUN_NAME}"

if [ -n "$SLURM_ARRAY_JOB_ID" ]; then
    RUN_DIR="/scratch/efeldma5/waterdiffusion/runs/remask_eval_${SLURM_ARRAY_JOB_ID}"
fi

mkdir -p "$RUN_DIR"
mkdir -p slurm_logs

echo "Job started on $(date)"
source /scratch/efeldma5/waterdiffusion/watervenv/bin/activate

python scripts/waterbench_shared.py \
    --shard "$SLURM_ARRAY_TASK_ID" \
    --num_shards "$NUM_SHARDS" \
    --run_dir "$RUN_DIR" \
    --watermark_path "$WATERMARK_FILE" \
    --delta "$DELTA" \
    --gen_length "$GEN_LENGTH" \
    --num_samples "$NUM_SAMPLES"

if [ "$SLURM_ARRAY_TASK_ID" -eq $((NUM_SHARDS - 1)) ]; then
    echo "Waiting for other shards to finish before merging..."
    sleep 30 
    python scripts/waterbench_shared.py \
        --run_dir "$RUN_DIR" \
        --merge_only \
        --watermark_path "$WATERMARK_FILE"
fi

echo "Job finished on $(date)"
