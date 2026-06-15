#!/bin/bash
#SBATCH --job-name=qwen-ppl-reeval
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --time=48:00:00

# Re-evaluate PPL using Qwen-32B for all proposal runs and Zurich baseline.
# This processes existing completions — no generation, just scoring.
#
# Submit with:
#   sbatch scripts/bash/reevaluate_ppl_qwen_all.sh

BASE_DIR="/scratch/efeldma5/waterdiffusion"
PROJECT_DIR="${BASE_DIR}/waterdiffusion"
SCRIPT_PATH="${PROJECT_DIR}/scripts/reevaluate_ppl_qwen_all.py"

echo "Job started on $(date)"
echo "Node: $(hostname)"
nvidia-smi

mkdir -p slurm_logs
source "${BASE_DIR}/watervenv/bin/activate"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

python "$SCRIPT_PATH" \
    --project-root "$PROJECT_DIR" \
    --batch-size 4 \
    --model "Qwen/Qwen2.5-32B-Instruct"

echo "Job finished on $(date)"
