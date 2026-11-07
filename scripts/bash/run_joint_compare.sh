#!/bin/bash
#SBATCH --job-name=joint-cmp
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
#SBATCH --array=0-7

module load hosts/hopper gnu/12.3.0
module load cuda/12.4.0
module load python/3.12.1-33

BASE_DIR="/scratch/efeldma5/waterdiffusion"
PROJECT_DIR="${BASE_DIR}/waterdiffusion"
VENV=${VENV:-"${BASE_DIR}/dlmvenv"}
source "${VENV}/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
cd "$PROJECT_DIR"

TEMP=${TEMP:-0.5}
REMASK=${REMASK:-low_confidence}
NUM_SAMPLES=${NUM_SAMPLES:-600}
RUN_DIR=${RUN_DIR:-"runs/joint_cmp_T${TEMP}_${SLURM_ARRAY_JOB_ID}"}

METHODS=(
  none
  dgmark
  zurich:delta=1 zurich:delta=2 zurich:delta=4 zurich:delta=6
  order:order=0.5 order:order=1 order:order=2 order:order=4
  joint:delta=1,order=1 joint:delta=2,order=1 joint:delta=2,order=2 joint:delta=4,order=1
)

echo "Task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT} | T=${TEMP} remask=${REMASK} | ${RUN_DIR}"
python3 scripts/benchmark.py generate \
  --run_dir "$RUN_DIR" \
  --shard "$SLURM_ARRAY_TASK_ID" --num_shards "$SLURM_ARRAY_TASK_COUNT" \
  --num_samples "$NUM_SAMPLES" \
  --temperature "$TEMP" --remasking "$REMASK" \
  --gen_length 300 --steps 300 --block_length 25 \
  --methods "${METHODS[@]}"
