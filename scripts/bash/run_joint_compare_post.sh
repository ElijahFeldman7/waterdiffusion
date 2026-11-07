#!/bin/bash
# Oracle PPL + summary for a finished joint_cmp run.
#   sbatch --dependency=afterok:<ARRAY_JOB_ID> scripts/bash/run_joint_compare_post.sh runs/joint_cmp_T0.5_<ARRAY_JOB_ID>
#SBATCH --job-name=joint-cmp-post
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --time=24:00:00

module load hosts/hopper gnu/12.3.0
module load cuda/12.4.0
module load python/3.12.1-33

BASE_DIR="/scratch/efeldma5/waterdiffusion"
PROJECT_DIR="${BASE_DIR}/waterdiffusion"
source "${BASE_DIR}/watervenv/bin/activate"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
cd "$PROJECT_DIR"

RUN_DIR="$1"
PPL_MODEL=${PPL_MODEL:-Qwen/Qwen2.5-32B-Instruct}

python scripts/benchmark.py ppl --run_dir "$RUN_DIR" --ppl_model "$PPL_MODEL" --batch_size 4
python scripts/benchmark.py summarize --run_dir "$RUN_DIR" --min_tokens 100
