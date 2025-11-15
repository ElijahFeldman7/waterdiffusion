#!/bin/bash

#SBATCH --job-name=water-diffusion
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --time=12:00:00

echo "Job started on $(date)"
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi
echo ""

echo "Setting up the environment..."
module purge
module load gnu10
module load python/3.9.9-jh
module load cuda
echo "Loaded modules:"
module list
echo ""

source /path/to/venv/bin/activate
echo "Venv activated: $VIRTUAL_ENV"
echo ""

echo "Starting Python training"
python /scratch/path/to/file