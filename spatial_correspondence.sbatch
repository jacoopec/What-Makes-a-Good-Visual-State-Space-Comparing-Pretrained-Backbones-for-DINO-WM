#!/bin/bash
#SBATCH --job-name=spatial-corr
#SBATCH --partition=all_usr_prod
#SBATCH --account=cvcs2026
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_RTX_A5000_24G|gpu_RTX6000_24G
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=spatial-corr-%j.out
#SBATCH --error=spatial-corr-%j.err

set -euo pipefail
# Submit from the repository root; pass the analysis CLI arguments after this file.
# Single GPU, one backbone and one frame at a time. --mem is HOST RAM, not VRAM.
# Constraints/account follow the workspace GPU_INFO.md and File_batch examples.
# No site-specific VRAM utilization threshold is documented in this repository.
module purge
module load anaconda3/2023.09-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-dino_wm}"
cd "${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export MPLBACKEND=Agg
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Populate pretrained caches before submission. Keep SLURM's CUDA_VISIBLE_DEVICES.
# Optional allocator cap: append --cuda-memory-fraction 0.8 (not a total VRAM limit).
srun python -u spatial_correspondence.py --device cuda "$@"
