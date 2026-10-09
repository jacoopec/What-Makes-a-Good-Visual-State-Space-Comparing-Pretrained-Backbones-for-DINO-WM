#!/bin/bash

#SBATCH --job-name=RCIMG
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --account=cvcs2026
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --output=/homes/jpecchini/dino_wm/rcimg-%j.out
#SBATCH --error=/homes/jpecchini/dino_wm/rcimg-%j.err

module purge
module load anaconda3/2023.09-0

# Initialize conda for this non-interactive bash shell
eval "$(conda shell.bash hook)"

conda activate dino_wm


export DATASETDIR="/work/cvcs2026/DINO_WM"

export TORCH_HOME="/homes/jpecchini/.cache/torch"
cd /homes/jpecchini/dino_wm

srun python reconstruct_image.py \
    --env pusht \
    --encoder mae \
    --checkpoint /work/cvcs2026/DINO_WM/pesi2/outputs/pusht_mae/checkpoints/model_latest.pth
    
echo "Finished"