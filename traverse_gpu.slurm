#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:2
#SBATCH --job-name=wall

module load anaconda3
conda activate torch-env
python main.py