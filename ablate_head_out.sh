#!/bin/bash
#SBATCH -n10
#SBATCH --job-name=ablate_head
#SBATCH -N1
#SBATCH -p DGX
#SBATCH --gpus=1
#SBATCH --mem=200gb
#SBATCH --time=150:00:00
#SBATCH --output=my_job_%j.out
#SBATCH --error=my_job_%j.err

conda activate torch
cd script
python run_all.py --ablate --ablate-component head --no-plot --batch 5 --model pythia-6.9b --total-effect
