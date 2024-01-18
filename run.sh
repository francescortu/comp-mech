#!/bin/bash
#SBATCH -n10
#SBATCH --job-name=pythia6b
#SBATCH -N1
#SBATCH -p DGX
#SBATCH --gpus=1
#SBATCH --mem=200gb
#SBATCH --time=200:00:00
#SBATCH --output=my_job_%j.out
#SBATCH --error=my_job_%j.err

conda activate torch
cd script
python run_all.py --ablate --no-plot --model pythia-6.9b --batch 5 --total-effect  --ablate-component mlp_out
#python score_models.py --all
