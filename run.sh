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
# python run_all.py --ablate --no-plot --model pythia-6.9b --batch 5 --total-effect  --ablate-component mlp_out
python run_all.py --ablate --no-plot --model pythia-6.9b --batch 5 --total-effet --ablate-component mlp_out --start 0 --slice 2500
# python run_all.py --ablate --no-plot --model pythia-6.9b --batch 5 --total-effet --ablate-component mlp_out --start 2500 --slice 5000
# python run_all.py --ablate --no-plot --model pythia-6.9b --batch 5 --total-effet --ablate-component mlp_out --start 5000 --slice 7500
# python run_all.py --ablate --no-plot --model pythia-6.9b --batch 5 --total-effet --ablate-component mlp_out --start 7500 --slice 10000
#python score_models.py --all
