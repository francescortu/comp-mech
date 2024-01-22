#!/bin/bash
#SBATCH -n5
#SBATCH --job-name=pythia6b
#SBATCH -N1
#SBATCH -p DGX
#SBATCH --gpus=1
#SBATCH --mem=150gb
#SBATCH --time=150:00:00
#SBATCH --output=my_job_%j.out
#SBATCH --error=my_job_%j.err

conda activate torch
cd script
# python run_all.py --ablate --no-plot --model pythia-6.9b --batch 50 --total-effect  --ablate-component mlp_out
#python run_all.py --ablate --no-plot --model pythia-6.9b --batch 50 --total-effect --ablate-component mlp_out --start 0 --slice 2500 --folder copyVSfact
#python run_all.py --ablate --no-plot --model pythia-6.9b --batch 20 --total-effect --ablate-component mlp_out --start 2500 --slice 5000 --folder copyVSfact
#python run_all.py --ablate --no-plot --model pythia-6.9b --batch 20 --total-effect --ablate-component mlp_out --start 5000 --slice 7500 --folder copyVSfact
python run_all.py --ablate --no-plot --model pythia-6.9b --batch 20 --total-effect --ablate-component mlp_out --start 7500 --slice 10000 --folder copyVSfact
#python score_models.py --all
