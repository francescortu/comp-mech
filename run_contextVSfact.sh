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
python run_all.py --ablate --no-plot --model pythia-6.9b --batch 5 --total-effect --start 0 --slice 5000 --folder copyVSfact
python run_all --ablate --no-plot --model pythia-6.9b --batch 5 --total-effect --slice 1000 --total-effect
python run_all --ablate --no-plot --model pythia-6.9b --batch 5 --total-effect --slice 500 --total-effect 
