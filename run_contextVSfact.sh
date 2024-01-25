#!/bin/bash
#SBATCH -n5
#SBATCH --job-name=pythia6b
#SBATCH -N1
#SBATCH -p DGX
#SBATCH --gpus=1
#SBATCH --mem=150gb
#SBATCH --time=150:00:00
#SBATCH --output=ablate_context%j.out
#SBATCH --error=ablate_context%j.err

conda activate torch
cd script
python run_all.py --ablate --no-plot --model pythia-6.9b --batch 5  -slice 2500 --folder contextVSfact
#python run_all.py --ablate --no-plot --model pythia-6.9b --batch 5  --slice 2000  --folder contextVSfact
#python run_all.py --ablate --no-plot --model pythia-6.9b --batch 5  --slice 500   --folder contextVSfact
