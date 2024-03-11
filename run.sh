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
python run_all.py --logit_lens --model pythia-6.9b --folder contextVSfact
python run_all.py --logit_lens --model pythia-6.9b --folder copyVSfact_factual
python run_all.py --logit_lens --model gpt2 --folder contextVSfact

#python run_all.py --all --no-plot --model gpt2 --batch 5 --total-effect  

#python run_all.py --all --no-plot --model gpt2-medium --batch 5 --total-effect  
#python run_all.py --all --no-plot --model gpt2-large --batch 5 --total-effect  
#python run_all.py --all --no-plot --model gpt2-xl --batch 5 --total-effect  
#python run_all.py --all --no-plot --model pythia-6.9b --batch 5 --total-effect  
#python run_all.py --ablate --no-plot --model pythia-6.9b --batch 5 --total-effect --ablate-component mlp_out --start 0 --slice 2500 --folder copyVSfact
#python run_all.py --ablate --no-plot --model pythia-6.9b --batch 5 --total-effect --ablate-component mlp_out --start 2500 --slice 5000 --folder copyVSfact
# python run_all.py --ablate --no-plot --model pythia-6.9b --batch 5 --total-effect --ablate-component mlp_out --start 5000 --slice 7500 --folder copyVSfact
#python run_all.py --all --no-plot --model gpt2 --batch 20 --total-effect --folder contextVSfact
#python run_all.py --pattern  --no-plot --model pythia-6.9b --batch 5 --total-effect --folder contextVSfact
#python score_models.py --all
