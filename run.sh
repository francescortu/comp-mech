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

cd Script
#python run_all.py --model-name gpt2 --logit-lens --batch 30 --flag __WITH_SUBJECT --experiment copyVSfact
python score_models.py --similarity --num-samples 10 --experiment copyVSfact --similarity-type modify-self-similarity

