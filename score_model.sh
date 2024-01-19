#!/bin/bash
#SBATCH -n10
#SBATCH --job-name=similarity
#SBATCH -N1
#SBATCH -p DGX
#SBATCH --gpus=1
#SBATCH --mem=200gb
#SBATCH --time=150:00:00
#SBATCH --output=my_job_%j.out
#SBATCH --error=my_job_%j.err


conda activate torch
cd script
python score_models.py --similarity --num-samples 1 --similarity-type word2vec
#python score_models.py --all --num-samples 10 --similarity-type logit
