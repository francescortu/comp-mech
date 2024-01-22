#!/bin/bash
#SBATCH -n10
#SBATCH --job-name=similarity
#SBATCH -N1
#SBATCH -p DGX
#SBATCH --gpus=1
#SBATCH --mem=100gb
#SBATCH --time=150:00:00
#SBATCH --output=similarity_%j.out
#SBATCH --error=similarity_%j.err


conda activate torch
cd script
python score_models.py --similarity --num-samples 10 --similarity-type word2vec --experiment copyVSfact
#python score_models.py --all --num-samples 10 --similarity-type logit
