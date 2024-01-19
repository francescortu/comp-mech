#!/bin/bash
#SBATCH -n10
#SBATCH --job-name=openai
#SBATCH -N1
#SBATCH -p THIN
#SBATCH --mem=30gb
#SBATCH --time=2:00:00
#SBATCH --output=openai_%j.out
#SBATCH --error=openai_%j.err

conda activate torch
cd script
python context_dataset_openai.py --n_samples 10000 --model_name gpt2
python context_dataset_openai.py --n_samples 10000 --model_name gpt2-medium
