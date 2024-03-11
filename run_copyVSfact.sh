#!/bin/bash
#SBATCH -n5
#SBATCH --job-name=pythia6b
#SBATCH -N1
#SBATCH -p DGX
#SBATCH --gpus=1
#SBATCH --mem=150gb
#SBATCH --time=150:00:00
#SBATCH --output=ablate_copy%j.out
#SBATCH --error=ablate_copy%j.err

conda activate torch
cd script
python run_all.py --logit_lens --batch 10 --folder copyVSfact --model_name pythia-6.9b --flag _index
python run_all.py --logit_lens --batch 10 --folder copyVSfact --model_name gpt2 --flag _index
#python run_all.py --ablate --no-plot --model pythia-6.9b --batch 10  --slice 2500 --folder copyVSfact --ablate-component mlp_out
#python run_all.py --ablate --no-plot --model pythia-6.9b --batch 10  --slice 2500 --folder copyVSfact --ablate-component attn_out
#python run_all.py --ablate --no-plot --model pythia-6.9b --batch 10 --slice 20 --folder copyVSfact --ablate-component head
#python run_all.py --ablate --no-plot --model pythia-6.9b --batch 5  --slice 2000  --folder contextVSfact
#python run_all.py --ablate --no-plot --model pythia-6.9b --batch 5  --slice 500   --folder contextVSfact
