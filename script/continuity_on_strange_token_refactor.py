import sys
sys.path.append('..')
sys.path.append('../src')
sys.path.append('../data')

import torch
from transformer_lens import HookedTransformer
import json
from src.model import WrapHookedTransformer
from src.utils import float_range
from tqdm import tqdm

import transformer_lens.utils as utils
from transformer_lens.utils import get_act_name
from functools import partial
from transformer_lens import patching

import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass

@dataclass
class Config:
    dataset = "target_win"
    sample = 1000
    model = "gpt2"
    alphas = [0.001, 0.04, 0.001]


data = json.load(open("../data/{}_dataset_gpt2small_filtered.json".format(Config.dataset)))
model = WrapHookedTransformer.from_pretrained(Config.model, device="cuda")

num_alphas = len(float_range(Config.alphas[0], Config.alphas[1], Config.alphas[2]))
num_samples = Config.sample

# Initialize arrays
target_win = np.zeros([num_alphas, num_samples])
orthogonal_win = np.zeros([num_alphas, num_samples])
target_win_over_orthogonal = np.zeros([num_alphas, num_samples])


for i, alpha in enumerate(float_range(Config.alphas[0], Config.alphas[1], Config.alphas[2])):
    df_targets_orthogonal_tokens = pd.DataFrame()
    for sample in range(num_samples):
        tmp_target_win = 0
        tmp_orthogonal_win = 0
        tmp_target_win_over_orthogonal = 0
        dataset_per_length = {}
        for d in tqdm(data, total=len(data)):
            orthogonal_token = model.to_orthogonal_tokens(d["target"], alpha=alpha)
            if alpha == 0.0:
                orthogonal_token = d["target"]
            d["premise"] = d["prompt"] + orthogonal_token + " " + d["prompt"]
            d["orthogonal_token"] = orthogonal_token
            d["length"] = len(model.to_str_tokens(d["premise"]))
            if d["length"] not in dataset_per_length:
                dataset_per_length[d["length"]] = []
            dataset_per_length[d["length"]].append(d)
        #create a pytorch dataloader for each length
        dataloaders = {}
        for length in sorted(dataset_per_length.keys()):
            dataloaders[length] = torch.utils.data.DataLoader(dataset_per_length[length], batch_size=100, shuffle=True)
        
        for length in sorted(dataset_per_length.keys()):
            # get logits for each example
            for batch in tqdm(dataloaders[length]):
                logit = model(batch["premise"])
                probs = torch.softmax(logit, dim=-1)
                batch_index = torch.arange(probs.shape[0])
                target_probs = probs[batch_index, -1, model.to_tokens(batch["target"], prepend_bos=False).squeeze(-1)]
                orthogonal_tokens = model.to_tokens(batch["orthogonal_token"], prepend_bos=False).squeeze(-1)
                if len(orthogonal_tokens.shape) == 2:
                    orthogonal_tokens = orthogonal_tokens[:, 0]
                orthogonal_probs = probs[batch_index, -1, orthogonal_tokens]
                predictions = probs[:,-1,:].max(dim=-1)[0]
                # for each element of the batch check the prediction and update the win counter
                for j in range(len(batch["premise"])):
                    if target_probs[j] == predictions[j]:
                        tmp_target_win += 1
                    elif orthogonal_probs[j] == predictions[j]:
                        tmp_orthogonal_win += 1
                    if target_probs[j] > orthogonal_probs[j]:
                        tmp_target_win_over_orthogonal += 1
                        
 
                targets = [batch["target"][j] for j in range(len(batch["premise"]))]
                orthogonal_tokens = [batch["orthogonal_token"][j] for j in range(len(batch["premise"]))]
                df_targets_orthogonal_tokens[f"target_{sample}"] = targets
                df_targets_orthogonal_tokens[f"orthogonal_token_{sample}"] = orthogonal_tokens
                        
                # if sample == 0:
                #     #print on a file the target and the orthogonal token
                #     with open(f"data/known_1000_{alpha}.txt", "a") as f:
                #         for j in range(len(batch["premise"])):
                #             f.write(f"{batch['prompt'][j]} {batch['target'][j]} {batch['orthogonal_token'][j]}\n")

        target_win[i, sample] = tmp_target_win/len(data)
        orthogonal_win[i, sample] = tmp_orthogonal_win/len(data)
        target_win_over_orthogonal[i, sample] = tmp_target_win_over_orthogonal/len(data)
    df_targets_orthogonal_tokens.to_csv(f"data/targets_orthogonal_tokens_alpha_{alpha}.csv", index=False)

    
    
    
    
    
target_win = target_win
orthogonal_win = orthogonal_win
target_win_over_orthogonal = target_win_over_orthogonal

import os
script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
target_win_path = os.path.join(script_dir, "../data/target_win_gpt2-xl.npy")
orthogonal_win_path = os.path.join(script_dir, "../data/orthogonal_win_gpt2-xl.npy")
target_win_over_orthogonal_path = os.path.join(script_dir, "../data/target_win_over_orthogonal_gpt2-xl.npy")
# Save the results
np.save(target_win_path, target_win)
np.save(orthogonal_win_path, orthogonal_win)
np.save(target_win_over_orthogonal_path, target_win_over_orthogonal)
df_targets_orthogonal_tokens.to_csv("../data/targets_orthogonal_tokens.csv", index=False)
