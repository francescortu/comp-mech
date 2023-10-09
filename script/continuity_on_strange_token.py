import sys
sys.path.append('..')
sys.path.append('../src')
sys.path.append('../data')

import torch
from transformer_lens import HookedTransformer
import json
from src.model import WrapHookedTransformer
from tqdm import tqdm

import transformer_lens.utils as utils
from transformer_lens.utils import get_act_name
from functools import partial
from transformer_lens import patching

import numpy as np
import torch
from tqdm import tqdm


data = json.load(open("../data/known_1000.json"))
model = WrapHookedTransformer.from_pretrained("gpt2-xl", device="cuda")
dataset = []
for d in tqdm(data, total=len(data)):
    dataset.append(
        {"prompt": d["prompt"],
         "target": " " + d["attribute"]}
    )
    

alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
num_alphas = len(alphas)
num_samples = 20

# Initialize arrays
target_win = np.zeros([num_alphas, num_samples])
orthogonal_win = np.zeros([num_alphas, num_samples])
target_win_over_orthogonal = np.zeros([num_alphas, num_samples])


for i, alpha in enumerate(range(0.001, 0.4, 0.001)):
    for sample in range(num_samples):
        tmp_target_win = 0
        tmp_orthogonal_win = 0
        tmp_target_win_over_orthogonal = 0
        dataset_per_length = {}
        for d in tqdm(dataset, total=len(dataset)):
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
                if sample == 0:
                    #print on a file the target and the orthogonal token
                    with open(f"data/known_1000_{alpha}.txt", "a") as f:
                        for j in range(len(batch["premise"])):
                            f.write(f"{batch['prompt'][j]} {batch['target'][j]} {batch['orthogonal_token'][j]}\n")

        target_win[i, sample] = tmp_target_win/len(dataset)
        orthogonal_win[i, sample] = tmp_orthogonal_win/len(dataset)
        target_win_over_orthogonal[i, sample] = tmp_target_win_over_orthogonal/len(dataset)
                
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
