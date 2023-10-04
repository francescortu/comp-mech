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

data = json.load(open("../data/known_1000.json"))
model = WrapHookedTransformer.from_pretrained("gpt2", device="cuda")
dataset = []
for d in tqdm(data, total=len(data)):
    dataset.append(
        {"prompt": d["prompt"],
         "target": " " + d["attribute"]}
    )
    
import numpy as np
import torch
from tqdm import tqdm

alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
num_alphas = len(alphas)
num_samples = 10

# Initialize arrays
target_win = np.zeros([num_alphas, num_samples])
orthogonal_win = np.zeros([num_alphas, num_samples])
target_win_over_orthogonal = np.zeros([num_alphas, num_samples])

for i, alpha in enumerate(alphas):
    for sample in range(num_samples):
        # Update dataset with orthogonal tokens and lengths
        for d in tqdm(dataset, total=len(dataset)):
            orthogonal_token = model.to_orthogonal_tokens(d["target"], alpha=0.5)
            d["premise"] = d["prompt"] + orthogonal_token + " " + d["prompt"]
            d["orthogonal_token"] = orthogonal_token
            d["length"] = len(model.to_str_tokens(d["premise"]))

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)
        
        target_win_for_sample = 0
        orthogonal_win_for_sample = 0
        target_win_over_orthogonal_for_sample = 0

        for batch in tqdm(dataloader):
            logit = model(batch["premise"])
            probs = torch.softmax(logit, dim=-1)
            batch_index = torch.arange(probs.shape[0])
            
            target_tokens = model.to_tokens(batch["target"], prepend_bos=False).squeeze(-1)
            orthogonal_tokens = model.to_tokens(batch["orthogonal_token"], prepend_bos=False).squeeze(-1)
            
            if len(orthogonal_tokens.shape) == 2:
                orthogonal_tokens = orthogonal_tokens[:, 0]
            
            target_probs = probs[batch_index, -1, target_tokens]
            orthogonal_probs = probs[batch_index, -1, orthogonal_tokens]
            predictions = probs[:, -1, :].max(dim=-1)[0]

            target_win_for_sample += (target_probs == predictions).sum().item()
            orthogonal_win_for_sample += (orthogonal_probs == predictions).sum().item()
            target_win_over_orthogonal_for_sample += (target_probs > orthogonal_probs).sum().item()

        dataset_length = len(dataset)
        target_win[i, sample] = target_win_for_sample / dataset_length
        orthogonal_win[i, sample] = orthogonal_win_for_sample / dataset_length
        target_win_over_orthogonal[i, sample] = target_win_over_orthogonal_for_sample / dataset_length

print(target_win.mean(axis=1), target_win.std(axis=1))
print(orthogonal_win.mean(axis=1), orthogonal_win.std(axis=1))
print(target_win_over_orthogonal.mean(axis=1), target_win_over_orthogonal.std(axis=1))

# Save the results
np.save("target_win.npy", target_win)
np.save("orthogonal_win.npy", orthogonal_win)
np.save("target_win_over_orthogonal.npy", target_win_over_orthogonal)
