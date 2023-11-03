import sys
sys.path.append('..')
sys.path.append('../src')
sys.path.append('../data')

import torch
from transformer_lens import HookedTransformer
import json
from src.patching import get_act_patch_mlp_out
from src.model import WrapHookedTransformer
from src.dataset import Dataset
import transformer_lens.utils as utils
from transformer_lens.utils import get_act_name
from functools import partial
from transformer_lens import patching
import plotly.express as px
import random
import numpy as np
from tqdm import tqdm
import pandas as pd

class Config:
    num_samples = 15
    dataset_slice = 100
    batch_size = 100
    start = 0
    step = 0.01
    end = 1
    num_steps = int((end - start) / step) + 1
    

config = Config()


MODEL_NAME = "gpt2small"
MAX_LEN = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = WrapHookedTransformer.from_pretrained("gpt2", device=DEVICE)
dataset = Dataset(f"../data/dataset_gpt2_filtered.json")
dataset.print_statistics()
# dataset.filter(filter_key="cp", filter_interval=config.filter_interval)
# dataset.filter(filter_key="mem", filter_interval=config.filter_interval)
dataset.random_sample(config.dataset_slice, MAX_LEN)
dataset.compute_noise_level(model)


dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

target_win = np.zeros((config.num_steps, config.num_samples))
for j,alpha in enumerate([x for x in np.arange(config.start, config.end + config.step, config.step)]):
    orthogonal_word_per_sample = {
        "premise": [],
        "target": []
    }
    for sample in tqdm(range(config.num_samples)):
        target_win_tmp = 0
        for batch in dataloader:
            pos_batch = batch["mem_dataset"]
            neg_batch = batch["cp_dataset"]
            pos_target_ids = {
                "target": model.to_tokens(pos_batch["target"], prepend_bos=False),
                "orthogonal": model.to_tokens(pos_batch["orthogonal_token"], prepend_bos=False),
            }
            neg_target_ids = {
                "target": model.to_tokens(neg_batch["target"], prepend_bos=False),
                "orthogonal": model.to_tokens(neg_batch["orthogonal_token"], prepend_bos=False),
            }
            # add orthogonal token
            for i, d in enumerate(pos_batch["prompt"]):
                orthogonal_token = model.to_orthogonal_tokens(pos_batch["target"][i], alpha=alpha)
                pos_batch["premise"][i] = d + orthogonal_token + " " + d
                if f"sample_{sample}" not in orthogonal_word_per_sample:
                    orthogonal_word_per_sample[f"sample_{sample}"] = []
                orthogonal_word_per_sample[f"sample_{sample}"].append(orthogonal_token)
                if sample == 0:
                    orthogonal_word_per_sample["target"].append(pos_batch["target"][i])
                    orthogonal_word_per_sample["premise"].append(pos_batch["premise"][i])
            # compute logits
            logits = model(pos_batch["premise"])
            probabilities = torch.softmax(logits, dim=-1)
            target_win_tmp += (probabilities[:,-1,:].argmax(dim=-1) == pos_target_ids["target"].squeeze(-1)).sum().item()
            
            for i, d in enumerate(neg_batch["prompt"]):
                orthogonal_token = model.to_orthogonal_tokens(neg_batch["target"][i], alpha=alpha)
                neg_batch["premise"][i] = d + orthogonal_token + " " + d
                if f"sample_{sample}" not in orthogonal_word_per_sample:
                    orthogonal_word_per_sample[f"sample_{sample}"] = []
                orthogonal_word_per_sample[f"sample_{sample}"].append(orthogonal_token)
                if sample == 0:
                    orthogonal_word_per_sample["target"].append(neg_batch["target"][i])
                    orthogonal_word_per_sample["premise"].append(neg_batch["premise"][i])
            # compute logits
            logits = model(neg_batch["premise"])
            probabilities = torch.softmax(logits, dim=-1)
            target_win_tmp += (probabilities[:,-1,:].argmax(dim=-1) == neg_target_ids["target"].squeeze(-1)).sum().item()

        target_win[j,sample] = target_win_tmp / (2*len(pos_batch["prompt"]))
    # save orthogonal words
    dataframe = pd.DataFrame(orthogonal_word_per_sample)
    dataframe.to_csv("../results/continuity_cp_token/orthogonal_words_{}_alpha_{}.csv".format(MODEL_NAME, alpha))
        


#save
np.save("../results/continuity_cp_token/target_win_{}_alpha.npy".format(MODEL_NAME), target_win)