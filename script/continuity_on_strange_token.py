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
    num_samples = 10
    dataset_slice = 1000
    batch_size = 100
    start = 0
    step = 0.1
    end = 1
    num_steps = int((end - start) / step) + 1
    

config = Config()


MODEL_NAME = "gpt2small"
MAX_LEN = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = WrapHookedTransformer.from_pretrained("gpt2", device=DEVICE)
target_data = json.load(open("../data/target_win_dataset_{}_filtered.json".format(MODEL_NAME)))
orthogonal_data = json.load(
    open("../data/orthogonal_win_dataset_{}_filtered.json".format(MODEL_NAME))
)
orthogonal_data = random.sample(orthogonal_data, len(target_data))
dataset = Dataset(target_data, orthogonal_data, model)
dataset.random_sample(config.dataset_slice, MAX_LEN)


dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

target_win = np.zeros((config.num_steps, config.num_samples))
for j,alpha in enumerate([x for x in np.arange(config.start, config.end + config.step, config.step)]):
    orthogonal_word_per_sample = {
        "target": []
    }
    for sample in tqdm(range(config.num_samples)):
        target_win_tmp = 0
        for batch in dataloader:
            pos_batch = batch["pos_dataset"]
            pos_target_ids = {
                "target": model.to_tokens(pos_batch["target"], prepend_bos=False),
                "orthogonal": model.to_tokens(pos_batch["orthogonal_token"], prepend_bos=False),
            }
            
            # add orthogonal token
            for i, d in enumerate(pos_batch["prompt"]):
                orthogonal_token = model.to_orthogonal_tokens2(d, alpha=alpha)
                pos_batch["premise"][i] = d + orthogonal_token + " " + d
                if f"sample_{sample}" not in orthogonal_word_per_sample:
                    orthogonal_word_per_sample[f"sample_{sample}"] = []
                orthogonal_word_per_sample[f"sample_{sample}"].append(orthogonal_token)
                if sample == 0:
                    orthogonal_word_per_sample["target"].append(pos_batch["target"][i])
            
            # compute logits
            logits = model(pos_batch["premise"])
            probabilities = torch.softmax(logits, dim=-1)
            
            target_win_tmp += (probabilities[:,-1,:].argmax(dim=-1) == pos_target_ids["target"].squeeze(-1)).sum().item()
            
        target_win[j,sample] = target_win_tmp / len(dataset)
    # save orthogonal words
    dataframe = pd.DataFrame(orthogonal_word_per_sample)
    dataframe.to_csv("../results/continuity_cp_token/orthogonal_words_{}_alpha_{}.csv".format(MODEL_NAME, alpha))
        


#save
np.save("../results/continuity_cp_token/target_win_{}_alpha.npy".format(MODEL_NAME), target_win)