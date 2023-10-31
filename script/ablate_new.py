import sys
sys.path.append('..')
sys.path.append('../src')
sys.path.append('../data')

import torch
import plotly.express as px
import src.nanda_plot
from src.model import WrapHookedTransformer

from src.nanda_plot import imshow_reversed, imshow
import pandas as pd
from tqdm import tqdm
import numpy as np


from src.result_analyzer import ResultAnalyzer



model = WrapHookedTransformer.from_pretrained("gpt2", device="cuda", refactor_factored_attn_matrices=False)


import json
my_data = json.load(open("../data/counterfact_small_15_final.json"))
# sample random 50 examples
import random
random.seed(124)
random.shuffle(my_data)
my_data = my_data
print(len(my_data))

from torch.utils.data import DataLoader, Dataset

class CounterfactDataset(Dataset):
    def __init__(self, data):
        pad_token = model.tokenizer.pad_token
        self.clean_prompts = [d["template"].format(pad_token) for d in data]
        self.corrupted_prompts = [d["template"].format(d["target_new"]) for d in data]

        target1 = [model.to_tokens(d["target_true"], prepend_bos=False) for d in data]
        target2 = [model.to_tokens(d["target_new"], prepend_bos=False) for d in data]
        tensor_1 = torch.stack(target1, dim=0)
        tensor_2 = torch.stack(target2, dim=0)
        # stack the tensors
        self.target = torch.stack([tensor_1, tensor_2], dim=1).squeeze()
    def __len__(self):
        return len(self.clean_prompts)
    def __getitem__(self, idx):
        return {
            "clean_prompts": self.clean_prompts[idx],
            "corrupted_prompts": self.corrupted_prompts[idx],
            "target": self.target[idx]
        }
    def filter_from_idx(self, index, exclude=False):
        if exclude:
            self.target = [self.target[i] for i in range(len(self.target)) if i not in index]
            self.clean_prompts = [self.clean_prompts[i] for i in range(len(self.clean_prompts)) if i not in index]
        
            self.corrupted_prompts = [self.corrupted_prompts[i] for i in range(len(self.corrupted_prompts)) if i not in index]
        else:
            self.target = [self.target[i] for i in index][:200]
            self.clean_prompts = [self.clean_prompts[i] for i in index][:200]
            self.corrupted_prompts = [self.corrupted_prompts[i] for i in index][:200]
    
    def slice(self, end, start=0):
        self.target = self.target[start:end]
        self.clean_prompts = self.clean_prompts[start:end]
        self.corrupted_prompts = self.corrupted_prompts[start:end]
    
        
        
dataset = CounterfactDataset(my_data)
batch_size=50
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
num_batches = len(dataloader)
torch.set_grad_enabled(False)

clean_logits = []
corrupted_logits = []
target = []
for batch in tqdm(dataloader, total=num_batches):
    clean_logits.append(model(batch["clean_prompts"])[:,-1,:].cpu())
    corrupted_logits.append(model(batch["corrupted_prompts"])[:,-1,:].cpu())
    target.append(batch["target"].cpu())
clean_logit = torch.cat(clean_logits, dim=0)
corrupted_logit = torch.cat(corrupted_logits, dim=0)
target = torch.cat(target, dim=0)

def to_logit_token(logit, target):
    logit = torch.log_softmax(logit, dim=-1)
    logit_mem = torch.zeros(target.shape[0])
    logit_cp = torch.zeros(target.shape[0])
    for i in range(target.shape[0]):
        logit_mem[i] = logit[i, target[i,0]]
        logit_cp[i] = logit[i, target[i,1]]
    return logit_mem, logit_cp

CLEAN_LOGIT_MEM = to_logit_token(clean_logit, target)[0]
CLEAN_LOGIT_CP = to_logit_token(clean_logit, target)[1]
CORRUPTED_LOGIT_MEM = to_logit_token(corrupted_logit, target)[0]
CORRUPTED_LOGIT_CP = to_logit_token(corrupted_logit, target)[1]

def normalize_logit_token(logit, target):
    logit_mem, logit_cp = to_logit_token(logit, target)
    # percentage increase or decrease of logit_mem
    logit_mem = 100 * (logit_mem - CLEAN_LOGIT_MEM) / CLEAN_LOGIT_MEM
    # percentage increase or decrease of logit_cp
    logit_cp = 100 * (logit_cp - CLEAN_LOGIT_CP) / CLEAN_LOGIT_CP
    return logit_mem, logit_cp

# get the index of the outliers
outliers_under = torch.where(CORRUPTED_LOGIT_MEM < (CORRUPTED_LOGIT_MEM.mean() - CORRUPTED_LOGIT_MEM.std()) )[0]
outliers_over = torch.where(CORRUPTED_LOGIT_MEM > (CORRUPTED_LOGIT_MEM.mean() + CORRUPTED_LOGIT_MEM.std()) )[0]
outliers_indexes = torch.cat([outliers_under, outliers_over], dim=0).tolist()
both_high = torch.where((CORRUPTED_LOGIT_MEM > (CORRUPTED_LOGIT_MEM.mean() + CORRUPTED_LOGIT_MEM.std())) & (CLEAN_LOGIT_CP > (CLEAN_LOGIT_CP.mean() + CLEAN_LOGIT_CP.std())))[0]

print("Both high: ", len(both_high), len(list(set(both_high.tolist()).intersection(set(outliers_indexes)))))
print("Outliers found: ", 100 * len(outliers_indexes)/len(my_data), "%")

maxdatasize = ((len(my_data) - len(outliers_indexes)) // batch_size) * batch_size

dataset.filter_from_idx(outliers_indexes, exclude=True)
dataset.slice(maxdatasize)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
num_batches = len(dataloader)
torch.set_grad_enabled(False)
print("dataset len:",len(dataset))

clean_logits = []
corrupted_logits = []
target = []
for batch in tqdm(dataloader, total=num_batches):
    clean_logits.append(model(batch["clean_prompts"])[:,-1,:].cpu())
    corrupted_logits.append(model(batch["corrupted_prompts"])[:,-1,:].cpu())
    target.append(batch["target"].cpu())
clean_logit = torch.cat(clean_logits, dim=0)
corrupted_logit = torch.cat(corrupted_logits, dim=0)
target = torch.cat(target, dim=0)

def to_logit_token(logit, target):
    logit = torch.log_softmax(logit, dim=-1)
    logit_mem = torch.zeros(target.shape[0])
    logit_cp = torch.zeros(target.shape[0])
    for i in range(target.shape[0]):
        logit_mem[i] = logit[i, target[i,0]] 
        logit_cp[i] = logit[i, target[i,1]]
    return logit_mem, logit_cp

CLEAN_LOGIT_MEM = to_logit_token(clean_logit, target)[0]
CLEAN_LOGIT_CP = to_logit_token(clean_logit, target)[1]
CORRUPTED_LOGIT_MEM = to_logit_token(corrupted_logit, target)[0]
CORRUPTED_LOGIT_CP = to_logit_token(corrupted_logit, target)[1]

def normalize_logit_token(logit_mem, logit_cp,  baseline="corrupted",):
    # logit_mem, logit_cp = to_logit_token(logit, target)
    # percentage increase or decrease of logit_mem
    if baseline == "clean":
        logit_mem = 100 * (logit_mem - CLEAN_LOGIT_MEM) / CLEAN_LOGIT_MEM
        # percentage increase or decrease of logit_cp
        logit_cp = 100 * (logit_cp - CLEAN_LOGIT_CP) / CLEAN_LOGIT_CP
        return -logit_mem, -logit_cp
    elif baseline == "corrupted":
        logit_mem = 100 * (logit_mem - CORRUPTED_LOGIT_MEM) / CORRUPTED_LOGIT_MEM
        # percentage increase or decrease of logit_cp
        logit_cp = 100 * (logit_cp - CORRUPTED_LOGIT_CP) / CORRUPTED_LOGIT_CP
        return -logit_mem, -logit_cp
    
    
    # patch attention head
from functools import partial
from tqdm import tqdm
import einops
from copy import deepcopy


def heads_hook(activation, hook, head, clean_activation, pos1=None, pos2=None):
    activation[:, head, -1, :] = 0
    return activation

def freeze_hook(activation, hook, head, clean_activation, pos1=None, pos2=None):
    activation = clean_activation
    return activation


examples_mem = torch.zeros((model.cfg.n_layers, model.cfg.n_heads, num_batches, batch_size))
examples_cp = torch.zeros((model.cfg.n_layers, model.cfg.n_heads, num_batches, batch_size))


for idx, batch in enumerate(dataloader):
    corrupted_logit, corrupted_cache = model.run_with_cache(batch["clean_prompts"])
    clean_logit, clean_cache = model.run_with_cache(batch["corrupted_prompts"])
    hooks = {}
    for layer in tqdm(range(model.cfg.n_layers), desc="setting hooks"):
        for head in range(model.cfg.n_heads):
            hooks[f"L{layer}H{head}"] = (f"blocks.{layer}.attn.hook_pattern",
                            partial(
                                freeze_hook,
                                head=head,
                                clean_activation=clean_cache[f"blocks.{layer}.attn.hook_pattern"],
                                )
                            )
            
    for layer in tqdm(range(model.cfg.n_layers), desc="running model"):
        for head in range(model.cfg.n_heads):
            tmp_hooks = deepcopy(hooks)
            tmp_hooks[f"L{layer}H{head}"] = (f"blocks.{layer}.attn.hook_pattern",
                            partial(
                                heads_hook,
                                head=head,
                                clean_activation=corrupted_cache[f"blocks.{layer}.attn.hook_pattern"],
                                )
                            )
            
            list_hooks = list(tmp_hooks.values())
            model.reset_hooks()
            logit = model.run_with_hooks(
                batch["corrupted_prompts"],
                fwd_hooks=list_hooks,
            )[:,-1,:]
            mem, cp = to_logit_token(logit, batch["target"])
            # norm_mem, norm_cp = normalize_logit_token(mem, cp, baseline="corrupted")
            examples_mem[layer, head, idx, :] = mem
            examples_cp[layer, head, idx, :] = cp
        #remove the hooks for the previous layer
        for head in range(model.cfg.n_heads):
            hooks.pop(f"L{layer}H{head}")
        
examples_mem = einops.rearrange(examples_mem, "l h b s -> l h (b s)")
examples_cp = einops.rearrange(examples_cp, "l h b s -> l h (b s)")
for layer in range(model.cfg.n_layers):
    for head in range(model.cfg.n_heads):
        norm_mem, norm_cp = normalize_logit_token(examples_mem[layer, head, :], examples_cp[layer, head, :], baseline="corrupted")
        examples_mem[layer, head, :] = norm_mem
        examples_cp[layer, head, :] = norm_cp
result_cp = examples_cp.mean(dim=-1)
result_cp_std = examples_cp.std(dim=-1)
result_mem = examples_mem.mean(dim=-1)
result_mem_std = examples_mem.std(dim=-1)


def save_result(result_cp, result_mem, filename):
    path = f"../results/circuit_discovery/{filename}"
    torch.save(result_cp, path + "_cp.pt")
    torch.save(result_mem, path + "_mem.pt")
    
save_result(examples_cp, examples_mem, "clean->corrupted_full")