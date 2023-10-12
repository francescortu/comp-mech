import sys
sys.path.append('..')
sys.path.append('../src')
sys.path.append('../data')

import torch
from transformer_lens import HookedTransformer
import json
from src.model import WrapHookedTransformer
from src.dataset import Dataset
import transformer_lens.utils as utils
from transformer_lens.utils import get_act_name
from functools import partial
from transformer_lens import patching
import plotly.express as px
import random
from src.locate_mechanism import construct_result_dict, indirect_effect
from src.utils import list_of_dicts_to_dict_of_lists
from dataclasses import dataclass
from scipy.stats import ttest_1samp


class Config:
    num_samples: int = 5
    batch_size: int = 5
    mem_win_noise_position = [1,2,3,8,9,10,11]
    mem_win_noise_mlt = 20
    cp_win_noise_position = [1,2,3,8,9,10,11]
    cp_win_noise_mlt = 20
    name_save_file = "gpt2"
    name_dataset = "dataset_gpt2.json"
    max_len = 16
    keys_to_compute = [
        # "logit_lens_mem",
        # "logit_lens_cp",
        # "resid_pos",
        "attn_head_out",
        # "attn_head_by_pos",
        # "per_block",
        "mlp_out",
        "attn_out_by_pos"
    ]
    
config = Config()

def dict_of_lists_to_dict_of_tensors(dict_of_lists):
    dict_of_tensors = {}
    for key, tensor_list in dict_of_lists.items():
        # If the key is "example_str_token", keep it as a list of strings
        if key == "example_str_tokens":
            dict_of_tensors[key] = tensor_list
            continue
        
        # Check if the first element of the list is a tensor
        if isinstance(tensor_list[0], torch.Tensor):
            dict_of_tensors[key] = torch.stack(tensor_list)
        # If the first element is a list, convert each inner list to a tensor and then stack
        elif isinstance(tensor_list[0], list):
            tensor_list = [torch.tensor(item) for item in tensor_list]
            dict_of_tensors[key] = torch.stack(tensor_list)
        else:
            print(f"Unsupported data type for key {key}: {type(tensor_list[0])}")
            raise ValueError(f"Unsupported data type for key {key}: {type(tensor_list[0])}")
    return dict_of_tensors

torch.set_grad_enabled(False)



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = WrapHookedTransformer.from_pretrained("gpt2", device=DEVICE)
dataset = Dataset("../data/{}".format(config.name_dataset))
dataset.random_sample(config.num_samples, config.max_len)
dataset.compute_noise_level(model)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

print("Method used to compute the metric: {}".format(config.keys_to_compute))

pos_result = []
pos_result_var = []
neg_result = []
neg_result_var = []
pos_clean_caches = []
neg_clean_caches = []
for batch in dataloader:

    pos_batch = batch["pos_dataset"]
    neg_batch = batch["neg_dataset"]
    pos_target_ids = {
        "target": model.to_tokens(pos_batch["target"], prepend_bos=False),
        "orthogonal": model.to_tokens(pos_batch["orthogonal_token"], prepend_bos=False),
    }
    neg_target_ids = {
        "target": model.to_tokens(neg_batch["target"], prepend_bos=False),
        "orthogonal": model.to_tokens(neg_batch["orthogonal_token"], prepend_bos=False),
    }
    pos_input_ids = model.to_tokens(pos_batch["premise"], prepend_bos=True)
    neg_input_ids = model.to_tokens(neg_batch["premise"], prepend_bos=True)
    pos_embs_corrupted = dataset.add_noise(
        model,
        pos_batch["premise"],
        noise_index = torch.tensor(config.mem_win_noise_position),
        target_win=8,
        noise_mlt=config.mem_win_noise_mlt
    )
    neg_embs_corrupted = dataset.add_noise(
        model,
        neg_batch["premise"],
        noise_index = torch.tensor(config.cp_win_noise_position),
        target_win=8,
        noise_mlt=config.cp_win_noise_mlt
    )

    pos_corrupted_logit, pos_corrupted_cache = model.run_with_cache_from_embed(pos_embs_corrupted)
    pos_clean_logit, pos_clean_cache = model.run_with_cache(pos_batch["premise"])
    neg_corrupted_logit, neg_corrupted_cache = model.run_with_cache_from_embed(neg_embs_corrupted)
    neg_clean_logit, neg_clean_cache = model.run_with_cache(neg_batch["premise"])
    
    pos_clean_caches.append(pos_clean_cache)
    neg_clean_caches.append(neg_clean_cache)
    
    def check_reversed_probs( corrupted_logits, target_pos, orthogonal_pos):
        corrupted_logits = torch.softmax(corrupted_logits, dim=-1)
        target_probs = corrupted_logits[:,-1,:].gather(-1, index=target_pos).squeeze(-1)
        orthogonal_probs = corrupted_logits[:,-1,:].gather(-1, index=orthogonal_pos).squeeze(-1)
        return (target_probs - orthogonal_probs).mean()

    print("Traget - Orthogonal", check_reversed_probs( pos_corrupted_logit,  pos_target_ids["target"], pos_target_ids["orthogonal"],))
    print("Target - orthogonal", check_reversed_probs( neg_corrupted_logit, neg_target_ids["target"], neg_target_ids["orthogonal"]))
    
    

    def pos_metric(logits):
        improved = indirect_effect(
            logits=logits,
            corrupted_logits=pos_corrupted_logit,
            first_ids_pos=pos_target_ids["target"]
        )
        # improved = improved/POS_BASELINE
        return improved
        
    def neg_metric(logits):
        improved = indirect_effect(
            logits=logits,
            corrupted_logits=neg_corrupted_logit,
            first_ids_pos=neg_target_ids["orthogonal"]
        )
        return improved
    
    

    
    print("pos metric", pos_metric(logits=pos_clean_logit))
    print("neg metric", neg_metric(logits=neg_clean_logit))
    
    

        
    shared_args = {
        "model": model,
        "input_ids": pos_input_ids,
        "clean_cache": pos_clean_cache,
        "metric": pos_metric,
        "embs_corrupted": pos_embs_corrupted,
        "interval": 1,
        "target_ids": pos_target_ids,
    }
    pos_result.append(
        construct_result_dict(shared_args, config.keys_to_compute),
    )
    pos_result[0]["example_str_tokens"] = pos_batch["premise"][0]
    

    
    shared_args = {
        "model": model,
        "input_ids": neg_input_ids,
        "clean_cache": neg_clean_cache,
        "metric": neg_metric,
        "embs_corrupted": neg_embs_corrupted,
        "interval": 1,
        "target_ids": neg_target_ids,
    }
    
    neg_result.append(
        construct_result_dict(shared_args, config.keys_to_compute),
    )
    neg_result[0]["example_str_tokens"] = neg_batch["premise"][0]
    

    


pos_result = list_of_dicts_to_dict_of_lists(pos_result)
neg_result = list_of_dicts_to_dict_of_lists(neg_result)

full_result = {
    "pos": pos_result,
    "neg": neg_result
}

torch.save(full_result, "../results/locate_mechanism/{}_full_result.pt".format(config.name_save_file))
