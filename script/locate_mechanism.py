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
from src.utils import (
    embs_to_tokens_ids,
    patch_resid_pre,
    patch_attn_head_out_all_pos,
    patch_attn_head_by_pos,
    patch_per_block_all_poss,
    patch_attn_head_all_pos_every,
    logit_lens,
    list_of_dicts_to_dict_of_lists,
)
from dataclasses import dataclass


class Config:
    num_samples: int = 1000
    batch_size: int = 100
    mem_win_noise_position = [1,2,3,6,7]
    mem_win_noise_mlt = 15
    cp_win_noise_position = [7]
    cp_win_noise_mlt = 20
config = Config()

def dict_of_lists_to_dict_of_tensors(dict_of_lists):
    dict_of_tensors = {}
    for key, tensor_list in dict_of_lists.items():
        # If the key is "example_str_token", keep it as a list of strings
        if key == "example_str_token" or key == "logit_lens":
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

MODEL_NAME = "gpt2small"
MAX_LEN = 14
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = WrapHookedTransformer.from_pretrained("gpt2", device=DEVICE)
target_data = json.load(open("../data/target_win_dataset_{}_filtered.json".format(MODEL_NAME)))
orthogonal_data = json.load(
    open("../data/orthogonal_win_dataset_{}_filtered.json".format(MODEL_NAME))
)
orthogonal_data = random.sample(orthogonal_data, len(target_data))
dataset = Dataset(target_data, orthogonal_data, model)
dataset.random_sample(config.num_samples, 14)


dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=False)


pos_result = []
pos_result_var = []
neg_result = []
neg_result_var = []
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
    pos_embs_corrupted = model.add_noise(
        pos_batch["premise"],
        noise_index = torch.tensor(config.mem_win_noise_position),
        target_win=7,
        noise_mlt=config.mem_win_noise_mlt
    )
    neg_embs_corrupted = model.add_noise(
        neg_batch["premise"],
        noise_index = torch.tensor(config.cp_win_noise_position),
        target_win=7,
        noise_mlt=config.cp_win_noise_mlt
    )

    pos_corrupted_logit, pos_corrupted_cache = model.run_with_cache_from_embed(pos_embs_corrupted)
    pos_clean_logit, pos_clean_cache = model.run_with_cache(pos_batch["premise"])
    neg_corrupted_logit, neg_corrupted_cache = model.run_with_cache_from_embed(neg_embs_corrupted)
    neg_clean_logit, neg_clean_cache = model.run_with_cache(neg_batch["premise"])
    
    def check_reversed_probs( corrupted_logits, target_pos, orthogonal_pos):
        corrupted_logits = torch.softmax(corrupted_logits, dim=-1)
        target_probs = corrupted_logits[:,-1,:].gather(-1, index=target_pos).squeeze(-1)
        orthogonal_probs = corrupted_logits[:,-1,:].gather(-1, index=orthogonal_pos).squeeze(-1)
        return (target_probs - orthogonal_probs).mean()

    print("Traget - Orthogonal", check_reversed_probs( pos_corrupted_logit,  pos_target_ids["target"], pos_target_ids["orthogonal"],))
    print("Target - orthogonal", check_reversed_probs( neg_corrupted_logit, neg_target_ids["target"], neg_target_ids["orthogonal"]))
    

    def indirect_effect(logits, corrupted_logits, first_ids_pos, return_type="mean"):
        logits = torch.softmax(logits, dim=-1)
        corrupted_logits = torch.softmax(corrupted_logits, dim=-1)
        # Use torch.gather to get the desired values
        logits_values = torch.gather(logits[:, -1, :], 1, first_ids_pos).squeeze()
        corrupted_logits_values = torch.gather(corrupted_logits[:, -1, :], 1, first_ids_pos).squeeze()
        
        delta_value = logits_values - corrupted_logits_values

        return delta_value
            
    POS_BASELINE = indirect_effect(
        logits=pos_clean_logit,
        corrupted_logits=pos_corrupted_logit,
        first_ids_pos=pos_target_ids["target"]
    )
    NEG_BASELINE = indirect_effect(
        logits=neg_clean_logit,
        corrupted_logits=neg_corrupted_logit,
        first_ids_pos=neg_target_ids["orthogonal"]
    )


    def pos_metric(logits, return_type="mean"):
        improved = indirect_effect(
            logits=logits,
            corrupted_logits=pos_corrupted_logit,
            first_ids_pos=pos_target_ids["target"]
        )
        # improved = improved/POS_BASELINE
        if return_type == "mean":
            return improved.mean()
        elif return_type == "var":
            return improved.var()
        
    def neg_metric(logits, return_type="mean"):
        improved = indirect_effect(
            logits=logits,
            corrupted_logits=neg_corrupted_logit,
            first_ids_pos=neg_target_ids["orthogonal"]
        )
        # improved = improved/NEG_BASELINE
        if return_type == "mean":
            return improved.mean(dim=0)
        elif return_type == "var":
            return improved.var(dim=0)
    
    pos_metric_var = partial(pos_metric, return_type="var")
    neg_metric_var = partial(neg_metric, return_type="var")

    
    print("pos metric", pos_metric(logits=pos_clean_logit), "var", pos_metric_var(logits=pos_clean_logit))
    print("neg metric", neg_metric(logits=neg_clean_logit), "var", neg_metric_var(logits=neg_clean_logit))

    pos_result.append({
        "example_str_token": model.to_str_tokens(pos_input_ids[0]),
        "logit_lens": logit_lens(
            pos_clean_cache,
            model,
            pos_input_ids,
            pos_target_ids["target"],
            pos_target_ids["orthogonal"],
        ),
        "patch_resid_position": patch_resid_pre(
                        model,
                        pos_input_ids,
                        pos_input_ids,
                        pos_clean_cache,
                        pos_metric,
                        pos_embs_corrupted),
        "patch_attn_head_out": patch_attn_head_out_all_pos(
                        model,
                        pos_input_ids,
                        pos_input_ids,
                        pos_clean_cache,
                        pos_metric,
                        pos_embs_corrupted),
        # "patch_attn_head_by_pos": patch_attn_head_by_pos(
        #     model,
        #     pos_input_ids,
        #     pos_input_ids,
        #     pos_clean_cache,
        #     pos_metric,
        #     pos_embs_corrupted
        # ),
        "patch_per_block":  patch_per_block_all_poss(model,
                        pos_input_ids,
                        pos_input_ids,
                        pos_clean_cache,
                        pos_metric,
                        1,
                        pos_embs_corrupted
                      ),
        "patch_mlp_out": get_act_patch_mlp_out(model, pos_input_ids, pos_clean_cache, pos_metric, patch_interval=2, corrupted_embeddings=pos_embs_corrupted)
    })

    pos_result_var.append({
        "example_str_token": model.to_str_tokens(pos_input_ids[0]),
        "logit_lens": logit_lens(
            pos_clean_cache,
            model,
            pos_input_ids,
            pos_target_ids["target"],
            pos_target_ids["orthogonal"],
        ),
        "patch_resid_position": patch_resid_pre(
                        model,
                        pos_input_ids,
                        pos_input_ids,
                        pos_clean_cache,
                        pos_metric_var,
                        pos_embs_corrupted),
        "patch_attn_head_out": patch_attn_head_out_all_pos(
                        model,  
                        pos_input_ids,
                        pos_input_ids,
                        pos_clean_cache,
                        pos_metric_var,
                        pos_embs_corrupted),
        # "patch_attn_head_by_pos": patch_attn_head_by_pos(
        #     model,
        #     pos_input_ids,
        #     pos_input_ids,
        #     pos_clean_cache,
        #     pos_metric_var,
        #     pos_embs_corrupted
        # ),
        "patch_per_block":  patch_per_block_all_poss(model,
                        pos_input_ids,
                        pos_input_ids,
                        pos_clean_cache,
                        pos_metric_var,
                        1,
                        pos_embs_corrupted
                      ),
        "patch_mlp_out": get_act_patch_mlp_out(model, pos_input_ids, pos_clean_cache, pos_metric_var, patch_interval=2, corrupted_embeddings=pos_embs_corrupted)
    })
    

    neg_result.append({
        "example_str_token": model.to_str_tokens(neg_input_ids[0]),
        "logit_lens": logit_lens(
            neg_clean_cache,
            model,
            neg_input_ids,
            neg_target_ids["target"],
            neg_target_ids["orthogonal"],
        ),
        "patch_resid_position": patch_resid_pre(
            model,
            neg_input_ids,
            neg_input_ids,
            neg_clean_cache,
            neg_metric,
            neg_embs_corrupted
        ),
        "patch_attn_head_out": patch_attn_head_out_all_pos(
            model,
            neg_input_ids,
            neg_input_ids,
            neg_clean_cache,
            neg_metric,
            neg_embs_corrupted
        ),
        # "patch_attn_head_by_pos": patch_attn_head_by_pos(
        #     model,
        #     neg_input_ids,
        #     neg_input_ids,
        #     neg_clean_cache,
        #     neg_metric,
        #     neg_embs_corrupted
        # ),
        "patch_per_block": patch_per_block_all_poss(
            model,
            neg_input_ids,
            neg_input_ids,
            neg_clean_cache,
            neg_metric,
            1,
            neg_embs_corrupted
        ),
        "patch_mlp_out": get_act_patch_mlp_out(model, neg_input_ids, neg_clean_cache, neg_metric, patch_interval=2, corrupted_embeddings=neg_embs_corrupted)
    })
    
    neg_result_var.append({
            "example_str_token": model.to_str_tokens(neg_input_ids[0]),
            "logit_lens": logit_lens(
                neg_clean_cache,
                model,
                neg_input_ids,
                neg_target_ids["target"],
                neg_target_ids["orthogonal"],
            ),
            "patch_resid_position": patch_resid_pre(
                model,
                neg_input_ids,
                neg_input_ids,
                neg_clean_cache,
                neg_metric_var,
                neg_embs_corrupted
            ),
            "patch_attn_head_out": patch_attn_head_out_all_pos(
                model,
                neg_input_ids,
                neg_input_ids,
                neg_clean_cache,
                neg_metric_var,
                neg_embs_corrupted
            ),
            # "patch_attn_head_by_pos": patch_attn_head_by_pos(
            #     model,
            #     neg_input_ids,
            #     neg_input_ids,
            #     neg_clean_cache,
            #     neg_metric_var,
            #     neg_embs_corrupted
            # ),
            "patch_per_block": patch_per_block_all_poss(
                model,
                neg_input_ids,
                neg_input_ids,
                neg_clean_cache,
                neg_metric_var,
                1,
                neg_embs_corrupted
            ),
            "patch_mlp_out": get_act_patch_mlp_out(model, neg_input_ids, neg_clean_cache, neg_metric_var, patch_interval=2, corrupted_embeddings=neg_embs_corrupted)
        })

pos_result = list_of_dicts_to_dict_of_lists(pos_result)
neg_result = list_of_dicts_to_dict_of_lists(neg_result)
pos_result = dict_of_lists_to_dict_of_tensors(pos_result)
neg_result = dict_of_lists_to_dict_of_tensors(neg_result)
pos_result_var = list_of_dicts_to_dict_of_lists(pos_result_var)
neg_result_var = list_of_dicts_to_dict_of_lists(neg_result_var)
pos_result_var = dict_of_lists_to_dict_of_tensors(pos_result_var)
neg_result_var = dict_of_lists_to_dict_of_tensors(neg_result_var)

#print shape of the results
print({key: value.shape for key, value in pos_result.items() if key not in  ["example_str_token", "logit_lens"]})
# compute the mean and std of the metrics
result = {
    "pos": {
        key: {
            "mean": value.clone().detach().mean(0),
            "std": value.clone().detach().std(0)
        }
        for key, value in pos_result.items() if key not in  ["example_str_token", "logit_lens"]    
    },
    "neg": {
        key: {
            "mean": value.clone().detach().mean(0),
            "std": value.clone().detach().std(0)
        }
        for key, value in neg_result.items() if key not in  ["example_str_token", "logit_lens"] 
    }
}
result_var = {
    "pos": {
        key: {
            "mean": value.clone().detach().mean(0),
            "std": value.clone().detach().std(0)
        }
        for key, value in pos_result_var.items() if key not in  ["example_str_token", "logit_lens"]    
    },
    "neg": {
        key: {
            "mean": value.clone().detach().mean(0),
            "std": value.clone().detach().std(0)
        }
        for key, value in neg_result_var.items() if key not in  ["example_str_token", "logit_lens"] 
    }
}
## COMPUTE pooled variance





result["pos"]["example_str_token"] = pos_result["example_str_token"][0]
result["pos"]["logit_lens"] = pos_result["logit_lens"][0]
result["neg"]["example_str_token"] = neg_result["example_str_token"][0]
result["neg"]["logit_lens"] = neg_result["logit_lens"][0]
torch.save(result, "../results/indirect_effect_{}.pt".format(MODEL_NAME))
torch.save(result_var, "../results/indirect_effect_var_{}.pt".format(MODEL_NAME))
