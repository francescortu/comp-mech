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
target_data = json.load(open("../data/target_win_dataset_{}.json".format(MODEL_NAME)))
orthogonal_data = json.load(
    open("../data/orthogonal_win_dataset_{}.json".format(MODEL_NAME))
)
orthogonal_data = random.sample(orthogonal_data, len(target_data))
dataset = Dataset(target_data, orthogonal_data, model)
dataset.random_sample(20, 14)


dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False)


pos_result = []
neg_result = []
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
        noise_index = torch.tensor([1,2,3,7]),
        target_win=7,
        noise_mlt=15
    )
    neg_embs_corrupted = model.add_noise(
        neg_batch["premise"],
        noise_index = torch.tensor([1,2,3,7]),
        target_win=7,
        noise_mlt=15
    )

    pos_corrupted_logit, pos_corrupted_cache = model.run_with_cache_from_embed(pos_embs_corrupted)
    pos_clean_logit, pos_clean_cache = model.run_with_cache(pos_batch["premise"])
    neg_corrupted_logit, neg_corrupted_cache = model.run_with_cache_from_embed(neg_embs_corrupted)
    neg_clean_logit, neg_clean_cache = model.run_with_cache(neg_batch["premise"])

    def indirect_effect(logits, corrupted_logits, first_ids_pos):
        logits = torch.softmax(logits, dim=-1)
        corrupted_logits = torch.softmax(corrupted_logits, dim=-1)
        batch_index = torch.arange(logits.shape[0])
        delta_value = (
            logits[batch_index, -1, first_ids_pos] - corrupted_logits[batch_index, -1, first_ids_pos]
        )
        return delta_value.mean()

    pos_metric = partial(
        indirect_effect,
        corrupted_logits=pos_corrupted_logit,
        first_ids_pos=pos_target_ids["target"]
    )
    neg_metric = partial(
        indirect_effect,
        corrupted_logits=neg_corrupted_logit,
        first_ids_pos=neg_target_ids["orthogonal"]
    )
    print("pos metric", pos_metric(logits=pos_clean_logit))
    print("neg metric", neg_metric(logits=neg_clean_logit))

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
            pos_embs_corrupted
        ),
        "patch_attn_head_out": patch_attn_head_out_all_pos(
            model,
            pos_input_ids,
            pos_input_ids,
            pos_clean_cache,
            pos_metric,
            pos_embs_corrupted
        ),
        # "patch_attn_head_by_pos": patch_attn_head_by_pos(
        #     model,
        #     pos_input_ids,
        #     pos_input_ids,
        #     pos_clean_cache,
        #     pos_metric,
        #     pos_embs_corrupted
        # ),
        "patch_per_block": patch_per_block_all_poss(
            model,
            pos_input_ids,
            pos_input_ids,
            pos_clean_cache,
            pos_metric,
            1,
            pos_embs_corrupted
        ),
        "patch_mlp_out": get_act_patch_mlp_out(model, pos_input_ids, pos_clean_cache, pos_metric, patch_interval=2, corrupted_embeddings=pos_embs_corrupted)
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

pos_result = list_of_dicts_to_dict_of_lists(pos_result)
neg_result = list_of_dicts_to_dict_of_lists(neg_result)
pos_result = dict_of_lists_to_dict_of_tensors(pos_result)
neg_result = dict_of_lists_to_dict_of_tensors(neg_result)

#print shape of the results
print({key: value.shape for key, value in pos_result.items() if key not in  ["example_str_token", "logit_lens"]})
# compute the mean and std of the metrics
result = {
    "pos": {
        key: {
            "mean": torch.tensor(value).mean(0),
            "std": torch.tensor(value).std(0)
        }
        for key, value in pos_result.items() if key not in  ["example_str_token", "logit_lens"]
    },
    "neg": {
        key: {
            "mean": torch.tensor(value).mean(0),
            "std": torch.tensor(value).std(0)
        }
        for key, value in neg_result.items() if key not in  ["example_str_token", "logit_lens"] 
    }
}
torch.save(result, "../results/indirect_effect_{}.pt".format(MODEL_NAME))
