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
)

torch.set_grad_enabled(False)

MODEL_NAME = "gpt2small"
MAX_LEN = 14
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


model = WrapHookedTransformer.from_pretrained("gpt2", device=DEVICE)
target_data = json.load(open("../data/target_win_dataset_{}.json".format(MODEL_NAME)))
# suffhle
orthogonal_data = json.load(
    open("../data/orthogonal_win_dataset_{}.json".format(MODEL_NAME))
)
orthogonal_data = random.sample(orthogonal_data, len(target_data))
dataset = Dataset(target_data, orthogonal_data, model)
dataset.random_sample(150, 14)

print("Dataset loaded")

logits_per_length = dataset.logits(model)
tokens_dict_per_length = dataset.get_tensor_token(model)

delta_per_length = {}
for max_len in list(tokens_dict_per_length.keys()):
    if max_len not in delta_per_length:
        delta_per_length[max_len] = {}
    # probs = torch.softmax(logits_per_length[max_len], dim=-1)
    probs = logits_per_length[max_len]
    batch_index = torch.arange(probs.shape[0])
    delta_per_length[max_len] = (
        probs[batch_index, -1, tokens_dict_per_length[max_len]["target"]]
        - probs[batch_index, -1, tokens_dict_per_length[max_len]["orthogonal_token"]]
    )
delta_per_length


delta = delta_per_length[max_len]
data = dataset.dataset_per_length[max_len]
logits = logits_per_length[max_len]
input_ids = model.to_tokens([d["premise"] for d in data])

pos_logits = logits[delta > 0]
neg_logits = logits[delta < 0]

pos_data = [d for d, dlt in zip(data, delta) if dlt > 0]
neg_data = [d for d, dlt in zip(data, delta) if dlt < 0]

pos_target_ids = {
    "target": tokens_dict_per_length[max_len]["target"][delta > 0],
    "orthogonal_token": tokens_dict_per_length[max_len]["orthogonal_token"][delta > 0],
}
neg_target_ids = {
    "target": tokens_dict_per_length[max_len]["target"][delta < 0],
    "orthogonal_token": tokens_dict_per_length[max_len]["orthogonal_token"][delta < 0],
}

pos_delta = torch.stack([dlt for dlt in delta if dlt > 0])
neg_delta = torch.stack([dlt for dlt in delta if dlt < 0])

pos_input_ids = model.to_tokens([d["premise"] for d in pos_data])
neg_input_ids = model.to_tokens([d["premise"] for d in neg_data])

pos_embs_corrupted = model.add_noise(
    [d["premise"] for d in pos_data],
    noise_index=torch.tensor([1, 2, 3, 7]),
    target_win=7,
    noise_mlt=15,
)
# pos_embs_corrupted = model.add_noise([d["premise"] for d in pos_data],  noise_index = torch.tensor([1, 8]))
neg_embs_corrupted = model.add_noise(
    [d["premise"] for d in neg_data],
    noise_index=torch.tensor([1, 2, 3, 7]),
    target_win=7,
    noise_mlt=4,
)
# neg_embs_corrupted = model.add_noise([d["premise"] for d in neg_data],  noise_index = torch.tensor([1,2,3,4,5,6,8,9,10,11,12]))


pos_input_ids_corrupted = embs_to_tokens_ids(pos_embs_corrupted, model)
neg_input_ids_corrupted = embs_to_tokens_ids(neg_embs_corrupted, model)

pos_delta_corrupted = (
    model(pos_input_ids_corrupted)[
        torch.arange(pos_input_ids_corrupted.shape[0]), -1, pos_target_ids["target"]
    ]
    - model(pos_input_ids_corrupted)[
        torch.arange(pos_input_ids_corrupted.shape[0]),
        -1,
        pos_target_ids["orthogonal_token"],
    ]
)
neg_delta_corrupted = (
    model(neg_input_ids_corrupted)[
        torch.arange(neg_input_ids_corrupted.shape[0]), -1, neg_target_ids["target"]
    ]
    - model(neg_input_ids_corrupted)[
        torch.arange(neg_input_ids_corrupted.shape[0]),
        -1,
        neg_target_ids["orthogonal_token"],
    ]
)

corrupted_logit, corrupted_cache = model.run_with_cache_from_embed(pos_embs_corrupted)


def indirect_effect(logits, corrupted_logits, first_ids_pos):
    logits = torch.softmax(logits, dim=-1)
    corrupted_logits = torch.softmax(corrupted_logits, dim=-1)
    batch_index = torch.arange(logits.shape[0])
    delta_value = (
        logits[batch_index, -1, first_ids_pos] - corrupted_logits[batch_index, -1, first_ids_pos]
    )
    return delta_value.mean()


pos_corrupted_logit, pos_corrupted_cache = model.run_with_cache_from_embed(
    pos_embs_corrupted
)
pos_clean_logit, pos_clean_cache = model.run_with_cache(pos_input_ids)
neg_corrupted_logit, neg_corrupted_cache = model.run_with_cache_from_embed(
    neg_embs_corrupted
)
neg_clean_logit, neg_clean_cache = model.run_with_cache(neg_input_ids)

pos_metric = partial(
    indirect_effect,
    corrupted_logits=pos_corrupted_logit,
    first_ids_pos=pos_target_ids["target"]
)
neg_metric = partial(
    indirect_effect,
    corrupted_logits=neg_corrupted_logit,
    first_ids_pos=neg_target_ids["orthogonal_token"]
)
print("pos metric", pos_metric(logits=pos_clean_logit))
print("neg metric", neg_metric(logits=neg_clean_logit))

print("Computing results")

pos_result = {
    "example_str_token": model.to_str_tokens(pos_input_ids[0]),
    "logit_lens": logit_lens(
        pos_clean_cache,
        model,
        pos_input_ids,
        pos_target_ids["target"],
        pos_target_ids["orthogonal_token"],
    ),
    "patch_resid_position": patch_resid_pre(
        model,
        pos_input_ids,
        pos_input_ids_corrupted,
        pos_clean_cache,
        pos_metric,
        pos_embs_corrupted
    ),
    "patch_attn_head_out": patch_attn_head_out_all_pos(
        model,
        pos_input_ids,
        pos_input_ids_corrupted,
        pos_clean_cache,
        pos_metric,
        pos_embs_corrupted
    ),
    "patch_attn_head_by_pos": patch_attn_head_by_pos(
        model,
        pos_input_ids,
        pos_input_ids_corrupted,
        pos_clean_cache,
        pos_metric,
        pos_embs_corrupted
    ),
    "patch_per_block": patch_per_block_all_poss(
        model,
        pos_input_ids_corrupted,
        pos_input_ids,
        pos_clean_cache,
        pos_metric,
        1,
        pos_embs_corrupted
    ),
       "patch_mlp_out": get_act_patch_mlp_out(model, pos_input_ids_corrupted, pos_clean_cache, pos_metric, patch_interval=2, corrupted_embeddings=pos_embs_corrupted)
}

neg_result = {
    "example_str_token": model.to_str_tokens(neg_input_ids[0]),
    "logit_lens": logit_lens(
        neg_clean_cache,
        model,
        neg_input_ids,
        neg_target_ids["target"],
        neg_target_ids["orthogonal_token"],
    ),
    "patch_resid_position": patch_resid_pre(
        model,
        neg_input_ids,
        neg_input_ids_corrupted,
        neg_clean_cache,
        neg_metric,
        neg_embs_corrupted
    ),
    "patch_attn_head_out": patch_attn_head_out_all_pos(
        model,
        neg_input_ids,
        neg_input_ids_corrupted,
        neg_clean_cache,
        neg_metric,
        neg_embs_corrupted
    ),
    "patch_attn_head_by_pos": patch_attn_head_by_pos(
        model,
        neg_input_ids,
        neg_input_ids_corrupted,
        neg_clean_cache,
        neg_metric,
        neg_embs_corrupted
    ),
    "patch_per_block": patch_per_block_all_poss(
        model,
        neg_input_ids_corrupted,
        neg_input_ids,
        neg_clean_cache,
        neg_metric,
        1,
        neg_embs_corrupted
    ),
    "patch_mlp_out": get_act_patch_mlp_out(model, neg_input_ids_corrupted, neg_clean_cache, neg_metric, patch_interval=2, corrupted_embeddings=neg_embs_corrupted)
}


## save results
torch.save(pos_result, "../results/pos_result.pt")
torch.save(neg_result, "../results/neg_result.pt")