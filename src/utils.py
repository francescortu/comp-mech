import torch
from transformer_lens import HookedTransformer

# from src.model import WrapHookedTransformer
from src.patching import (
    get_act_patch_block_every,
    get_act_patch_resid_pre,
    get_act_patch_attn_head_out_all_pos,
    get_act_patch_attn_head_out_by_pos
)
import transformer_lens.patching as patching

from typing import List
import warnings
import einops


class C:
    """Color class for printing colored text in terminal"""

    RED = "\033[91m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    END = "\033[0m"


def get_predictions(model, logits, k, return_type):
    if return_type == "probabilities":
        logits = torch.softmax(logits, dim=-1)

    prediction_tkn_ids = logits[0, -1, :].topk(k).indices.cpu().detach().numpy()
    prediction_tkns = [model.to_string(tkn_id) for tkn_id in prediction_tkn_ids]
    best_logits = logits[0, -1, prediction_tkn_ids]

    return best_logits, prediction_tkns


def squeeze_last_dims(tensor):
    if len(tensor.shape) == 3 and tensor.shape[1] == 1 and tensor.shape[2] == 1:
        return tensor.squeeze(-1).squeeze(-1)
    if len(tensor.shape) == 2 and tensor.shape[1] == 1:
        return tensor.squeeze(-1)
    else:
        return tensor


def suppress_warnings(fn):
    def wrapper(*args, **kwargs):
        # Save the current warnings state
        current_filters = warnings.filters[:]
        warnings.filterwarnings("ignore")
        try:
            return fn(*args, **kwargs)
        finally:
            # Restore the warnings state
            warnings.filters = current_filters

    return wrapper


def embs_to_tokens_ids(noisy_embs, model):
    input_embedding_norm = torch.functional.F.normalize(noisy_embs, p=2, dim=2)
    embedding_matrix_norm = torch.functional.F.normalize(model.W_E, p=2, dim=1)
    similarity = torch.matmul(input_embedding_norm, embedding_matrix_norm.T)
    corrupted_tokens = torch.argmax(similarity, dim=2)
    return corrupted_tokens


def patch_resid_pre(model, input_ids, input_ids_corrupted, clean_cache, metric, corrupted_embeddings):
    resid_pre_act_patch_results = get_act_patch_resid_pre(
        model = model, corrupted_tokens=input_ids_corrupted, clean_cache=clean_cache, patching_metric= metric, corrupted_embeddings=corrupted_embeddings
    )
    return resid_pre_act_patch_results


def patch_attn_head_out_all_pos(
    model, input_ids, input_ids_corrupted, clean_cache, metric, corrupted_embeddings
):
    attn_head_out_all_pos_act_patch_results = get_act_patch_attn_head_out_all_pos(
        model,
        input_ids_corrupted,
        clean_cache,
        metric,
        corrupted_embeddings=corrupted_embeddings,
    )
    return attn_head_out_all_pos_act_patch_results


def patch_attn_head_by_pos(model, input_ids, input_ids_corrupted, clean_cache, metric, corrupted_embeddings):
    ALL_HEAD_LABELS = [
        f"L{i}H{j}" for i in range(model.cfg.n_layers) for j in range(model.cfg.n_heads)
    ]
    import einops

    attn_head_out_act_patch_results = get_act_patch_attn_head_out_by_pos(
        model = model, corrupted_tokens=input_ids_corrupted, clean_cache=clean_cache, patching_metric= metric, corrupted_embeddings=corrupted_embeddings

    )
    attn_head_out_act_patch_results = einops.rearrange(
        attn_head_out_act_patch_results, "layer pos head -> (layer head) pos"
    )
    return attn_head_out_act_patch_results


def patch_per_block_all_poss(
    model,
    input_ids_corrupted,
    input_ids,
    clean_cache,
    metric,
    interval,
    corrupted_embeddings,
):
    every_block_result = get_act_patch_block_every(
        model,
        input_ids_corrupted,
        clean_cache,
        metric,
        patch_interval=interval,
        corrupted_embeddings=corrupted_embeddings,
    )
    return every_block_result


def patch_attn_head_all_pos_every(
    model, input_ids_corrupted, input_ids, clean_cache, metric
):
    every_head_all_pos_act_patch_result = (
        patching.get_act_patch_attn_head_all_pos_every(
            model, input_ids_corrupted, clean_cache, metric
        )
    )
    return every_head_all_pos_act_patch_result


def logit_lens(cache, model, input_ids, target_ids, orthogonal_ids):
    accumulater_residual, labels = cache.accumulated_resid(
        layer=-1, incl_mid=True, return_labels=True, pos_slice=-1
    )
    accumulater_residual = cache.apply_ln_to_stack(
        accumulater_residual, layer=-1, pos_slice=-1
    )
    unembed_accumulated_residual = einops.einsum(
        accumulater_residual,
        model.W_U,
        "n_comp batch d_model, d_model vocab -> n_comp batch vocab",
    )
    batch_index = torch.arange(input_ids.shape[0])
    target_lens = (
        unembed_accumulated_residual[:, batch_index, target_ids]
        .mean(-1)
        .detach()
        .cpu()
        .numpy()
    )
    orthogonal_lens = (
        unembed_accumulated_residual[:, batch_index, orthogonal_ids]
        .mean(-1)
        .detach()
        .cpu()
        .numpy()
    )

    return {
        "target_lens": target_lens,
        "orthogonal_lens": orthogonal_lens,
        "labels": labels,
    }


def list_of_dicts_to_dict_of_lists(list_of_dicts):
    # Initialize an empty dictionary to store the result
    dict_of_lists = {}
    
    # Iterate over each dictionary in the list
    for d in list_of_dicts:
        # Iterate over each key-value pair in the dictionary
        for key, value in d.items():
            # If the key is not already in the result dictionary, add it with an empty list as its value
            if key not in dict_of_lists:
                dict_of_lists[key] = []
            # Append the value to the list corresponding to the key in the result dictionary
            dict_of_lists[key].append(value)
    
    return dict_of_lists

def dict_of_lists_to_dict_of_tensors(dict_of_lists):
    dict_of_tensors = {}
    for key, tensor_list in dict_of_lists.items():
        dict_of_tensors[key] = torch.stack(tensor_list)
    return dict_of_tensors