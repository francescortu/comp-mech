from src.utils import (
    embs_to_tokens_ids,
)
import transformer_lens.patching as patching
import torch
from src.patching import (
    get_act_patch_block_every,
    get_act_patch_resid_pre,
    get_act_patch_attn_head_out_all_pos,
    get_act_patch_attn_head_out_by_pos,
    get_act_patch_attn_out
)
import einops
from src.patching import get_act_patch_mlp_out
from scipy.stats import ttest_1samp

def indirect_effect(logits, corrupted_logits, first_ids_pos):
    logits = torch.nn.functional.softmax(logits, dim=-1)
    corrupted_logits = torch.nn.functional.softmax(corrupted_logits, dim=-1)
    # Use torch.gather to get the desired values
    logits_values = torch.gather(logits[:, -1, :], 1, first_ids_pos).squeeze()
    corrupted_logits_values = torch.gather(corrupted_logits[:, -1, :], 1, first_ids_pos).squeeze()
    
    delta_value = logits_values - corrupted_logits_values
    ttest = ttest_1samp(delta_value.cpu().detach().numpy(), 0)
    return {
        "mean": delta_value.mean(),
        "std": delta_value.std(),
        "t-value": torch.tensor(ttest[0], device=delta_value.device),
        "p-value": torch.tensor(ttest[1], device=delta_value.device),
        "full_delta": delta_value,
    }


def patch_attn_out_by_pos(model, input_ids, input_ids_corrupted, clean_cache, metric, corrupted_embedding, interval=1):
    return get_act_patch_attn_out(
        model=model, corrupted_tokens=input_ids_corrupted, clean_cache=clean_cache, patching_metric=metric, corrupted_embeddings=corrupted_embedding, patch_interval=interval
    )

def patch_resid_pre(model, input_ids, input_ids_corrupted, clean_cache, metric, corrupted_embeddings, interval=1):
    resid_pre_act_patch_results = get_act_patch_resid_pre(
        model = model, corrupted_tokens=input_ids_corrupted, clean_cache=clean_cache, patching_metric= metric, corrupted_embeddings=corrupted_embeddings, interval=interval
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


def logit_lens(cache, model, input_ids, target_ids):
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
    )


    return target_lens

def wrapper_patch_attention_out_by_pos(shared_args):
    return patch_attn_out_by_pos(
        shared_args["model"],
        shared_args["input_ids"],
        shared_args["input_ids"],
        shared_args["clean_cache"],
        shared_args["metric"],
        shared_args["embs_corrupted"],
        shared_args["interval"],
    )

def wrapper_logit_lens_mem(shared_args):
    return logit_lens(
        shared_args["clean_cache"],
        shared_args["model"],
        shared_args["input_ids"],
        shared_args["target_ids"]["target"],
    )
    
def wrapper_logit_lens_cp(shared_args):
    return logit_lens(
        shared_args["clean_cache"],
        shared_args["model"],
        shared_args["input_ids"],
        shared_args["target_ids"]["orthogonal"],
    )
    
def wrapper_patch_resid_pre(shared_args):
    return patch_resid_pre(
        shared_args["model"],
        shared_args["input_ids"],
        shared_args["input_ids"],
        shared_args["clean_cache"],
        shared_args["metric"],
        shared_args["embs_corrupted"],
        shared_args["interval"],
    )
def wrapper_patch_attn_head_out_all_pos(shared_args):
    return patch_attn_head_out_all_pos(
        shared_args["model"],
        shared_args["input_ids"],
        shared_args["input_ids"],
        shared_args["clean_cache"],
        shared_args["metric"],
        shared_args["embs_corrupted"],
    )
def wrapper_patch_attn_head_by_pos(shared_args):
    return patch_attn_head_by_pos(
        shared_args["model"],
        shared_args["input_ids"],
        shared_args["input_ids"],
        shared_args["clean_cache"],
        shared_args["metric"],
        shared_args["embs_corrupted"],
    )
def wrapper_patch_per_block_all_poss(shared_args):
    return patch_per_block_all_poss(
        shared_args["model"],
        shared_args["input_ids"],
        shared_args["input_ids"],
        shared_args["clean_cache"],
        shared_args["metric"],
        shared_args["interval"],
        shared_args["embs_corrupted"],
    )

def wrap_patch_mlp_out(shared_args):
    return get_act_patch_mlp_out(
        model = shared_args["model"], 
        corrupted_tokens = shared_args["input_ids"], 
        clean_cache = shared_args["clean_cache"], 
        patching_metric=shared_args["metric"], 
        patch_interval=shared_args["interval"],
        corrupted_embeddings=shared_args["embs_corrupted"])
    
COMPUTE_WRAPPER = {
    "logit_lens_mem": wrapper_logit_lens_mem,
    "logit_lens_cp": wrapper_logit_lens_cp,
    "resid_pos": wrapper_patch_resid_pre,
    "attn_head_out": wrapper_patch_attn_head_out_all_pos,
    "attn_head_by_pos": wrapper_patch_attn_head_by_pos,
    "per_block": wrapper_patch_per_block_all_poss,
    "mlp_out": wrap_patch_mlp_out,
    "attn_out_by_pos": wrapper_patch_attention_out_by_pos,
}

def construct_result_dict(shared_args, keys_to_compute):
    result_dict = {}
    print("Computing the following keys: {}".format(keys_to_compute))
    for key in keys_to_compute:
        if key in COMPUTE_WRAPPER:
            result_dict[key] = COMPUTE_WRAPPER[key](shared_args)
    return result_dict