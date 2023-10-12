import transformer_lens.patching as tlens_patch
from functools import partial
import pandas as pd
import transformer_lens.utils as utils


import itertools
from functools import partial
from typing import Callable, Optional, Sequence, Tuple, Union

import einops
import pandas as pd
import torch
from jaxtyping import Float, Int
from tqdm.auto import tqdm
from typing_extensions import Literal

import transformer_lens.utils as utils
from transformer_lens import ActivationCache, HookedTransformer

# %%
Logits = torch.Tensor
AxisNames = Literal["layer", "pos", "head_index", "head", "src_pos", "dest_pos"]

# %%


# %%
from typing import Sequence


def make_df_from_ranges(
    column_max_ranges: Sequence[int], column_names: Sequence[str]
) -> pd.DataFrame:
    """
    Takes in a list of column names and max ranges for each column, and returns a dataframe with the cartesian product of the range for each column (ie iterating through all combinations from zero to column_max_range - 1, in order, incrementing the final column first)
    """
    rows = list(
        itertools.product(
            *[range(axis_max_range) for axis_max_range in column_max_ranges]
        )
    )
    df = pd.DataFrame(rows, columns=column_names)
    return df


# %%
CorruptedActivation = torch.Tensor
PatchedActivation = torch.Tensor


# %%
from typing import Sequence




def generic_activation_patch_stacked(
    model: HookedTransformer,
    corrupted_tokens: Int[torch.Tensor, "batch pos"],
    clean_cache: ActivationCache,
    patching_metric: Callable[
        [Float[torch.Tensor, "batch pos d_vocab"]], Float[torch.Tensor, ""]
    ],
    patch_setter: Callable[
        [CorruptedActivation, Sequence[int], ActivationCache], PatchedActivation
    ],
    activation_name: str,
    corrupted_embeddings: Float[torch.Tensor, "batch pos d_model"] = None,
    index_axis_names: Optional[Sequence[AxisNames]] = None,
    index_df: Optional[pd.DataFrame] = None,
    return_index_df: bool = False,
    patch_interval: Optional[int] = 1,
) -> Union[torch.Tensor, Tuple[torch.Tensor, pd.DataFrame]]:
    """
    A generic function to do activation patching, will be specialised to specific use cases.

    Activation patching is about studying the counterfactual effect of a specific activation between a clean run and a corrupted run. The idea is have two inputs, clean and corrupted, which have two different outputs, and differ in some key detail. Eg "The Eiffel Tower is in" vs "The Colosseum is in". Then to take a cached set of activations from the "clean" run, and a set of corrupted.

    Internally, the key function comes from three things: A list of tuples of indices (eg (layer, position, head_index)), a index_to_act_name function which identifies the right activation for each index, a patch_setter function which takes the corrupted activation, the index and the clean cache, and a metric for how well the patched model has recovered.

    The indices can either be given explicitly as a pandas dataframe, or by listing the relevant axis names and having them inferred from the tokens and the model config. It is assumed that the first column is always layer.

    This function then iterates over every tuple of indices, does the relevant patch, and stores it

    Args:
        model: The relevant model
        corrupted_tokens: The input tokens for the corrupted run
        clean_cache: The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)
        patch_setter: A function which acts on (corrupted_activation, index, clean_cache) to edit the activation and patch in the relevant chunk of the clean activation
        activation_name: The name of the activation being patched
        index_axis_names: The names of the axes to (fully) iterate over, implicitly fills in index_df
        index_df: The dataframe of indices, columns are axis names and each row is a tuple of indices. Will be inferred from index_axis_names if not given. When this is input, the output will be a flattened tensor with an element per row of index_df
        return_index_df: A Boolean flag for whether to return the dataframe of indices too

    Returns:
        patched_output: The tensor of the patching metric for each patch. By default it has one dimension for each index dimension, via index_df set explicitly it is flattened with one element per row.
        index_df *optional*: The dataframe of indices
    """
    batch_size = corrupted_tokens.shape[0]
    max_len = corrupted_tokens.shape[1]
    if index_df is None:
        assert index_axis_names is not None

        # Get the max range for all possible axes
        max_axis_range = {
            "layer": model.cfg.n_layers,
            "pos": corrupted_tokens.shape[-1],
            "head_index": model.cfg.n_heads,
        }
        max_axis_range["src_pos"] = max_axis_range["pos"]
        max_axis_range["dest_pos"] = max_axis_range["pos"]
        max_axis_range["head"] = max_axis_range["head_index"]

        # Get the max range for each axis we iterate over
        index_axis_max_range = [
            max_axis_range[axis_name] for axis_name in index_axis_names
        ]

        # Get the dataframe where each row is a tuple of indices
        index_df = make_df_from_ranges(index_axis_max_range, index_axis_names)

        flattened_output = False
    else:
        # A dataframe of indices was provided. Verify that we did not *also* receive index_axis_names
        assert index_axis_names is None
        index_axis_max_range = index_df.max().to_list()

        flattened_output = True

    # Create an empty tensor to show the patched metric for each patch
    if flattened_output:
        patched_metric_output = {
           "mean": torch.zeros(len(index_df), device=model.cfg.device),
           "std": torch.zeros(len(index_df), device=model.cfg.device),
           "t-value": torch.zeros(len(index_df), device=model.cfg.device),
           "p-value": torch.zeros(len(index_df), device=model.cfg.device),
           "patched_logits": torch.zeros(len(index_df), device=model.cfg.device),
        }
    else:
        # DEVICE = model.cfg.device
        DEVICE = "cpu"
        patched_metric_output = {
            "mean": torch.zeros(index_axis_max_range, device=DEVICE),
            "std": torch.zeros(index_axis_max_range, device=DEVICE),
            "t-value": torch.zeros(index_axis_max_range, device=DEVICE),
            "p-value": torch.zeros(index_axis_max_range, device=DEVICE),
            "patched_logits": torch.zeros(index_axis_max_range + [batch_size, model.cfg.d_vocab], device=DEVICE),
            "full_delta": torch.zeros(index_axis_max_range + [batch_size], device=DEVICE),
        }

    # A generic patching hook - for each index, it applies the patch_setter appropriately to patch the activation
    def patching_hook(corrupted_activation, hook, index, clean_activation):
        return patch_setter(corrupted_activation, index, clean_activation)


    # Iterate over every list of indices, and make the appropriate patch!
    for c, index_row in enumerate(tqdm((list(index_df.iterrows())))):
        # print("-------------------")
        index = index_row[1].to_list() # (layer, pos)

        if patch_interval > 1:
            if index[0] + patch_interval > model.cfg.n_layers:
                continue
            # create a tuple from index[0] to index[0] + stack_patch_size
            layer = tuple(range(index[0], index[0] + patch_interval))
            index = [ layer, index[1]]

            
            hooks = []
            for l in index[0]:
                current_activation_name = utils.get_act_name(activation_name, layer=l)
                current_hook = partial(
                    patching_hook,
                    index=index,
                    clean_activation=clean_cache[current_activation_name],
                )
                # print(current_activation_name)
                hooks.append((current_activation_name, current_hook))
            # Run the model with the patching hook and get the logits!
        
        else:
        # The current activation name is just the activation name plus the layer (assumed to be the first element of the input)
            current_activation_name = utils.get_act_name(activation_name, layer=index[0])

        # The hook function cannot receive additional inputs, so we use partial to include the specific index and the corresponding clean activation
            current_hook = partial(
                patching_hook,
                index=index,
                clean_activation=clean_cache[current_activation_name],
            )
            hooks = [(current_activation_name, current_hook)]

        # Run the model with the patching hook and get the logits!
        if corrupted_embeddings is not None:
            def embed_hook(cache, hook, corrupted_embeddings):
                cache[:,:,:] = corrupted_embeddings
                return cache
            embeds_hook = partial(embed_hook, corrupted_embeddings=corrupted_embeddings)
            hooks.append(("hook_embed", embeds_hook))
            
        patched_logits = model.run_with_hooks(
            corrupted_tokens, fwd_hooks=hooks
        )

        # Calculate the patching metric and store
        if flattened_output:
            patched_metric_output[c] = patching_metric(patched_logits).item()
        else:
            output_metric = patching_metric(patched_logits)
            patched_metric_output["mean"][tuple(index)] = output_metric["mean"].to(DEVICE).item()
            patched_metric_output["std"][tuple(index)] = output_metric["std"].to(DEVICE).item()
            patched_metric_output["t-value"][tuple(index)] = output_metric["t-value"].to(DEVICE).item()
            patched_metric_output["p-value"][tuple(index)] = output_metric["p-value"].to(DEVICE).item()
            patched_metric_output["patched_logits"][tuple(index)][:,:] = patched_logits[:,-1,:].to(DEVICE)
            patched_metric_output["full_delta"][tuple(index)][:] = output_metric["full_delta"].to(DEVICE)


    if return_index_df:
        return patched_metric_output, index_df
    else:
        return patched_metric_output
    
get_act_patch_mlp_out= partial(
    generic_activation_patch_stacked,
    patch_setter=tlens_patch.layer_pos_patch_setter,
    activation_name="mlp_out",
    index_axis_names=("layer", "pos"),
)

get_act_patch_attn_out = partial(
    generic_activation_patch_stacked,
    patch_setter=tlens_patch.layer_pos_patch_setter,
    activation_name="attn_out",
    index_axis_names=("layer", "pos"),
)

get_act_patch_resid_pre = partial(
    generic_activation_patch_stacked,
    patch_setter=tlens_patch.layer_pos_patch_setter,
    activation_name="resid_pre",
    index_axis_names=("layer", "pos"),
)

get_act_patch_attn_head_out_all_pos = partial(
    generic_activation_patch_stacked,
    patch_setter=tlens_patch.layer_head_vector_patch_setter,
    activation_name="z",
    index_axis_names=("layer", "head"),
)

get_act_patch_attn_head_out_by_pos = partial(
    generic_activation_patch_stacked,
    patch_setter=tlens_patch.layer_pos_head_vector_patch_setter,
    activation_name="z",
    index_axis_names=("layer", "pos", "head"),
)



def get_act_patch_block_every(
    model, corrupted_tokens, clean_cache, metric, patch_interval=1, corrupted_embeddings=None
) -> Float[torch.Tensor, "patch_type layer pos"]:
    """Helper function to get activation patching results for the residual stream (at the start of each block), output of each Attention layer and output of each MLP layer. Wrapper around each's patching function, returns a stacked tensor of shape [3, n_layers, pos]

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [3, n_layers, pos]
    """
    act_patch_results = []
    act_patch_results.append(
        get_act_patch_resid_pre(model, corrupted_tokens, clean_cache, metric, corrupted_embeddings=corrupted_embeddings)
    )
    act_patch_results.append(
        get_act_patch_attn_out(model, corrupted_tokens, clean_cache, metric, patch_interval=patch_interval, corrupted_embeddings=corrupted_embeddings)
    )
    act_patch_results.append(
        get_act_patch_mlp_out(model, corrupted_tokens, clean_cache, metric, patch_interval=patch_interval, corrupted_embeddings=corrupted_embeddings)
    )
    return torch.stack(act_patch_results, dim=0)

