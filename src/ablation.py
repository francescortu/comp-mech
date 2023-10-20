import transformer_lens.patching as tlens_patch
from functools import partial
import pandas as pd
import transformer_lens.utils as utils
from src.utils import list_of_dicts_to_dict_of_lists

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

Logits = torch.Tensor
AxisNames = Literal["layer", "pos", "head_index", "head", "src_pos", "dest_pos"]

from typing import Sequence
PatchedActivation = torch.Tensor

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

def generic_activation_ablation_stacked(
    model: HookedTransformer,
    input_tokens: Int[torch.Tensor, "batch pos"],
    patching_metric: Callable[
        [Float[torch.Tensor, "batch pos d_vocab"]], Float[torch.Tensor, ""]
    ],
    patch_setter: Callable[
        [ActivationCache, Sequence[int]], PatchedActivation
    ],
    activation_name: str,
    index_axis_names: Optional[Sequence[AxisNames]] = None,
    index_df: Optional[pd.DataFrame] = None,
    return_index_df: bool = False,
    patch_interval: Optional[int] = 1,
    target_ids=None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, pd.DataFrame]]:
    batch_size = input_tokens.shape[0]
    max_len = input_tokens.shape[1]



    # Get the max range for all possible axes
    max_axis_range = {
        "layer": model.cfg.n_layers,
        "pos": input_tokens.shape[-1],
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

    DEVICE = "cpu"
    patched_metric_output = {
        "ablated_probs_mem": torch.zeros(
            index_axis_max_range + [batch_size], device=DEVICE
        ),
        "ablated_probs_cp": torch.zeros(
            index_axis_max_range + [batch_size], device=DEVICE
        ),
        "mem_delta": torch.zeros(
            index_axis_max_range + [batch_size], device=DEVICE
        ),
        "cp_delta": torch.zeros(
            index_axis_max_range + [batch_size], device=DEVICE
        ),
        "kl-mean": torch.zeros(index_axis_max_range, device=DEVICE),
        "kl-std": torch.zeros(index_axis_max_range, device=DEVICE),
        "loss": torch.zeros(index_axis_max_range, device=DEVICE),
    }
    
    def ablating_hook(activation, hook, index):
        return patch_setter(activation, index)
    
    for c, index_row in enumerate(tqdm((list(index_df.iterrows())))):
        # print("-------------------")
        index = index_row[1].to_list()  # (layer, pos)

        if patch_interval > 1:
            if index[0] + patch_interval > model.cfg.n_layers:
                continue
            # create a tuple from index[0] to index[0] + stack_patch_size
            layer = tuple(range(index[0], index[0] + patch_interval))
            index = [layer, index[1]]

            hooks = []
            for l in index[0]:
                current_activation_name = utils.get_act_name(activation_name, layer=l)
                current_hook = partial(
                    ablating_hook,
                    index=index,
                )
                    
                # print(current_activation_name)
                hooks.append((current_activation_name, current_hook))
            # Run the model with the patching hook and get the logits!

        else:
            # The current activation name is just the activation name plus the layer (assumed to be the first element of the input)
            current_activation_name = utils.get_act_name(
                activation_name, layer=index[0]
            )

            # The hook function cannot receive additional inputs, so we use partial to include the specific index and the corresponding clean activation
            current_hook = partial(
                ablating_hook,
                index=index,
            )
            hooks = [(current_activation_name, current_hook)]

        # Run the model with the patching hook and get the logits!
        
        patched_logits = model.run_with_hooks(input_tokens, fwd_hooks=hooks)
        loss = model.run_with_hooks(input_tokens, fwd_hooks=hooks, return_type="loss")

        # Calculate the patching metric and store

            # check if index[0] is int or tuple
        if isinstance(index[0], Tuple):
            index[0] = int(index[0][0])
            
        
        # from two tensor of shape [batch_size]
        output_metric = patching_metric(patched_logits, target_ids)
        patched_metric_output["ablated_probs_mem"][tuple(index)][:] = (
            torch.log_softmax(patched_logits, dim=-1)[:, -1, :].gather(-1, index=target_ids["mem_token"]).squeeze(-1).to(DEVICE)
        )
        patched_metric_output["ablated_probs_cp"][tuple(index)][:] = (
            torch.log_softmax(patched_logits, dim=-1)[:, -1, :].gather(-1, index=target_ids["cp_token"]).squeeze(-1).to(DEVICE)
        )
        patched_metric_output["mem_delta"][tuple(index)] = output_metric[
            "mem_delta"
        ].to(DEVICE)
        patched_metric_output["cp_delta"][tuple(index)] = output_metric[
            "cp_delta"
        ].to(DEVICE)
        patched_metric_output["kl-mean"][tuple(index)] = (
            output_metric["kl-mean"].to(DEVICE).item()
        )
        patched_metric_output["kl-std"][tuple(index)] = (
            output_metric["kl-std"].to(DEVICE).item()
        )
        patched_metric_output["loss"][tuple(index)] = loss.to(DEVICE).item()

    if return_index_df:
        return patched_metric_output, index_df
    else:
        return patched_metric_output


def layer_pos_patch_setter(corrupted_activation, index):
    """
    Applies the activation patch where index = [layer, pos]

    Implicitly assumes that the activation axis order is [batch, pos, ...], which is true of everything that is not an attention pattern shaped tensor.
    """
    assert len(index) == 2
    layer, pos = index
    corrupted_activation[:, pos, ...] = 0
    return corrupted_activation

def layer_pos_head_vector_patch_setter(
    corrupted_activation,
    index
):
    """
    Applies the activation patch where index = [layer, pos, head_index]

    Implicitly assumes that the activation axis order is [batch, pos, head_index, ...], which is true of all attention head vector activations (q, k, v, z, result) but *not* of attention patterns.
    """
    assert len(index) == 3
    layer, pos, head_index = index
    corrupted_activation[:, pos, head_index] = 0
    return corrupted_activation


def layer_head_vector_patch_setter(
    corrupted_activation,
    index
):
    """
    Applies the activation patch where index = [layer,  head_index]

    Implicitly assumes that the activation axis order is [batch, pos, head_index, ...], which is true of all attention head vector activations (q, k, v, z, result) but *not* of attention patterns.
    """
    assert len(index) == 2
    layer, head_index = index
    corrupted_activation[:, :, head_index] = 0

    return corrupted_activation
    
get_act_ablation_mlp_out = partial(
    generic_activation_ablation_stacked,
    patch_setter=layer_pos_patch_setter,
    activation_name="mlp_out",
    index_axis_names=("layer", "pos"),
)

get_act_ablation_attn_out = partial(
    generic_activation_ablation_stacked,
    patch_setter=layer_pos_patch_setter,
    activation_name="attn_out",
    index_axis_names=("layer", "pos"),
)

get_act_ablation_resid_pre = partial(
    generic_activation_ablation_stacked,
    patch_setter=layer_pos_patch_setter,
    activation_name="resid_pre",
    index_axis_names=("layer", "pos"),
)

get_act_ablation_attn_head_out_all_pos = partial(
    generic_activation_ablation_stacked,
    patch_setter=layer_head_vector_patch_setter,
    activation_name="z",
    index_axis_names=("layer", "head"),
)

get_act_ablation_attn_head_out_by_pos = partial(
    generic_activation_ablation_stacked,
    patch_setter=layer_pos_head_vector_patch_setter,
    activation_name="z",
    index_axis_names=("layer", "pos", "head"),
)

from src.model import WrapHookedTransformer
from src.dataset import Dataset

def kl_divergence(logit, logit_clean):
    logprobs = torch.nn.functional.log_softmax(logit, dim=-1)
    logprob_clean = torch.nn.functional.log_softmax(logit_clean, dim=-1)
    batch_size = logprobs.shape[0]
    result = torch.zeros(batch_size, device=logprobs.device)
    for i in range(batch_size):
        result[i] = torch.nn.functional.kl_div(
            logprobs[i, -1, :], logprob_clean[i, -1, :], reduction="sum", log_target=True
        )
    return result

def ablation_metric(logits: torch.Tensor,target_ids:dict, clean_logits:torch.Tensor):
    probs = torch.log_softmax(logits, dim=-1)
    clean_probs = torch.log_softmax(clean_logits, dim=-1)
    
    mem_prob = torch.gather(probs[:, -1, :], 1, target_ids["mem_token"]).squeeze()
    cp_probs = torch.gather(probs[:, -1, :], 1, target_ids["cp_token"]).squeeze()
    
    mem_clean_probs = torch.gather(clean_probs[:, -1, :], 1, target_ids["mem_token"]).squeeze()
    cp_clean_probs = torch.gather(clean_probs[:, -1, :], 1, target_ids["cp_token"]).squeeze()
    
    mem_delta =mem_clean_probs - mem_prob
    cp_delta = cp_clean_probs - cp_probs
    kl_div = kl_divergence(clean_logits, logits)
    return {
        "mem_delta": mem_delta,
        "cp_delta": cp_delta,
        "kl-mean": kl_div.mean(),
        "kl-std": kl_div.std(),
    }
    


class Ablator():
    def __init__(self, config):
        self.config = config
        self.model, self.dataset = self.load_model_and_data()
    
    def load_model_and_data(self) -> Tuple[WrapHookedTransformer, Dataset]:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        model = WrapHookedTransformer.from_pretrained(self.config.model_name, device=DEVICE)
        dataset = Dataset(f"../data/{self.config.name_dataset}")
        dataset.select_lenght(self.config.n, self.config.length)
        dataset.print_statistics()
        return model, dataset
    
    def process_batch(self, batch):
        target_ids = {
            "mem_token": self.model.to_tokens(batch["target"], prepend_bos=False),
            "cp_token": self.model.to_tokens(batch["orthogonal_token"], prepend_bos=False),
        }
        input_ids = self.model.to_tokens(batch["premise"])

        clean_logit = self.model(input_ids)
        clean_probs = torch.log_softmax(clean_logit, dim=-1)[:,-1,:]
        clean_probs_mem_token = clean_probs.gather(-1, index=target_ids["mem_token"]).squeeze(-1)
        clean_probs_orthogonal_token = clean_probs.gather(-1, index=target_ids["cp_token"]).squeeze(-1)
        
        metric = partial(ablation_metric, clean_logits=clean_logit)
        result = {
            "attn_head_out":get_act_ablation_attn_head_out_all_pos(
                model=self.model,
                input_tokens=input_ids,
                patching_metric=metric,
                patch_interval=self.config.patch_interval,
                target_ids=target_ids,
                ),
            "mlp_out":get_act_ablation_mlp_out(
                model=self.model,
                input_tokens=input_ids,
                patching_metric=metric,
                patch_interval=self.config.patch_interval,
                target_ids=target_ids,),
            "attn_out_by_pos":get_act_ablation_attn_out(
                                model=self.model,
                input_tokens=input_ids,
                patching_metric=metric,
                patch_interval=self.config.patch_interval,
                target_ids=target_ids,
                ),
            "premise": batch["premise"],
            "clean_probs_mem": clean_probs_mem_token,
            "clean_probs_cp": clean_probs_orthogonal_token,
            }
        return result
    
    def ablate(self):
        # ablate mem dataset
        print("ablate mem dataset for length: ", self.config.length)
        self.dataset.select_dataset("mem")
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=False)
        mem_results = []
        for batch in dataloader:
            mem_result = self.process_batch(batch)
            mem_results.append(mem_result)
            
        # ablate cp dataset
        print("ablate cp dataset for length: ", self.config.length)
        self.dataset.select_dataset("cp")
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=False)
        cp_results = []
        for batch in dataloader:
            cp_result = self.process_batch(batch)
            cp_results.append(cp_result)
            
        # save results
        full_result = {
            "mem": list_of_dicts_to_dict_of_lists(mem_results),
            "cp": list_of_dicts_to_dict_of_lists(cp_results)
        }
        
        for key in full_result.keys():
            for subkey in full_result[key].keys():
                if subkey in ["attn_head_out", "attn_out_by_pos",  "mlp_out"]:
                    full_result[key][subkey] = {k: [d[k] for d in full_result[key][subkey]] for k in full_result[key][subkey][0].keys()}
                    for subsubkey in full_result[key][subkey].keys():
                        if subsubkey in ['ablated_probs_mem', 'ablated_probs_cp', 'mem_delta', 'cp_delta']:
                            # here we have list of tensor of shape (component, component, batch)
                            full_result[key][subkey][subsubkey] = torch.cat(full_result[key][subkey][subsubkey], dim=-1)
                            full_result[key][subkey][subsubkey] = einops.rearrange(full_result[key][subkey][subsubkey], 'c1 c2 b -> b c1 c2')
                        else:
                            #here we have list of tensor of shape (component, component)
                            full_result[key][subkey][subsubkey] = torch.stack(full_result[key][subkey][subsubkey])
                if subkey in ["clean_probs_mem", "clean_probs_cp"]:
                    full_result[key][subkey] = torch.cat(full_result[key][subkey], dim=0)
       
    
        torch.save(full_result, f"../results/locate_mechanism/{self.config.name_save_file}.pt")