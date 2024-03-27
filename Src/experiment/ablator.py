from operator import index
from re import sub
import re
from more_itertools import prepend
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from Src.dataset import BaseDataset
from Src.model import BaseModel
from Src.base_experiment import BaseExperiment, to_logit_token
from typing import Callable, Optional, Tuple, Dict, Any, Literal, Union, List
import pandas as pd
from Src.experiment import LogitStorage, HeadLogitStorage
from functools import partial
from copy import deepcopy
import inspect

def get_partial_arg_names(partial_func):
    # Get the original function from the partial object
    original_func = partial_func.func
    
    # Get the signature of the original function
    sig = inspect.signature(original_func)
    
    # Get a list of the parameter names of the original function
    param_names = list(sig.parameters.keys())
    
    # Initialize lists to hold the names of fixed positional and keyword arguments
    fixed_positional_arg_names = []
    fixed_keyword_arg_names = list(partial_func.keywords.keys())
    
    # Iterate over the fixed positional arguments
    for i, arg in enumerate(partial_func.args):
        if i < len(param_names):
            fixed_positional_arg_names.append(param_names[i])
        else:
            # This handles cases where there are more positional arguments than the original function expects
            fixed_positional_arg_names.append(f"arg_{i}")
    
    return fixed_positional_arg_names, fixed_keyword_arg_names


class Ablator(BaseExperiment):
    def __init__(
        self,
        dataset: BaseDataset,
        model: BaseModel,
        batch_size: int,
        experiment: Literal["copyVSfact"],
        total_effect: bool = True,
    ):
        super().__init__(dataset, model, batch_size, experiment)
        self.total_effect = total_effect
        self.hooks = []

    def set_heads(self, heads:List[Tuple[int,int]], position: Literal["all", "attribute"] = "attribute", value: float = 0.0):
        """
        heads: list of tuples (layer, head) to ablate
        position: "all" or "attribute" to ablate all the entries or only the attribute entries
        value: value to multiply the ablated entries (default 0.0) pattern <- value * pattern
        """
        self.reset_hooks()
        def hook_fn(attention_pattern:torch.Tensor,hook, value:float, position: Literal["all", "attribute"], batch, head:int):
            if position == "all":
                attention_pattern[:,head, -1, :] = value * attention_pattern[:,head, -1, :]
                return attention_pattern
            elif position == "attribute":
                attribute_positions = batch["obj_pos"]
                #for each element in the batch and for each head, ablate the attention pattern with the corresponding position of the attribute
                for i, pos in enumerate(attribute_positions):
                    attention_pattern[i, head, -1, pos] = value * attention_pattern[i, head, -1, pos]
                return attention_pattern
            else:
                raise ValueError("position must be 'all' or 'attribute'")
            
        for layer, head in heads:
            self.hooks.append((f"blocks.{layer}.attn.hook_attn_scores", partial(hook_fn, value=value, position=position, head=head)))
            
                
    def reset_hooks(self):
        self.hooks = []            
    
    def set_hooks(self, hooks) -> None:
        self.hooks = hooks

    def __run_with_hooks__(self, batch):
        self.model.reset_hooks()
        # with torch.no_grad():
            #re-write hooks: if the hook are batch dependent, we should pass the batch to the hook, otherwise we can just pass the hook
        def hook_fn(hook, batch):
            #if batch is a argument of the hook, we pass it
            if "batch" in list(inspect.signature(hook).parameters.keys()):
                return partial(hook, batch=batch)
            else:
                return hook
        actual_hooks = []
        for hook_name, hook in self.hooks:
            actual_hooks.append((hook_name, hook_fn(hook, batch)))
        
        logit = self.model.run_with_hooks(
            batch["prompt"], prepend_bos=False, fwd_hooks=actual_hooks
        )

        return logit[:, -1, :]
    
    def ablate_length(
        self, length: int, normalize_logit: Literal["none", "softmax", "sigmoid"] = "none"
    ):
        """
        Apply the hooks to the model for the given length and return the logit of the model
        """
        mem = []
        cp = []
        diff = []
        mem_winners = []
        cp_winners = []
        self.dataset.set_len(length)
        for batch in DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=False,
        ):
            logit = self.__run_with_hooks__(batch)
            logit_mem, logit_cp, _, _, mem_winner, cp_winner = to_logit_token(
                logit, batch["target"], normalize=normalize_logit, return_winners=True
            )
            mem.append(logit_mem)
            cp.append(logit_cp)
            diff.append(logit_mem - logit_cp)
            mem_winners.append(mem_winner)
            cp_winners.append(cp_winner)
            
        return torch.cat(mem), torch.cat(cp), torch.cat(diff), torch.cat(mem_winners), torch.cat(cp_winners)
        
    
    def ablate(self, normalize_logit: Literal["none", "softmax", "sigmoid"] = "none"):
        """
        Apply the hooks to the model for all the lengths and return the logit of the model along with the count of the examples where the model predicted the factual and counterfactual token
        
        Returns:
        - mem: logit of the model for the factual token
        - cp: logit of the model for the counterfactual token
        - diff: difference between the logit of the model for the factual and counterfactual token
        - mem_win: count of the examples where the model predicted the factual token
        - cp_win: count of the examples where the model predicted the counterfactual token
        """
        lengths = self.dataset.get_lengths()
        result = {"mem": [], "cp": [], "diff": [], "mem_win": [], "cp_win": []}
        for length in tqdm(lengths, desc="Ablating"):
            mem, cp, diff, mem_win, cp_win = self.ablate_length(length, normalize_logit)
            result["mem"].append(mem)
            result["cp"].append(cp)
            result["diff"].append(diff)
            result["mem_win"].append(mem_win)
            result["cp_win"].append(cp_win)
        
        return {
            "mem": torch.cat(result["mem"]),
            "cp": torch.cat(result["cp"]),
            "diff": torch.cat(result["diff"]),
            "mem_win": torch.cat(result["mem_win"]),
            "cp_win": torch.cat(result["cp_win"])
        }

    def run(
        self,
        normalize_logit: Literal["none", "softmax", "sigmoid"] = "none",
        save_name: Optional[str] = None,
    ):
        """
        Run the attention ablation/modification experiment and return the results
        
        Returns:
        - mem: mean logit of the model for the factual token
        - cp: mean logit of the model for the counterfactual token
        - diff: mean difference between the logit of the model for the factual and counterfactual token
        - mem_std: standard deviation of the logit of the model for the factual token
        - cp_std: standard deviation of the logit of the model for the counterfactual token
        - diff_std: standard deviation of the difference between the logit of the model for the factual and counterfactual token
        - mem_win: count of the examples where the model predicted the factual token
        - cp_win: count of the examples where the model predicted the counterfactual token
        """
        result = self.ablate(normalize_logit)

        data = {
            "mem": result["mem"].mean().item(),
            "cp": result["cp"].mean().item(),
            "diff": result["diff"].mean().item(),
            "mem_std": result["mem"].std().item(),
            "cp_std": result["cp"].std().item(),
            "diff_std": result["diff"].std().item(),
            "mem_win": result["mem_win"].sum().item(),
            "cp_win": result["cp_win"].sum().item(),
        }
        
        return pd.DataFrame(data, index=[0])
            