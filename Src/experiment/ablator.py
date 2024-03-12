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

    def set_hooks(self, hooks) -> None:
        self.hooks = hooks

    def __run_with_hooks__(self, batch):
        self.model.reset_hooks()
        with torch.no_grad():
            logit = self.model.run_with_hooks(
                batch["prompt"], prepend_bos=False, fwd_hooks=self.hooks
            )

        return logit[:, -1, :]
    
    def ablate_length(
        self, length: int, normalize_logit: Literal["none", "softmax", "sigmoid"] = "none"
    ):
        mem = []
        cp = []
        diff = []
        for batch in tqdm(
            DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=False,
            ),
            desc="Ablating",
        ):
            logit = self.__run_with_hooks__(batch)
            logit_mem, logit_cp, _, _ = to_logit_token(
                logit, batch["target"], normalize=normalize_logit
            )
            mem.append(logit_mem)
            cp.append(logit_cp)
            diff.append(logit_mem - logit_cp)
            
        return torch.cat(mem), torch.cat(cp), torch.cat(diff)
        
    
    def ablate(self, normalize_logit: Literal["none", "softmax", "sigmoid"] = "none"):
        lengths = self.dataset.get_lengths()
        result = {"mem": [], "cp": [], "diff": []}
        for length in tqdm(lengths, desc="Ablating"):
            mem, cp, diff = self.ablate_length(length, normalize_logit)
            result["mem"].append(mem)
            result["cp"].append(cp)
            result["diff"].append(diff)
        
        return {
            "mem": torch.cat(result["mem"]),
            "cp": torch.cat(result["cp"]),
            "diff": torch.cat(result["diff"])
        }

    def run(
        self,
        normalize_logit: Literal["none", "softmax", "sigmoid"] = "none",
        save_name: Optional[str] = None,
    ):
        result = self.ablate(normalize_logit)

        data = {
            "hooks": [str(h[0]) for h in self.hooks],
            "mem": result["mem"].mean().item(),
            "cp": result["cp"].mean().item(),
            "diff": result["diff"].mean().item(),
            "mem_std": result["mem"].std().item(),
            "cp_std": result["cp"].std().item(),
            "diff_std": result["diff"].std().item()
        }
        
        return pd.DataFrame(data)
            