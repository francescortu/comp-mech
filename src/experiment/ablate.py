from locale import normalize
from unittest import result

from cvxpy import length
from numpy import string_
from sklearn import base
from src.base_experiment import BaseExperiment, to_logit_token
from src.dataset import TlensDataset
from src.model import WrapHookedTransformer
from typing import  Tuple, Callable
import torch
from tqdm import tqdm
from functools import partial
from copy import deepcopy
from torch.utils.data import DataLoader
import einops

class Ablate(BaseExperiment):
    def __init__(self, dataset: TlensDataset, model: WrapHookedTransformer, batch_size, filter_outliers=False):
        super().__init__(dataset, model, batch_size, filter_outliers)

    @classmethod
    def from_experiment(cls, experiment: BaseExperiment):
        return cls(experiment.dataset, experiment.model,  experiment.batch_size, experiment.filter_outliers)
    
    def create_dataloader(self, filter_outliers:bool=False, **kwargs):
        if filter_outliers:
            print(self._filter_outliers)
            self._filter_outliers(**kwargs)
        else:
            self.slice_to_fit_batch()

    def slice_to_fit_batch(self) -> None:
        self.dataset.slice_to_fit_batch(self.batch_size)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        self.num_batches = len(self.dataloader)
        print("Number of examples after slicing:", len(self.dataloader) * self.batch_size)

    def get_normalize_metric(self) -> Callable[[torch.Tensor, torch.Tensor, str], Tuple[torch.Tensor, torch.Tensor]]:
        corrupted_logit, target = self.compute_logit()
        # clean_logit_mem, clean_logit_cp = to_logit_token(clean_logit, target)
        corrupted_logit_mem, corrupted_logit_cp, _, _ = to_logit_token(corrupted_logit, target)

        def normalize_logit_token(logit_mem:torch.Tensor, logit_cp:torch.Tensor, baseline:str="corrupted") -> Tuple[torch.Tensor, torch.Tensor]:
            # logit_mem, logit_cp = to_logit_token(logit, target)
            # percentage increase or decrease of logit_mem
            if baseline == "corrupted":
                logit_mem = 100 * (logit_mem - corrupted_logit_mem) / corrupted_logit_mem
                # percentage increase or decrease of logit_cp
                logit_cp = 100 * (logit_cp - corrupted_logit_cp) / corrupted_logit_cp
                return -logit_mem, -logit_cp
            elif baseline == "base_logit":
                return corrupted_logit_mem, corrupted_logit_cp  
            else :
                raise ValueError("baseline must be either 'corrupted' or 'base_logit'")

        return normalize_logit_token
    
    
    
    def ablate_heads(self, return_type:str="both") -> tuple[torch.Tensor, torch.Tensor]:
        assert return_type in ["diff", "both"], "return_type must be either 'diff' or 'both'"
        normalize_logit_token = self.get_normalize_metric()
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        self.num_batches = len(self.dataloader)

        def heads_hook(activation, hook, head, pos1=None, pos2=None):
            activation[:, head, -1, :] = 0
            return activation

        def freeze_hook(activation, hook, clean_activation, pos1=None, pos2=None):
            activation = clean_activation
            return activation


        examples_mem_list = [[[] for _ in range(self.model.cfg.n_heads)] for _ in range(self.model.cfg.n_layers)]
        examples_cp_list =  [[[] for _ in range(self.model.cfg.n_heads)] for _ in range(self.model.cfg.n_layers)]

        for idx, batch in tqdm(enumerate(self.dataloader), total=self.num_batches, desc="Ablating batches"):
            _, clean_cache = self.model.run_with_cache(batch["corrupted_prompts"])
            hooks = {}
            for layer in range(self.model.cfg.n_layers):
                # for head in range(self.model.cfg.n_heads):
                hooks[f"L{layer}"] = (f"blocks.{layer}.hook_attn_out",
                                      partial(
                                          freeze_hook,
                                          clean_activation=clean_cache[f"blocks.{layer}.hook_attn_out"],
                                      )
                                      )

            for layer in range(self.model.cfg.n_layers):
                for head in range(self.model.cfg.n_heads):
                    tmp_hooks = deepcopy(hooks)
                    tmp_hooks[f"L{layer}"] = (f"blocks.{layer}.attn.hook_pattern",
                                              partial(
                                                  heads_hook,
                                                  head=head,
                                              )
                                              )

                    list_hooks = list(tmp_hooks.values())
                    self.model.reset_hooks() 
                    logit = self.model.run_with_hooks( # type: ignore
                        batch["corrupted_prompts"],
                        fwd_hooks=list_hooks,
                    )[:, -1, :] # type: ignore
                    mem, cp, _, _ = to_logit_token(logit, batch["target"])

                    examples_mem_list[layer][head].append(mem.cpu())
                    examples_cp_list[layer][head].append(cp.cpu())
                # remove the hooks for the previous layer

                hooks.pop(f"L{layer}")
        
        for layer in range(self.model.cfg.n_layers):
            for head in range(self.model.cfg.n_heads):
                examples_mem_list[layer][head] = torch.cat(examples_mem_list[layer][head], dim=0) # type: ignore
                examples_cp_list[layer][head] = torch.cat(examples_cp_list[layer][head], dim=0) # type: ignore
        
        flattened_mem = [tensor for layer in examples_mem_list for head in layer for tensor in head]
        flattened_cp = [tensor for layer in examples_cp_list for head in layer for tensor in head]
        
        examples_mem = torch.stack(flattened_mem).view(self.model.cfg.n_layers, self.model.cfg.n_heads, -1)
        examples_cp = torch.stack(flattened_cp).view(self.model.cfg.n_layers, self.model.cfg.n_heads, -1)

        if return_type == "diff":
            # examples_mem = (einops.einsum(examples_mem, -examples_cp, "l h b, l h b -> l h b")).abs()
            examples_mem = (examples_mem -examples_cp).abs()
            examples_cp = examples_mem
            base_logit_mem, base_logit_cp = normalize_logit_token(examples_mem, examples_cp, "base_logit")
            base_diff = (base_logit_mem - base_logit_cp).abs()
            #percentage increade/decrease
            examples_cp =  (examples_cp - base_diff) 
            examples_mem = (examples_mem - base_diff) 
            
            return examples_mem, examples_cp
        
        base_logit_mem, base_logit_cp = normalize_logit_token(examples_mem, examples_cp, "corrupted")


        return base_logit_mem, base_logit_cp
    

    
    @staticmethod
    def count_mem_win(mem:torch.Tensor, cp:torch.Tensor) -> int:
        count = 0
        for i in range(len(mem)):
            if mem[i] > cp[i]:
                count += 1
        print("Number of examples where mem > cp:", count)
        return count
    

    def ablate_mlp_block(self) -> tuple[torch.Tensor, torch.Tensor]:
        normalize_logit_token = self.get_normalize_metric()
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        self.num_batches = len(self.dataloader)

        def mlp_hook(activation, hook,  pos1=None, pos2=None):
            activation[:,-1, :] = 0
            return activation

        def freeze_hook(activation, hook, clean_activation, pos1=None, pos2=None):
            activation = clean_activation
            return activation

        # examples_mem = torch.zeros((self.model.cfg.n_layers, self.num_batches, self.batch_size),
        #                            device="cpu")
        # examples_cp = torch.zeros((self.model.cfg.n_layers, self.num_batches, self.batch_size),
        #                           device="cpu")
        examples_mem_list = [[] for _ in range(self.model.cfg.n_layers)]
        examples_cp_list = [[] for _ in range(self.model.cfg.n_layers)]


        for idx, batch in tqdm(enumerate(self.dataloader), total=self.num_batches, desc="Ablating batches"):
            _, clean_cache = self.model.run_with_cache(batch["corrupted_prompts"])
            hooks = {}
            # for layer in range(self.model.cfg.n_layers):
            #     # for head in range(self.model.cfg.n_heads):
            #     hooks[f"L{layer}"] = (f"blocks.{layer}.hook_attn_out",
            #                           partial(
            #                               freeze_hook,
            #                               clean_activation=clean_cache[f"blocks.{layer}.hook_attn_out"],
            #                           )
            #                           )

            for layer in range(self.model.cfg.n_layers):
                # tmp_hooks = deepcopy(hooks)
                hook= [(f"blocks.{layer}.hook_mlp_out",
                                            partial(
                                                mlp_hook,
                                            )
                                            )]

                # list_hooks = list(tmp_hooks.values())
                self.model.reset_hooks()
                logit = self.model.run_with_hooks( # type: ignore
                    batch["corrupted_prompts"],
                    fwd_hooks=hook, # type: ignore
                )[:, -1, :]
                mem, cp, _, _ = to_logit_token(logit, batch["target"])
                # norm_mem, norm_cp = normalize_logit_token(mem, cp, baseline="corrupted")
                examples_mem_list[layer].append(mem.cpu())
                examples_cp_list[layer].append(cp.cpu())
            # remove the hooks for the previous layer

                # hooks.pop(f"L{layer}")

        for layer in range(self.model.cfg.n_layers):
            examples_mem_list[layer] = torch.cat(examples_mem_list[layer], dim=0) # type: ignore
            examples_cp_list[layer] = torch.cat(examples_cp_list[layer], dim=0) # type: ignore
            
        examples_mem = torch.stack(examples_mem_list).view(self.model.cfg.n_layers, -1) # type: ignore
        examples_cp = torch.stack(examples_cp_list).view(self.model.cfg.n_layers, -1) # type: ignore
        
        base_logit_mem, base_logit_cp = normalize_logit_token(examples_mem, examples_cp, "corrupted")
        
        
        # for layer in range(self.model.cfg.n_layers):
        #     for head in range(self.model.cfg.n_heads):
        #         norm_mem, norm_cp, = normalize_logit_token(examples_mem[layer, head, :], examples_cp[layer, head, :],
        #                                                   baseline="raw")
        #         examples_mem[layer, head, :] = norm_mem
        #         examples_cp[layer, head, :] = norm_cp

        return base_logit_mem, base_logit_cp
    
    
    def ablate_attn_block(self) -> tuple[torch.Tensor, torch.Tensor]:
        normalize_logit_token = self.get_normalize_metric()
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        self.num_batches = len(self.dataloader)

        def attn_hook(activation, hook,  pos1=None, pos2=None):
            activation[:,-1, :] = 0
            return activation

        def freeze_hook(activation, hook, clean_activation, pos1=None, pos2=None):
            activation = clean_activation
            return activation

        # examples_mem = torch.zeros((self.model.cfg.n_layers, self.num_batches, self.batch_size),
        #                            device="cpu")
        # examples_cp = torch.zeros((self.model.cfg.n_layers, self.num_batches, self.batch_size),
        #                           device="cpu")
        examples_mem_list = [[] for _ in range(self.model.cfg.n_layers)]
        examples_cp_list = [[] for _ in range(self.model.cfg.n_layers)]


        for idx, batch in tqdm(enumerate(self.dataloader), total=self.num_batches, desc="Ablating batches"):
            _, clean_cache = self.model.run_with_cache(batch["corrupted_prompts"])
            hooks = {}
            for layer in range(self.model.cfg.n_layers):
                # for head in range(self.model.cfg.n_heads):
                hooks[f"L{layer}"] = (f"blocks.{layer}.hook_attn_out",
                                      partial(
                                          freeze_hook,
                                          clean_activation=clean_cache[f"blocks.{layer}.hook_attn_out"],
                                      )
                                      )

            for layer in range(self.model.cfg.n_layers):
                tmp_hooks = deepcopy(hooks)
                tmp_hooks[f"L{layer}"] = (f"blocks.{layer}.hook_attn_out",
                                            partial(
                                                attn_hook,
                                            )
                                            )

                list_hooks = list(tmp_hooks.values())
                self.model.reset_hooks()
                logit = self.model.run_with_hooks( # type: ignore
                    batch["corrupted_prompts"],
                    fwd_hooks=list_hooks,
                )[:, -1, :]
                mem, cp, _, _ = to_logit_token(logit, batch["target"])
                # norm_mem, norm_cp = normalize_logit_token(mem, cp, baseline="corrupted")
                examples_mem_list[layer].append(mem.cpu())
                examples_cp_list[layer].append(cp.cpu())
            # remove the hooks for the previous layer

                hooks.pop(f"L{layer}")

        for layer in range(self.model.cfg.n_layers):
            examples_mem_list[layer] = torch.cat(examples_mem_list[layer], dim=0) # type: ignore
            examples_cp_list[layer] = torch.cat(examples_cp_list[layer], dim=0) # type: ignore
            
        examples_mem = torch.stack(examples_mem_list).view(self.model.cfg.n_layers, -1) # type: ignore
        examples_cp = torch.stack(examples_cp_list).view(self.model.cfg.n_layers, -1) # type: ignore
        
        base_logit_mem, base_logit_cp = normalize_logit_token(examples_mem, examples_cp, "corrupted")
        
        
        # for layer in range(self.model.cfg.n_layers):
        #     for head in range(self.model.cfg.n_heads):
        #         norm_mem, norm_cp, = normalize_logit_token(examples_mem[layer, head, :], examples_cp[layer, head, :],
        #                                                   baseline="raw")
        #         examples_mem[layer, head, :] = norm_mem
        #         examples_cp[layer, head, :] = norm_cp

        return base_logit_mem, base_logit_cp


class AblateMultiLen:
    def __init__(self, dataset: TlensDataset, model: WrapHookedTransformer, batch_size:int):
        self.model = model
        self.batch_size = batch_size
        self.ablate = Ablate(dataset, model, batch_size)
    
    @classmethod
    def from_experiment(cls, experiment: BaseExperiment):
        return cls(experiment.dataset, experiment.model,  experiment.batch_size)
    
    @classmethod
    def from_ablate(cls, ablate: Ablate):
        return cls(ablate.dataset, ablate.model, ablate.batch_size)


    def ablate_single_len(self, length:int, filter_outliers:bool=False, ablate_target:str="head", **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        assert ablate_target in ["head", "attn", "mlp_block"], "ablate_target must be either 'head', 'attn' or 'mlp'"
        self.ablate.set_len(length, slice_to_fit_batch=False)
        # n_samples = len(self.ablate.dataset)
        # if n_samples  < self.batch_size:
        #     return None, None
        # self.ablate.create_dataloader(filter_outliers=filter_outliers, **kwargs)
        if ablate_target == "head":
            return self.ablate.ablate_heads()
        if ablate_target == "attn":
            return self.ablate.ablate_attn_block()
        if ablate_target == "mlp_block":
            return self.ablate.ablate_mlp_block()
        else:
            raise ValueError("ablate_target must be either 'head' or 'attn'")

    def ablate_multi_len(self, filter_outliers:bool=False, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        lenghts = self.ablate.dataset.get_lengths()
        # lenghts = [11]
        result_cp_per_len = {}
        result_mem_per_len = {}
        for l in lenghts:
            print("Ablating examples of length", l, "...")
            mem, cp = self.ablate_single_len(l, filter_outliers=filter_outliers, **kwargs)
            if mem is not None and cp is not None:
                result_mem_per_len[l], result_cp_per_len[l] = mem, cp
        # concatenate the results
        result_cp = torch.cat(list(result_cp_per_len.values()), dim=-1)
        result_mem = torch.cat(list(result_mem_per_len.values()), dim=-1)

        print("result_cp.shape", result_cp.shape)
        
        
        return result_mem, result_cp
        
        
        # return result_mem, result_cp, result_mem_base, result_cp_base
        
        
        
class AblateMLP(BaseExperiment):
    def __init__(self, dataset: TlensDataset, model: WrapHookedTransformer, batch_size:int):
        super().__init__(dataset, model, batch_size)
        
    def get_normalize_metric(self) -> Callable[[torch.Tensor, torch.Tensor, str], Tuple[torch.Tensor, torch.Tensor]]:
        corrupted_logit, target = self.compute_logit()
        # clean_logit_mem, clean_logit_cp = to_logit_token(clean_logit, target)
        corrupted_logit_mem, corrupted_logit_cp, _, _ = to_logit_token(corrupted_logit, target)

        def normalize_logit_token(logit_mem:torch.Tensor, logit_cp:torch.Tensor, baseline:str="corrupted") -> Tuple[torch.Tensor, torch.Tensor]:
            # logit_mem, logit_cp = to_logit_token(logit, target)
            # percentage increase or decrease of logit_mem
            if baseline == "corrupted":
                logit_mem = 100 * (logit_mem - corrupted_logit_mem) / corrupted_logit_mem
                # percentage increase or decrease of logit_cp
                logit_cp = 100 * (logit_cp - corrupted_logit_cp) / corrupted_logit_cp
                return -logit_mem, -logit_cp
            elif baseline == "base_logit":
                return corrupted_logit_mem, corrupted_logit_cp  
            else :
                raise ValueError("baseline must be either 'corrupted' or 'base_logit'")

        return normalize_logit_token
    
    def ablate_single_len(self, length:int, target="mlp", interval=0):
        if target == "mlp":
            hook_string = "hook_mlp_out"
        elif target == "attn":
            hook_string = "hook_attn_out"
        else:
            raise ValueError("target must be either 'mlp' or 'attn'")
            
            
        self.set_len(length, interval=interval)
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        num_batches = len(dataloader)
        
        mem_logit_list = [[[] for _ in range(length)] for _ in range(self.model.cfg.n_layers)]
        cp_logit_list = [[[] for _ in range(length)] for _ in range(self.model.cfg.n_layers)]
        
        
        def mlp_hook(activation, hook, pos):
            activation[:, pos,:] = 0
            return activation
        
        def freeze_hook(activation, hook, clean_activation):
            activation = clean_activation
            return activation
        
        if num_batches == 0:
            return None, None, None, None
        # normalize_logit_token = self.get_normalize_metric()
        
        for idx, batch in tqdm(enumerate(dataloader), total=num_batches, desc="Ablating batches"):
            _, clean_cache = self.model.run_with_cache(batch["corrupted_prompts"])
            hooks = {}
            if target == "mlp":
                for layer in range(self.model.cfg.n_layers):
                    # for head in range(self.model.cfg.n_heads):
                    hooks[f"L{layer}"] = (f"blocks.{layer}.attn.hook_pattern",
                                        partial(
                                            freeze_hook,
                                            clean_activation=clean_cache[f"blocks.{layer}.attn.hook_pattern"],
                                        )
                                        )
            for layer in range(self.model.cfg.n_layers):
                for pos in range(length):
                    tmp_hooks = deepcopy(hooks)
                    tmp_hooks[f"L{layer}"] = (f"blocks.{layer}.{hook_string}",
                                                partial(
                                                    mlp_hook,
                                                    pos=pos,
                                                )
                                                )

                    list_hooks = list(tmp_hooks.values())
                    self.model.reset_hooks()
                    logit = self.model.run_with_hooks( # type: ignore
                        batch["corrupted_prompts"],
                        fwd_hooks=list_hooks,
                    )[:, -1, :]
                    mem, cp, _, _ = to_logit_token(logit, batch["target"], normalize="softmax")
                    mem_logit_list[layer][pos].append(mem.cpu())
                    cp_logit_list[layer][pos].append(cp.cpu())
                
                if target == "mlp":
                   hooks.pop(f"L{layer}")
                    
                    
        for layer in range(self.model.cfg.n_layers):
            for pos in range(length):
                mem_logit_list[layer][pos] = torch.cat(mem_logit_list[layer][pos], dim=0) # type: ignore
                cp_logit_list[layer][pos] = torch.cat(cp_logit_list[layer][pos], dim=0) # type: ignore
        
        
        flattened_mem_logit = [tensor for layer in mem_logit_list for pos in layer for tensor in pos]
        flattened_cp_logit = [tensor for layer in cp_logit_list for pos in layer for tensor in pos]
        
        mem_logit = torch.stack(flattened_mem_logit).view(self.model.cfg.n_layers, length, -1)
        cp_logit = torch.stack(flattened_cp_logit).view(self.model.cfg.n_layers, length, -1)
        # base_mem_logit, base_cp_logit = normalize_logit_token(mem_logit, cp_logit, "corrupted")
        
        base_mem_logit = mem_logit
        base_cp_logit = cp_logit
        
        base_mem_logit = einops.rearrange(base_mem_logit, "l p b -> b l p")
        base_cp_logit = einops.rearrange(base_cp_logit, "l p b -> b l p")
        
        object_pos = self.dataset.obj_pos[0]
        base_mem_aggr = self.aggregate_result(object_pos, base_mem_logit, length)
        base_cp_aggr = self.aggregate_result(object_pos, base_cp_logit, length)
        base_logit, target = self.compute_logit()
        base_logit_mem, base_logit_cp, _, _ = to_logit_token(base_logit, target)
        
        return base_mem_aggr, base_cp_aggr, base_logit_mem, base_logit_cp
    
    def ablate_multi_len(self, target, interval) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        lenghts = self.dataset.get_lengths()
        result_cp_per_len = {}
        result_mem_per_len = {}
        base_logit_mem = {}
        base_logit_cp = {}
        for l in lenghts:
            print("Ablating examples of length", l, "...")
            mem, cp, b_mem,b_cp = self.ablate_single_len(l,target, interval=interval)
            if mem is not None and cp is not None:
                result_mem_per_len[l], result_cp_per_len[l] = mem, cp
                base_logit_mem[l], base_logit_cp[l] = b_mem, b_cp
        # concatenate the results
        result_cp = torch.cat(list(result_cp_per_len.values()), dim=0)
        result_mem = torch.cat(list(result_mem_per_len.values()), dim=0)
        base_logit_mem = torch.cat(list(base_logit_mem.values()), dim=0)
        base_logit_cp = torch.cat(list(base_logit_cp.values()), dim=0)
        
        return result_mem, result_cp, base_logit_mem, base_logit_cp
                
                
                
                