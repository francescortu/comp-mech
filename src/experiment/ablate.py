from src.base_experiment import BaseExperiment, to_logit_token
from src.dataset import TlensDataset
from src.model import WrapHookedTransformer
from typing import  Tuple, Callable
import torch
from tqdm import tqdm
from functools import partial
from copy import deepcopy
from torch.utils.data import DataLoader

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
        corrupted_logit_mem, corrupted_logit_cp = to_logit_token(corrupted_logit, target)

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
                    mem, cp = to_logit_token(logit, batch["target"])

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
                if logit.shape[0] != self.batch_size:
                    print("Ops, the batch size is not correct")
                mem, cp = to_logit_token(logit, batch["target"])
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
        assert ablate_target in ["head", "attn"], "ablate_target must be either 'head' or 'attn'"
        self.ablate.set_len(length, slice_to_fit_batch=False)
        # n_samples = len(self.ablate.dataset)
        # if n_samples  < self.batch_size:
        #     return None, None
        # self.ablate.create_dataloader(filter_outliers=filter_outliers, **kwargs)
        if ablate_target == "head":
            return self.ablate.ablate_heads()
        if ablate_target == "attn":
            return self.ablate.ablate_attn_block()
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