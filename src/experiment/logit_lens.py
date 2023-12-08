import torch
from torch.utils.data import DataLoader
import einops
from tqdm import tqdm
from src.dataset import TlensDataset
from src.model import WrapHookedTransformer
from src.base_experiment import BaseExperiment, to_logit_token
from src.logit_storage import LogitStorage, IndexLogitStorage, HeadLogitStorage
from transformer_lens import ActivationCache
from typing import Optional, List, Tuple, Union, Dict, Any


class LogitLens(BaseExperiment):
    def __init__(self, dataset: TlensDataset, model: WrapHookedTransformer, batch_size:int):
        super().__init__(dataset, model, batch_size)
        self.valid_blocks = ["mlp_out", "resid_pre", "resid_post", "attn_out"]
        self.valid_heads = ["head"]
        
    def project_per_position(self, component_cached:torch.Tensor, length:int):
        # assert that the activation name is a f-string with a single placeholder for the layer
        assert component_cached.shape[1] == length, f"component_cached.shape[1] = {component_cached.shape[1]}, self.model.cfg.n_heads = {self.model.cfg.n_heads}"
        assert component_cached.shape[2] == self.model.cfg.d_model, f"component_cached.shape[2] = {component_cached.shape[2]}, self.model.cfg.d_model = {self.model.cfg.d_model}"
        
        for position in range(length):
            logit = einops.einsum(self.model.W_U, component_cached[:, position, :], "d d_v, b d -> b d_v")
            logit = self.model.ln_final(logit)
            yield logit
        
    def project_length(self, length:int, component:str, return_index:bool=False, normalize:str="none"):


        self.set_len(length, slice_to_fit_batch=False)
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        num_batches = len(dataloader)
        if num_batches == 0:
            return None
        
        if return_index:
            storer = IndexLogitStorage(self.model.cfg.n_layers,  length)
            if component in self.valid_heads:
                storer = HeadLogitStorage.from_index_logit_storage(storer, self.model.cfg.n_heads)
        else:
            storer = LogitStorage(self.model.cfg.n_layers,  length)
            if component in self.valid_heads:
                storer = HeadLogitStorage.from_logit_storage(storer, self.model.cfg.n_heads)
        
        for batch in tqdm(dataloader, total=num_batches):
            _, cache = self.model.run_with_cache(batch["corrupted_prompts"])
            for layer in range(self.model.cfg.n_layers):
                if component in self.valid_blocks:
                    cached_component = cache[component, layer]
                    for position, logit in enumerate(self.project_per_position(cached_component, length)):
                        logit_token = to_logit_token(logit, batch["target"], normalize=normalize, return_index=return_index)
                        storer.store(layer=layer, position=position, logit=logit_token)
                elif component in self.valid_heads:
                    cached_component = cache[f"blocks.{layer}.attn.hook_z"]
                    for head in range(self.model.cfg.n_heads):
                        output_head = einops.einsum(cached_component[:,:,head,:],self.model.blocks[layer].attn.W_O[head,:,:], "batch pos d_head, d_head d_model -> batch pos d_model") # type: ignore
                        for position, logit in enumerate(self.project_per_position(output_head, length)):
                            logit_token = to_logit_token(logit, batch["target"], normalize=normalize, return_index=return_index)
                            storer.store(layer=layer, position=position, head=head, logit=logit_token)
                else:
                    raise ValueError(f"component must be one of {self.valid_blocks + self.valid_heads}")
        object_positions = self.dataset.obj_pos[0]
        
        return storer.get_aggregate_logit(object_position=object_positions)
        
        
    def project(self, component:str, return_index:bool=False, normalize:str="none"):
        lengths = self.dataset.get_lengths()
        result = {}
        for l in lengths:
            result[l] = self.project_length(l, component, return_index=return_index, normalize=normalize)
            
        # select a random key to get the shape of the result
        tuple_shape = len(result[lengths[0]])
        #result is a dict {lenght: }
        aggregated_result = [torch.cat([result[l][idx_tuple] for l in lengths], dim=-1) for idx_tuple in range(tuple_shape)]
       

        return tuple(aggregated_result)
    
    def compute_mean_over_layers(self, result:Tuple[torch.Tensor, ...]):
        # for each layer, compute the mean over the lengths
        mean_result = tuple([result[idx_tuple].mean(dim=-1) for idx_tuple in range(len(result))])
        
        # compute the percentage increase for each position/head over the mean for the same layer across all positions/heads
        raise NotImplementedError("TODO")
    
    def run(self, component:str, return_index:bool=False, normalize:str="none"):
        result = self.project(component, return_index=return_index, normalize=normalize)
        
        import pandas as pd
        data = []
        for layer in range(self.model.cfg.n_layers):
            for position in range(result[0][layer].shape[0]):
                if component in self.valid_heads:
                    for head in range(self.model.cfg.n_heads):
                        data.append(
                            {
                                "component": f"H{head}",
                                "layer": layer,
                                "position": position,
                                "mem": result[0][layer][head][position].mean().item(),
                                "cp": result[1][layer][head][position].std().item(),
                                "mem_idx": None if not return_index else result[2][layer][head][position].argmax().item(),
                                "cp_idx": None if not return_index else result[3][layer][head][position].argmin().item(),
                            }
                        )
                else:
                    data.append(
                        {
                            "component": f"{component}",
                            "layer": layer,
                            "position": position,
                            "mem": result[0][layer][position].mean().item(),
                            "cp": result[1][layer][position].std().item(),
                            "mem_idx": None if not return_index else result[2][layer][position].argmax().item(),
                            "cp_idx": None if not return_index else result[3][layer][position].argmin().item(),
                        }
                    )
        
        return pd.DataFrame(data)
    
    
        
        