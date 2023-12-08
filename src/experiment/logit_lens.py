import torch
from torch.utils.data import DataLoader
import einops
from tqdm import tqdm
from src.dataset import TlensDataset
from src.model import WrapHookedTransformer
from src.base_experiment import BaseExperiment, to_logit_token, LogitStorage, IndexLogitStorage, HeadLogitStorage
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
        
    def project_length(self, length:int, component:str, return_index:bool=False, normalize:str="softmax"):


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
                        output_head = einops.einsum(cached_component[:,:,head,:],self.model.blocks[layer].attn.W_O[head,:,:], "batch pos d_head, d_head d_model -> batch pos d_model")
                        for position, logit in enumerate(self.project_per_position(output_head, length)):
                            logit_token = to_logit_token(logit, batch["target"], normalize=normalize, return_index=return_index)
                            storer.store(layer=layer, position=position, head=head, logit=logit_token)
                else:
                    raise ValueError(f"component must be one of {self.valid_blocks + self.valid_heads}")
        object_positions = self.dataset.obj_pos[0]
        
        return storer.get_aggregate_logit(object_position=object_positions)
        
        
    def project(self, component:str, return_index:bool=False, normalize:str="softmax"):
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
    
    def project_and_return_df(self, component:str, return_index:bool=False, normalize:str="softmax"):
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
    
    
        
                
                
# class LastComponentProject(BaseExperiment):
#     def __init__(self, dataset: TlensDataset, model: WrapHookedTransformer, batch_size:int):
#         super().__init__(dataset, model, batch_size)
        
#     def project_heads_single_len(self, length:int, normalize:str, return_index=False) -> dict:
#         self.set_len(length, slice_to_fit_batch=False)
#         dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
#         num_batches = len(dataloader)
        
#         mem_logit = [[[] for _ in range(self.model.cfg.n_heads)] for _ in range(self.model.cfg.n_layers)]
#         cp_logit = [[[] for _ in range(self.model.cfg.n_heads)] for _ in range(self.model.cfg.n_layers)]
#         mem_logit_idx = [[[] for _ in range(self.model.cfg.n_heads)] for _ in range(self.model.cfg.n_layers)]
#         cp_logit_idx = [[[] for _ in range(self.model.cfg.n_heads)] for _ in range(self.model.cfg.n_layers)]
        
#         if num_batches == 0:
#             return {"mem": None, "cp": None, "mem_idx": None, "cp_idx": None}
#         for idx, batch in tqdm(enumerate(dataloader), total=num_batches):
#             _, cache = self.model.run_with_cache(batch["corrupted_prompts"])
#             for layer in range(self.model.cfg.n_layers):
#                 for head in range(self.model.cfg.n_heads):
#                     # print("layer", layer, "head", head)
#                     output_head = cache[f"blocks.{layer}.attn.hook_z"]
#                     output_head = einops.einsum(cache[f"blocks.{layer}.attn.hook_z"][:,:,head,:],self.model.blocks[layer].attn.W_O[head,:,:], "batch pos d_head, d_head d_model -> batch pos d_model") # type: ignore
#                     logit_output = einops.einsum(self.model.W_U, output_head[:, -1, :], "d d_v, b d -> b d_v")
#                     logit_output = self.model.ln_final(logit_output)

#                     tmp_mem, tmp_cp, tmp_mem_idx, tmp_cp_idx = to_logit_token(logit_output, batch["target"], normalize=normalize, return_index=return_index)
                
#                     mem_logit[layer][head].append(tmp_mem.cpu())
#                     cp_logit[layer][head].append(tmp_cp.cpu())
#                     if tmp_mem_idx is not None:
#                         mem_logit_idx[layer][head].append(tmp_mem_idx.cpu())
#                     if tmp_cp_idx is not None:
#                         cp_logit_idx[layer][head].append(tmp_cp_idx.cpu())
                    
#         for layer in range(self.model.cfg.n_layers):
#             for head in range(self.model.cfg.n_heads):
#                 mem_logit[layer][head] = torch.cat(mem_logit[layer][head], dim=0) # type: ignore
#                 cp_logit[layer][head] = torch.cat(cp_logit[layer][head], dim=0) # type: ignore
#                 if return_index:
#                     mem_logit_idx[layer][head] = torch.cat(mem_logit_idx[layer][head], dim=0) # type: ignore
#                     cp_logit_idx[layer][head] = torch.cat(cp_logit_idx[layer][head], dim=0) # type: ignore
        
#         #flat
#         flat_mem_logit = [tensor for layer in mem_logit for head in layer for tensor in head]
#         flat_cp_logit = [tensor for layer in cp_logit for head in layer for tensor in head]
#         flat_mem_logit_idx = []  # Initialize flat_mem_logit_idx as an empty list
#         flat_cp_logit_idx = []  # Initialize flat_cp_logit_idx as an empty list
#         if return_index:
#             flat_mem_logit_idx = [tensor for layer in mem_logit_idx for head in layer for tensor in head]
#             flat_cp_logit_idx = [tensor for layer in cp_logit_idx for head in layer for tensor in head]

#         mem_logit = torch.stack(flat_mem_logit).view(self.model.cfg.n_layers, self.model.cfg.n_heads, -1)
#         cp_logit = torch.stack(flat_cp_logit).view(self.model.cfg.n_layers, self.model.cfg.n_heads, -1)
#         if return_index:
#             mem_logit_idx = torch.stack(flat_mem_logit_idx).view(self.model.cfg.n_layers, self.model.cfg.n_heads, -1)
#             cp_logit_idx = torch.stack(flat_cp_logit_idx).view(self.model.cfg.n_layers, self.model.cfg.n_heads, -1)

#             return {"mem": mem_logit, "cp": cp_logit, "mem_idx": mem_logit_idx, "cp_idx": cp_logit_idx}
        
#         return {"mem": mem_logit, "cp": cp_logit}
        
#     def project_heads(self, return_index=False, normalize:str="softmax"):
#         lengths = self.dataset.get_lengths()
#         result_mem = {}
#         result_cp = {}
#         result_mem_idx = {}
#         result_cp_idx = {}
#         for l in lengths:
#             # mem, cp, mem_idx, cp_idx = self.project_heads_single_len(l)
#             return_dict = self.project_heads_single_len(l, return_index=return_index, normalize=normalize)
#             if return_dict["mem"] is not None:
#                 result_mem[l] = return_dict["mem"]
#                 result_cp[l] = return_dict["cp"]
#                 if return_index:
#                     result_mem_idx[l] = return_dict["mem_idx"]
#                     result_cp_idx[l] = return_dict["cp_idx"]

#         # result_score = torch.stack(list(result.values()), dim=0).mean(dim=0)
#         result_mem = torch.cat(list(result_mem.values()), dim=-1)
#         result_cp = torch.cat(list(result_cp.values()), dim=-1)
#         if return_index:
#             result_mem_idx = torch.cat(list(result_mem_idx.values()), dim=-1)
#             result_cp_idx = torch.cat(list(result_cp_idx.values()), dim=-1)
#             return {"mem": result_mem, "cp": result_cp, "mem_idx": result_mem_idx, "cp_idx": result_cp_idx}

    
#         return {"mem": result_mem, "cp": result_cp}
        
#     def project_block(self, length, target, return_index=False, normalize:str = "softmax") -> dict:
#         if target == "mlp":
#             target_string = "mlp_out"
#         elif target == "attn":
#             target_string = "attn_out"
#         elif target == "resid":
#             target_string = "resid_post"
#         else:
#             raise ValueError("target must be either 'mlp' or 'attn'")
#         self.set_len(length, slice_to_fit_batch=False)
#         dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
#         num_batches = len(dataloader)

#         # logit_target = torch.zeros((self.model.cfg.n_layers, length, num_batches, self.batch_size), device="cpu")
#         mem_logit = [[] for _ in range(self.model.cfg.n_layers)]
#         cp_logit = [[] for _ in range(self.model.cfg.n_layers)]
#         mem_logit_idx = [[] for _ in range(self.model.cfg.n_layers)]
#         cp_logit_idx = [[] for _ in range(self.model.cfg.n_layers)]
        
#         if num_batches == 0:
#             return {"mem": None, "cp": None, "mem_idx": None, "cp_idx": None}
        
#         for idx, batch in tqdm(enumerate(dataloader), total=num_batches):
#             _, cache = self.model.run_with_cache(batch["corrupted_prompts"])
#             for layer in range(self.model.cfg.n_layers):
#                 residual_stream = cache[target_string, layer]
#                 logit_output = einops.einsum(self.model.W_U, residual_stream[:, -1, :], "d d_v, b d -> b d_v")
#                 logit_output = self.model.ln_final(logit_output)

#                 tmp_mem, tmp_cp, tmp_mem_idx, tmp_cp_idx = to_logit_token(logit_output, batch["target"], normalize=normalize, return_index=return_index)
#                 mem_logit[layer].append(tmp_mem.cpu())
#                 cp_logit[layer].append(tmp_cp.cpu())
#                 if return_index:
#                     if tmp_mem_idx is not None:
#                         mem_logit_idx[layer].append(tmp_mem_idx.cpu())
#                     if tmp_cp_idx is not None:
#                         cp_logit_idx[layer].append(tmp_cp_idx.cpu())


#         new_mem_logit = []
#         new_cp_logit = []
#         new_mem_logit_idx = []
#         new_cp_logit_idx = []

#         for layer in range(self.model.cfg.n_layers):
#             new_mem_logit.append(torch.cat(mem_logit[layer], dim=0))
#             new_cp_logit.append(torch.cat(cp_logit[layer], dim=0))
#             if return_index:
#                 new_mem_logit_idx.append(torch.cat(mem_logit_idx[layer], dim=0))
#                 new_cp_logit_idx.append(torch.cat(cp_logit_idx[layer], dim=0))

#         mem_logit = torch.stack(new_mem_logit).view(self.model.cfg.n_layers, -1)
#         cp_logit = torch.stack(new_cp_logit).view(self.model.cfg.n_layers, -1)
#         if return_index:
#             mem_logit_idx = torch.stack(new_mem_logit_idx).view(self.model.cfg.n_layers, -1)
#             cp_logit_idx = torch.stack(new_cp_logit_idx).view(self.model.cfg.n_layers, -1)
#             return {"mem": mem_logit, "cp": cp_logit, "mem_idx": mem_logit_idx, "cp_idx": cp_logit_idx}
#         return {"mem": mem_logit, "cp": cp_logit}

#     def project_blocks(self, target, return_index=False, normalize:str = "softmax"):
#         lenghts = self.dataset.get_lengths()
#         result_mem = {}
#         result_cp = {}
#         result_mem_idx = {}
#         result_cp_idx = {}
#         for l in lenghts:
#             return_dict = self.project_block(l, target, return_index=return_index, normalize=normalize)
            
#             if return_dict["mem"] is not None:
#                 result_mem[l] = return_dict["mem"]
#                 result_cp[l] = return_dict["cp"]
#                 if return_index:
#                     result_mem_idx[l] = return_dict["mem_idx"]
#                     result_cp_idx[l] = return_dict["cp_idx"]

#         # result_score = torch.stack(list(result.values()), dim=0).mean(dim=0)
#         result_mem = torch.cat(list(result_mem.values()), dim=-1)
#         result_cp = torch.cat(list(result_cp.values()), dim=-1)
#         if return_index:
#             result_mem_idx = torch.cat(list(result_mem_idx.values()), dim=-1)
#             result_cp_idx = torch.cat(list(result_cp_idx.values()), dim=-1)
#             return {"mem": result_mem, "cp": result_cp, "mem_idx": result_mem_idx, "cp_idx": result_cp_idx}
        
#         return {"mem": result_mem, "cp": result_cp}


#     def project(self, component:str, return_index:bool=False, normalize:str="softmax"):
#         valid_blocks = ["mlp", "resid_pre", "resid_post", "attn"]
#         valid_heads = ["head"]

#         if component in valid_blocks:
#             return self.project_blocks(target=component, return_index=return_index, normalize=normalize)
#         elif component in valid_heads:
#             return self.project_heads(return_index=return_index, normalize=normalize)
#         else:
#             raise ValueError(f"component must be one of {valid_blocks + valid_heads}")
        
        
        
