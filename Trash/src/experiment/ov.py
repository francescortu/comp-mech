import torch
from torch.utils.data import DataLoader
import einops
from src.dataset import TlensDataset
from src.model import WrapHookedTransformer
from src.base_experiment import BaseExperiment, to_logit_token
from typing import  Literal
import pandas as pd
from tqdm import tqdm



class OV_storage:
    def __init__(self, n_layers:int, n_heads:int):

        # create a dict of key (layer, head) and value a list
        self.mem_input = {f"L{i}H{j}": [] for i in range(n_layers) for j in range(n_heads)}
        self.cp_input = {f"L{i}H{j}": [] for i in range(n_layers) for j in range(n_heads)}
        
    def append(self, logit_diff_mem_input:torch.Tensor, logit_diff_cp_input:torch.Tensor, layer:int, head:int):
        self.mem_input[f"L{layer}H{head}"].append(logit_diff_mem_input.cpu())
        self.cp_input[f"L{layer}H{head}"].append(logit_diff_cp_input.cpu())
    
    def stack(self):
        mem_input_ = {k: torch.cat(v, dim=0) for k, v in self.mem_input.items()}
        cp_input_ = {k: torch.cat(v, dim=0) for k, v in self.cp_input.items()}
        return mem_input_, cp_input_

class OV(BaseExperiment):
    def __init__(self, dataset: TlensDataset, model: WrapHookedTransformer, batch_size:int, experiment: Literal["copyVSfact", "contextVSfact"] = "copyVSfact",):
        super().__init__(dataset, model, batch_size, experiment)
    
    def _ov_matrix(self, layer:int, head:int):
        return (self.model.W_U.T @ (self.model.OV[layer,head,:,:] @ self.model.W_E.T)) #! Positional encoding is not included in the OV matrix
    
    def _ov_matrix_with_positional_encoding(self, layer:int, head:int):
        return (self.model.W_U.T @ self.model.OV[layer,head,:,:] @ self.model.W_E_pos.T)

    def compute_logit_diff_(self, target:torch.Tensor, layer:int, head:int, normalize_logit:Literal["none", "softmax", "log_softmax"] = "none"):
        token_embedding = self.model.W_E[target] # (batch_size, 2, d_model)
        token_embedding = self.model.blocks[0].mlp(token_embedding) # type: ignore (batch_size, 2, d_model)
        logits = einops.einsum((self.model.W_U.T @ self.model.OV[layer, head, :,:]).AB, token_embedding, "d_vocab d_model, batch n_target d_model -> batch n_target d_vocab") # type: ignore
        
        logit_mem_mem_input, logit_cp_mem_input, _, _ = to_logit_token(logits[:,0,:], target, normalize=normalize_logit)
        logit_mem_cp_input, logit_cp_cp_input, _, _ = to_logit_token(logits[:,1,:], target, normalize=normalize_logit)
        
        logit_diff_mem_input = logit_mem_mem_input - logit_cp_mem_input
        logit_diff_cp_input = logit_mem_cp_input - logit_cp_cp_input
        
        return logit_diff_mem_input, logit_diff_cp_input
        

    def compute_logit_diff(self, target:torch.Tensor, layer:int, head:int):
        """
        Args:
            target (torch.Tensor): The target of shape (batch_size, 2) where the first column is the memory target and the second column is the copy target
            layer (int): The layer of the OV matrix
            head (int): The head of the OV matrix
        """
        ov_matrix = self._ov_matrix(layer, head)
        
        one_hot_target = torch.zeros(target.shape[0], target.shape[1], ov_matrix.shape[1], device=self.model.device)
        one_hot_target = one_hot_target.scatter_(2, target.unsqueeze(-1), 1)
        logits_mem_input = (ov_matrix @ one_hot_target[:,0,:].T).AB # type: ignore
        logits_cp_input = (ov_matrix @ one_hot_target[:,1,:].T).AB # type: ignore
        
        logits_mem_input = einops.rearrange(logits_mem_input, "vocab batch -> batch vocab")
        logits_cp_input = einops.rearrange(logits_cp_input, "vocab batch -> batch vocab")
        
        logit_mem_mem_input, logit_cp_mem_input, _ , _ = to_logit_token(logit = logits_mem_input, target=target, normalize="none")
        logit_mem_cp_input, logit_cp_cp_input, _ , _ = to_logit_token(logit = logits_cp_input, target=target, normalize="none")
        
        logit_diff_mem_input = logit_mem_mem_input - logit_cp_mem_input
        logit_diff_cp_input = logit_mem_cp_input - logit_cp_cp_input
        
        return logit_diff_mem_input, logit_diff_cp_input

    def compute_logit_dif_single_len(self, length:int, storage:OV_storage, **kwargs):
        self.set_len(length, slice_to_fit_batch=False)
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        
        for batch in dataloader:
            for layer in range(self.model.cfg.n_layers):
                for head in range(self.model.cfg.n_layers):
                    logit_diff_mem_input, logit_diff_cp_input = self.compute_logit_diff_(batch["target"], layer, head, **kwargs)
                    storage.append(logit_diff_mem_input, logit_diff_cp_input, layer=layer, head=head)
        
    def ov_diff(self, **kwargs):
        storage = OV_storage(n_layers=self.model.cfg.n_layers, n_heads=self.model.cfg.n_heads)
        lengths = self.dataset.get_lengths()
        for length in tqdm(lengths):
            if length == 11:
                continue
            self.compute_logit_dif_single_len(length, storage, **kwargs)
        
        return storage.stack()
    
    def run(self, normalize_logit:Literal["none", "softmax", "log_softmax"] = "none"):
        diff_mem_input, diff_cp_input = self.ov_diff(normalize_logit=normalize_logit)
        
        data = []
        for layer in range(self.model.cfg.n_layers):
            for head in range(self.model.cfg.n_heads):
                for i in range(diff_mem_input[f"L{layer}H{head}"].shape[0]):
                    data.append(
                        {
                            "layer": layer,
                            "head": head,
                            "mem_input": diff_mem_input[f"L{layer}H{head}"][i].item(),
                            "cp_input": diff_cp_input[f"L{layer}H{head}"][i].item(),
                        }
                    )
                    # data.append(
                    #     {
                    #         "layer": layer,
                    #         "head": head,
                            # f"mem_input_{i}": diff_mem_input[f"L{layer}H{head}"][i].item(),
                            # f"cp_input_{i}": diff_cp_input[f"L{layer}H{head}"][i].item()
                    #     }
                    # )
        return pd.DataFrame(data)