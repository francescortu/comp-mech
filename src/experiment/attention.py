from math import log
from src.base_experiment import BaseExperiment
import einops
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.base_experiment import to_logit_token



class AttentionPattern(BaseExperiment):        
    @classmethod
    def from_experiment(cls, experiment: BaseExperiment):
        return cls(experiment.dataset, experiment.model,  experiment.batch_size, experiment.filter_outliers)
                
    def get_attention_pattern_single_len(self, length:int, aggregate:bool=False) -> torch.Tensor:
        self.set_len(length, slice_to_fit_batch=False)
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        num_batches = len(dataloader)
        assert num_batches > 0, f"Lenght {length} has no examples"

        # attention_pattern = torch.zeros((self.model.cfg.n_layers, self.model.cfg.n_heads, len, len, self.num_batches, self.batch_size))
        attention_pattern = [[] for _ in range(self.model.cfg.n_layers)]
        for idx, batch in tqdm(enumerate(dataloader), total=num_batches, desc=f"Attention pattern at len {length}"):
            _, cache = self.model.run_with_cache(batch["corrupted_prompts"])
            for layer in range(self.model.cfg.n_layers):
                pattern = cache["pattern", layer].cpu() # (batch_size, n_heads, len, len)
                attention_pattern[layer].append(pattern) # list of [[(batch_size, n_heads, len, len),..], ...] for each layer
            torch.cuda.empty_cache()
        # from list of list to list of tensor cat along the batch dimension
        attention_pattern = [torch.cat(layer, dim=0) for layer in attention_pattern] # list of [(num_batches, n_heads, len, len), ...] for each layer
        
        # from list of tensor to tensor of shape (n_layers, num_batches, n_heads, len, len)
        attention_pattern = torch.stack(attention_pattern, dim=0)
        
        # rearrange to (n_batches, n_layers, n_heads, len, len)
        attention_pattern = einops.rearrange(attention_pattern, "l b h p q -> b l h p q")
        object_positions = self.dataset.obj_pos[0]
        if aggregate:
            attention_pattern = self.aggregate_result(object_positions, attention_pattern, length, dim=-2)

        return attention_pattern
    
    def attention_pattern_all_len(self) -> torch.Tensor:
        lenghts = self.dataset.get_lengths()
        attention_pattern = {}
        for le in lenghts:
            if le != 11:
                attention_pattern[le] = self.get_attention_pattern_single_len(le, aggregate=True)
        
        result_attn_pattern = torch.cat(list(attention_pattern.values()), dim=0)
        return result_attn_pattern
    
    def check_diff_after_head_single_len(self, length:int, layer:int, head:int):
        self.set_len(length, slice_to_fit_batch=False)
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        num_batches = len(dataloader)
        assert num_batches > 0, f"Lenght {length} has no examples"
        
        logit_diff_before = []
        logit_diff_after = []
        for idx, batch in tqdm(enumerate(dataloader), total=num_batches, desc=f"Attention pattern at len {length}", disable=True):
            logit_before, cache = self.model.run_with_cache(batch["corrupted_prompts"])
            residual_stream_before = cache["resid_pre", layer]
            output_attention_after = cache[f"blocks.{layer}.attn.hook_z"][:,:,head,:]
            # project to residual stream
            # compute logit
            residual_output_head = einops.einsum(self.model.blocks[layer].attn.W_O, output_attention_after, "n_heads d_head d_model, batch pos d_head -> batch pos d_model") + residual_stream_before # type: ignore
            logit_residual_stream_before = einops.einsum(self.model.W_U, residual_stream_before, "d_model d_v, batch pos d_model -> batch pos d_v")
            logit_residual_output_head = einops.einsum(self.model.W_U, residual_output_head, "d_model d_v, batch pos d_model -> batch pos d_v")
            #layer_norm
            logit_residual_stream_before = self.model.ln_final(logit_residual_stream_before)
            logit_residual_output_head = self.model.ln_final(logit_residual_output_head) 
            
            mem_pre, cp_pre = to_logit_token(logit_residual_stream_before[:,-1,:], batch["target"])
            mem_post, cp_post = to_logit_token(logit_residual_output_head[:,-1,:], batch["target"])
            logit_diff_before.append((mem_pre - cp_pre).abs())
            logit_diff_after.append((mem_post - cp_post).abs())
            
            torch.cuda.empty_cache()    
        
        logit_diff_before = torch.cat(logit_diff_before, dim=0)
        logit_diff_after = torch.cat(logit_diff_after, dim=0)
        return logit_diff_before, logit_diff_after
    
    def check_diff_after_head_all_len(self, layer:int, head:int):
        lenghts = self.dataset.get_lengths()
        logit_diff_before = {}
        logit_diff_after = {}
        for le in lenghts:
            if le != 11:
                logit_diff_before[le], logit_diff_after[le] = self.check_diff_after_head_single_len(le, layer, head)
        
        result_logit_diff_before = torch.cat(list(logit_diff_before.values()), dim=0)
        result_logit_diff_after = torch.cat(list(logit_diff_after.values()), dim=0)
        return result_logit_diff_before, result_logit_diff_after
            
            
    def ov_single_len(self, length:int, layer:int, head:int):
        self.set_len(length, slice_to_fit_batch=False)
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        num_batches = len(dataloader)
        assert num_batches > 0, f"Lenght {length} has no examples"
        W_OV = (self.model.blocks[layer].attn.W_V @ self.model.blocks[layer].attn.W_O)[head] # type: ignore

        for idx, batch in tqdm(enumerate(dataloader)):
            #!todo
            raise NotImplementedError
        
    
    def get_ov_matrix(self, layer:int, head:int):
        W_OV = (self.model.blocks[layer].attn.W_V @ self.model.blocks[layer].attn.W_O)[head]
        return W_OV
        