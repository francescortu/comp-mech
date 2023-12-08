from copy import deepcopy
from math import log
import re

from cvxpy import length
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
            
            mem_pre, cp_pre, _, _ = to_logit_token(logit_residual_stream_before[:,-1,:], batch["target"])
            mem_post, cp_post, _, _ = to_logit_token(logit_residual_output_head[:,-1,:], batch["target"])
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
        
    
    def get_ov_matrix(self, layer:int, head:int, remember=False):
        """WARNING: the return matrix could be huge!!"""
        W_OV = (self.model.blocks[layer].attn.W_V @ self.model.blocks[layer].attn.W_O)[head] # type: ignore
        return torch.matmul(torch.matmul(self.model.W_U.T,W_OV), self.model.W_E.T).cpu()
    
    def get_ov_output(self, layer:int, head:int, token, mlp=True):
        token_embedding = self.model.W_E[token,:] #shape d_vocab, d_model
        # reshape to have batch dim and pos dim
        token_embedding = token_embedding.unsqueeze(0).unsqueeze(0)
        
        effective_token_embedding = self.model.blocks[0].mlp(token_embedding) # type:ignore (batch,1,d_model)
        W_OV = (self.model.blocks[layer].attn.W_V @ self.model.blocks[layer].attn.W_O)[head] # type: ignore
        return einops.einsum((self.model.W_U.T @ W_OV), effective_token_embedding, "d_vocab d_model, batch pos d_model -> batch pos d_vocab").cpu()
    
    def get_ov_interaction(self,
                           layer:int,
                           head:int,
                           batch,
                           input_source_pos:str="1_1_subject",
                           mlp=True,
                           max_min_rescale=False,
                           rescale_to_standard=False,
                           return_full_output=False,
                           return_difference=False,
                           random=False,
                           residual_stream=None):
        """
        Return a (batched) matrix 2x2 where (i,j) means that i is the input (source token) of the OV circuit and j is the output (destination token). More in details:
        (0,0) -> how is affected the mem token if attend to the mem token (after MLP)
        (0,1) -> how is affected the copy token if attend to the mem token (after MLP)
        (1,0) -> how is affected the mem token if attend to the copy token (after MLP)
        (1,1) -> how is affected the copy token if attend to the copy token (after MLP)
        """
        # print(self.model.to_string(batch["target"]))
        target = deepcopy(batch["target"])
        input_target = deepcopy(batch["target"])
        input_target[:,0] = self.model.to_tokens(batch["corrupted_prompts"])[:,self.get_position(input_source_pos)]
       
        
        if random:
            target = torch.randint(0, self.model.cfg.d_vocab, size=batch["target"].shape)
            input_target = torch.randint(0, self.model.cfg.d_vocab, size=batch["target"].shape)
            
        # tmp = self.model.to_tokens(" Israel", prepend_bos=False).squeeze(0)
        # print(tmp.shape)
        # input_target[:,0] = einops.repeat(tmp, "1 -> 1 batch", batch=len(batch["corrupted_prompts"]))
        # print(self.model.to_string(input_target))
        # print(self.model.to_string(target))
        # #target is a tensor of shape [batch,2] 
        if residual_stream is not None:
            token_embeddings = residual_stream
        else:
            token_embeddings = self.model.W_E[input_target] # (batch,2,d_model)
        if mlp:
            effective_token_embeddings = self.model.blocks[0].mlp(token_embeddings) # type:ignore (batch,2,d_model) 
        else:
            effective_token_embeddings = token_embeddings
        W_OV = (self.model.blocks[layer].attn.W_V @ self.model.blocks[layer].attn.W_O)[head] # type: ignore
        ov_interaction = einops.einsum((self.model.W_U.T @ W_OV), effective_token_embeddings, "d_vocab d_model, batch n_target d_model -> batch n_target d_vocab") #(batch,2,d_vocab)
        ov_interaction = self.model.ln_final(ov_interaction)
        if max_min_rescale:
           min_vals = ov_interaction.min(dim=-1, keepdim=True).values
           max_vals = ov_interaction.max(dim=-1, keepdim=True).values
           ov_interaction = (ov_interaction - min_vals) / (max_vals - min_vals).clamp(min=1e-6)
            
        elif rescale_to_standard:
                    # Reshape for mean and std deviation calculation across batch and target dimensions
            reshaped_ov_interaction = ov_interaction.view(-1, ov_interaction.size(-1))

            # Compute the mean and standard deviation for each column
            mean_vals = reshaped_ov_interaction.mean(dim=0, keepdim=True)
            std_vals = reshaped_ov_interaction.std(dim=0, keepdim=True)

            # Apply standardization
            ov_interaction = (ov_interaction - mean_vals) / std_vals
        if return_full_output:
            return ov_interaction
        
        elif return_difference:
            diff_source_mem = ov_interaction[torch.arange(batch["target"].shape[0]),0,batch["target"][:,0]] - ov_interaction[torch.arange(batch["target"].shape[0]),0,batch["target"][:,1]] # (batch)
            diff_source_cp = ov_interaction[torch.arange(batch["target"].shape[0]),1,batch["target"][:,0]] - ov_interaction[torch.arange(batch["target"].shape[0]),1,batch["target"][:,1]] # (batch)
            # print(diff_source_mem.abs())
            return diff_source_mem, diff_source_cp
        
        #from (batch,2,d_vocab) to (batch,2,2)
        selected_ov_interaction = ov_interaction[torch.arange(ov_interaction.shape[0])[:, None], :, target]
        
        return selected_ov_interaction #- 0.5
    
    def compute_interaction_per_len(self, length:int, layer:int, head:int, from_resid:bool=False, return_difference=False, **kwargs):
        self.set_len(length, slice_to_fit_batch=False)
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        num_batches = len(dataloader)
        
        assert num_batches > 0, f"Lenght {length} has no examples"
        interaction_matrixes = []
        for idx, batch in tqdm(enumerate(dataloader), total=num_batches, desc=f"Interaction matrix at len {length}"):
            if from_resid:
                _, cache = self.model.run_with_cache(batch["corrupted_prompts"])
                residual_stream = cache["resid_pre", layer] # (batch, len, d_model)
                residual_target = torch.stack([residual_stream[:,5,:], residual_stream[:,batch["obj_pos"][0],:]], dim=1) 
                tmp_interaction = self.get_ov_interaction(layer, head, batch, residual_stream=residual_target, return_difference=return_difference, **kwargs)
            else:
                tmp_interaction = self.get_ov_interaction(layer, head, batch,return_difference=return_difference, **kwargs)
            interaction_matrixes.append(tmp_interaction)
            torch.cuda.empty_cache()
        if return_difference:
            # print(interaction_matrixes)
            tmp_interaction_mem, tmp_interaction_cp = zip(*interaction_matrixes)
            tmp_interaction_mem = torch.cat(tmp_interaction_mem, dim=0)
            tmp_interaction_cp = torch.cat(tmp_interaction_cp, dim=0)
            return tmp_interaction_mem.cpu(), tmp_interaction_cp.cpu()
            
        # interaction matrix is a list of tensor of shape (batch,2,2), [(batch,2,2), ...]
        # convert to tensor of shape (num_batches, batch, 2, 2)
        interaction_matrixes = torch.cat(interaction_matrixes, dim=0)
        torch.cuda.empty_cache()
        return interaction_matrixes.cpu()
        
    def compute_interaction(self, layer:int, head:int, **kwargs):
        lenghts = self.dataset.get_lengths()
        interaction_matrixes = {}
        for le in lenghts:
            if le != 11:
                interaction_matrixes[le] = self.compute_interaction_per_len(le, layer, head, **kwargs)
        
        result_interaction_matrixes = torch.cat(list(interaction_matrixes.values()), dim=0)
        return result_interaction_matrixes
    
    def compute_output_difference(self, layer:int, head:int, **kwargs):
        lengths = self.dataset.get_lengths()
        output_diff_source_mem = {}
        output_diff_source_cp = {}
        for le in lengths:
            if le != 11:
                output_diff_source_mem[le], output_diff_source_cp[le] = self.compute_interaction_per_len(le, layer, head, return_difference=True, **kwargs)
        
        result_output_diff_source_mem = torch.cat(list(output_diff_source_mem.values()), dim=0)
        result_output_diff_source_cp = torch.cat(list(output_diff_source_cp.values()), dim=0)
        return result_output_diff_source_mem, result_output_diff_source_cp
    
    
