from src.dataset import TlensDataset
from src.model import WrapHookedTransformer
from src.base_experiment import BaseExperiment
import einops
import torch
from tqdm import tqdm
from functools import partial
from copy import deepcopy



class AttentionPattern(BaseExperiment):
    def __init__(self, model: WrapHookedTransformer, dataset: TlensDataset, batch_size, filter_outliers=False):
        super().__init__(dataset, model, batch_size, filter_outliers)
                
    def get_attention_pattern_single_len(self, length, aggregate=False):
        self.set_len(length, slice_to_fit_batch=False)
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        num_batches = len(dataloader)
        assert num_batches > 0, f"Lenght {length} has no examples"

        # attention_pattern = torch.zeros((self.model.cfg.n_layers, self.model.cfg.n_heads, len, len, self.num_batches, self.batch_size))
        attention_pattern = [[] for _ in range(self.model.cfg.n_layers)]
        for idx, batch in tqdm(enumerate(dataloader), total=num_batches, desc=f"Attention pattern at len {length}"):
            logit, cache = self.model.run_with_cache(batch["corrupted_prompts"])
            for layer in range(self.model.cfg.n_layers):
                pattern = cache["pattern", layer] # (batch_size, n_heads, len, len)
                attention_pattern[layer].append(pattern.cpu()) # list of [[(batch_size, n_heads, len, len),..], ...] for each layer
                
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
    
    def attention_pattern_all_len(self):
        lenghts = self.dataset.get_lengths()
        attention_pattern = {}
        for l in lenghts:
            attention_pattern[l] = self.get_attention_pattern_single_len(l, aggregate=True)
        
        result_attn_pattern = torch.cat(list(attention_pattern.values()), dim=0)
        return result_attn_pattern