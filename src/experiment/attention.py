from src.base_experiment import BaseExperiment
import einops
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader



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