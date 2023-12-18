from tomlkit import value
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dataset import TlensDataset
from src.model import WrapHookedTransformer
from src.base_experiment import BaseExperiment
from typing import  Dict
import pandas as pd


class HeadPatternStorage():
    def __init__(self, n_layers:int, n_heads:int):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.storage = {f"L{i}H{j}":[] for i in range(n_layers) for j in range(n_heads)}
        
    def _get_position_to_aggregate(self, i:int, object_position:int, length:int):
        subject_1_1 = 5
        subject_1_2 = 6 if length > 15 else 5
        subject_1_3 = 7 if length > 17 else subject_1_2
        subject_2_1 = object_position + 2
        subject_2_2 = object_position + 3 if length > 15 else subject_2_1
        subject_2_3 = object_position + 4 if length > 17 else subject_2_2
        subject_2_2 = subject_2_2 if subject_2_2 < length else subject_2_1
        subject_2_3 = subject_2_3 if subject_2_3 < length else subject_2_2
        last_position = length - 1
        object_positions_pre = object_position - 1
        object_positions_next = object_position + 1
        
        if i == 0:
            return slice(0, subject_1_1)
        if i == 1:
            return subject_1_1
        if i == 2:
            return subject_1_2
        if i == 3:
            return subject_1_3
        if i == 4:
            return slice(subject_1_3 + 1, object_positions_pre)
        if i == 5:
            return object_positions_pre
        if i == 6:
            return object_position
        if i == 7:
            return object_positions_next
        if i == 8:
            return subject_2_1
        if i == 9:
            return subject_2_2
        if i == 10:
            return subject_2_3
        if i == 11:
            return slice(subject_2_3 + 1, last_position)
        if i == 12:
            return last_position
        
        
    def _aggregate_pattern(self, pattern:torch.Tensor, object_position:int) -> torch.Tensor:    
        """
        pattern shape: (batch_size, seq_len, seq_len)
        return shape:(batch_size, 13, 13)
        """
        length = pattern.shape[-1]

        
        aggregate_result = torch.zeros((pattern.shape[0], 13, 13))
        
        for i in range(13):
            position_to_aggregate_row = self._get_position_to_aggregate(i, object_position, length)
            for j in range(13):
                position_to_aggregate_col = self._get_position_to_aggregate(j, object_position, length)
                value_to_aggregate = pattern[:, position_to_aggregate_row, position_to_aggregate_col]
                if value_to_aggregate.ndim == 3:
                    value_to_aggregate = value_to_aggregate.mean(dim=(1, 2))
                elif value_to_aggregate.ndim == 2:
                    value_to_aggregate = value_to_aggregate.mean(dim=1)
                aggregate_result[:, i, j] = value_to_aggregate
        return aggregate_result
            
        
        
    def store(self, layer:int, head:int, pattern:torch.Tensor, object_position:int):
        aggregate_pattern = self._aggregate_pattern(pattern, object_position) # (batch_size, 13, 13)
        self.storage[f"L{layer}H{head}"].append(aggregate_pattern)

    def stack(self):
        """
        from a dict of list of tensors to a dict of tensor
        """
        stacked_result = {}
        for keys in self.storage.keys():
            stacked_result[keys] = torch.cat(self.storage[keys], dim=0)
        
        return stacked_result
    

class HeadPattern(BaseExperiment):
    def __init__(
        self, dataset: TlensDataset, model: WrapHookedTransformer, batch_size: int
    ):
        super().__init__(dataset, model, batch_size)
        
    def _extract_pattern(self, cache, layer: int, head: int):
        pattern = cache[f"blocks.{layer}.attn.hook_pattern"][:, head, :, :]
        return pattern
    
    def extract_single_len(self, length:int, storage:HeadPatternStorage):
        self.set_len(length)
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        
        # initialize storage
    
        object_position = self.dataset.obj_pos[0]
        for batch in dataloader:
            _, cache = self.model.run_with_cache(batch["prompt"])
            for layer in range(self.model.cfg.n_layers):
                for head in range(self.model.cfg.n_heads):
                    pattern = self._extract_pattern(cache, layer, head)
                    storage.store(layer, head, pattern.cpu(), object_position)
        
        torch.cuda.empty_cache()

    def extract(self) -> Dict[str, torch.Tensor]:
        self.storage = HeadPatternStorage(self.model.cfg.n_layers, self.model.cfg.n_heads)
        for length in tqdm(self.dataset.lengths):
            if length == 11:
                continue
            self.extract_single_len(length, self.storage)
        return self.storage.stack()

    def run(self):
        patter_all_heads = self.extract()
        
        data = []
        for layer in range(self.model.cfg.n_layers):
            for head in range(self.model.cfg.n_heads):
                for source_position in range(13):
                    for dest_position in range(13):
                        data.append({
                            "layer":layer,
                            "head":head,
                            "source_position":source_position,
                            "dest_position":dest_position,
                            "value":patter_all_heads[f"L{layer}H{head}"][:, source_position, dest_position].mean().item()
                        })
                        
        df = pd.DataFrame(data)
        return df