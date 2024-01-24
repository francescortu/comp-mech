from tomlkit import value
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dataset import TlensDataset
from src.model import WrapHookedTransformer
from src.base_experiment import BaseExperiment
from typing import  Dict, Literal
import pandas as pd


class HeadPatternStorage():
    def __init__(self, n_layers:int, n_heads:int, experiment:Literal["copyVSfact", "contextVSfact"]):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.experiment:Literal["copyVSfact", "contextVSfact"] = experiment
        self.storage = {f"L{i}H{j}":[] for i in range(n_layers) for j in range(n_heads)}
        
        
    def _get_position_to_aggregate_copyVSfact(self, i:int, object_position:int, length:int):
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
        
    def _get_position_to_aggregate_contextVSfact(self, i:int, object_position:int, length:int, subj_position:int):
        subject_1 = subj_position
        subject_2 = subj_position + 1 if (length - subj_position) > 15 else subject_1
        subject_3 = subj_position + 2 if (length - subj_position) > 17 else subject_2
        object_positions_next = object_position + 1 if object_position < length - 1 else object_position
        subject_pos_pre = subj_position - 1 if subj_position > 0 else 0
        last_position = length - 1
        
        if i == 0:
            return slice(0, object_position)
        if i == 1:
            return object_position
        if i == 2:
            if object_position + 1 == subject_pos_pre:
                return object_position + 1
            else: 
                return slice(object_position + 1, subject_pos_pre)
        if i == 3:
            return subject_pos_pre
        if i == 4:
            return subject_1
        if i == 5:
            return subject_2
        if i == 6:
            return subject_3
        if i == 7:
            if subject_3 + 1 == last_position:
                return subject_3
            else:
                return slice(subject_3 + 1, last_position)
        if i == 8:
            return last_position

        
        
    def _aggregate_pattern(self, pattern:torch.Tensor, object_position:int, **kwargs) -> torch.Tensor:    
        """
        pattern shape: (batch_size, seq_len, seq_len)
        return shape:(batch_size, 13, 13)
        """

        length = pattern.shape[-1]

        if self.experiment == "contextVSfact":
            return self._aggregate_pattern_contextVSfact(pattern, object_position, length, **kwargs)
        elif self.experiment == "copyVSfact":
            return self._aggregate_pattern_copyVSfact(pattern, object_position, length, **kwargs)
        else:
            raise NotImplementedError("Only copyVSfact and contextVSfact are supported")
        
        
    def _aggregate_pattern_contextVSfact(self, pattern:torch.Tensor, object_position:int, length:int, subj_position) -> torch.Tensor:
        batch_size = pattern.shape[0]
        aggregate_result = torch.zeros((batch_size, 9, 9))
        
        for batch_idx in range(batch_size):
            for i in range(9):
                position_to_aggregate_row = self._get_position_to_aggregate_contextVSfact(i, object_position, length, subj_position=subj_position[batch_idx])
                for j in range(9):
                    position_to_aggregate_col = self._get_position_to_aggregate_contextVSfact(j, object_position, length, subj_position=subj_position[batch_idx])
                    value_to_aggregate = pattern[batch_idx, position_to_aggregate_row, position_to_aggregate_col]
                    if value_to_aggregate.ndim == 2:
                        value_to_aggregate = value_to_aggregate.mean(dim=(0, 1))
                    elif value_to_aggregate.ndim == 1:
                        value_to_aggregate = value_to_aggregate.mean(dim=0)
                    aggregate_result[batch_idx, i, j] = value_to_aggregate
        return aggregate_result

    def _aggregate_pattern_copyVSfact(self, pattern:torch.Tensor, object_position:int, length:int, **kwargs) -> torch.Tensor:
        
        aggregate_result = torch.zeros((pattern.shape[0], 13, 13))
        
        for i in range(13):
            position_to_aggregate_row = self._get_position_to_aggregate_copyVSfact(i, object_position, length, **kwargs)
            for j in range(13):
                position_to_aggregate_col = self._get_position_to_aggregate_copyVSfact(j, object_position, length, **kwargs)
                value_to_aggregate = pattern[:, position_to_aggregate_row, position_to_aggregate_col]
                if value_to_aggregate.ndim == 3:
                    value_to_aggregate = value_to_aggregate.mean(dim=(1, 2))
                elif value_to_aggregate.ndim == 2:
                    value_to_aggregate = value_to_aggregate.mean(dim=1)
                aggregate_result[:, i, j] = value_to_aggregate
        return aggregate_result
            
        
        
    def store(self, layer:int, head:int, pattern:torch.Tensor, object_position:int, **kwargs):
        aggregate_pattern = self._aggregate_pattern(pattern, object_position, **kwargs) # (batch_size, 13, 13)
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
        self, dataset: TlensDataset, model: WrapHookedTransformer, batch_size: int, experiment: Literal["copyVSfact", "contextVSfact"],
    ):
        super().__init__(dataset, model, batch_size, experiment)
        
    def _extract_pattern(self, cache, layer: int, head: int):
        pattern = cache[f"blocks.{layer}.attn.hook_pattern"][:, head, :, :]
        return pattern
    
    def extract_single_len(self, length:int, storage:HeadPatternStorage):
        self.set_len(length)
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        
        # initialize storage
    
        object_position = self.dataset.obj_pos[0]
        for batch in tqdm(dataloader, total=len(dataloader)):
            _, cache = self.model.run_with_cache(batch["prompt"])
            for layer in range(self.model.cfg.n_layers):
                for head in range(self.model.cfg.n_heads):
                    pattern = self._extract_pattern(cache, layer, head)
                    if self.experiment == "contextVSfact":
                        storage.store(layer, head, pattern.cpu(), object_position, subj_position=batch["subj_pos"])
                    elif self.experiment == "copyVSfact":
                        storage.store(layer, head, pattern.cpu(), object_position)
                    else:
                        raise NotImplementedError("Only copyVSfact and contextVSfact are supported")
        
        torch.cuda.empty_cache()

    def extract(self) -> Dict[str, torch.Tensor]:
        self.storage = HeadPatternStorage(self.model.cfg.n_layers, self.model.cfg.n_heads, self.experiment)
        for length in tqdm(self.dataset.lengths):
            if length == 11:
                continue
            self.extract_single_len(length, self.storage)
        return self.storage.stack()

    def run(self):
        patter_all_heads = self.extract()
        
        data = []
        if self.experiment == "contextVSfact":
            n_grid = 10
        elif self.experiment == "copyVSfact":
            n_grid = 13
        else:
            raise NotImplementedError("Only copyVSfact and contextVSfact are supported")
        
        for layer in range(self.model.cfg.n_layers):
            for head in range(self.model.cfg.n_heads):
                for source_position in range(n_grid):
                    for dest_position in range(n_grid):
                        data.append({
                            "layer":layer,
                            "head":head,
                            "source_position":source_position,
                            "dest_position":dest_position,
                            "value":patter_all_heads[f"L{layer}H{head}"][:, source_position, dest_position].mean().item()
                        })
                        
        df = pd.DataFrame(data)
        return df