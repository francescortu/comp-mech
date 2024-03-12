from math import isnan
from tomlkit import value
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from Src.dataset import BaseDataset
from Src.model import BaseModel
from Src.base_experiment import BaseExperiment
from typing import  Dict, Literal
import pandas as pd

from Src.utils import AGGREGATED_DIMS
AGGREGATED_DIMS = 14

class HeadPatternStorage():
    def __init__(self, n_layers:int, n_heads:int, experiment:Literal["copyVSfact", "contextVSfact"]):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.experiment:Literal["copyVSfact", "contextVSfact"] = experiment
        self.storage = {f"L{i}H{j}":[] for i in range(n_layers) for j in range(n_heads)}
        
        
    def _get_position_to_aggregate_copyVSfact(self,
                                            i:int,
                                            object_position:torch.Tensor,
                                            first_subject_position:torch.Tensor,
                                            second_subject_position:torch.Tensor,
                                            subject_lengths:torch.Tensor, 
                                            length:int):
        
        if i == 0: # pre subject
            return slice(0, first_subject_position -1)
        if i == 1: # first subject token
            return first_subject_position 
        if i == 2: # between first and second subject
            last_subject = first_subject_position + subject_lengths
            if (last_subject -1 - (first_subject_position + 1)) < 0:
                return - 100
            elif (last_subject -1 - (first_subject_position + 1)) == 0:
                return first_subject_position + 1
            else:
                return slice(first_subject_position + 1, first_subject_position + last_subject -1)
        if i == 3: # last subject token
            return first_subject_position + subject_lengths
        if i == 4: # between subject and object
            if (object_position - 1 - (first_subject_position + subject_lengths +1)) < 0:
                return - 100
            elif (object_position - 1 - (first_subject_position + subject_lengths +1)) == 0:
                return first_subject_position + subject_lengths + 1
            else:
                return slice(first_subject_position + subject_lengths + 1, object_position - 1)
        if i == 5: # pre object
            return object_position - 1
        if i == 6: # object token
            return object_position
        if i == 7: # next object
            return object_position + 1
        if i == 8: # from next object to second subject
            if (second_subject_position - 1 - (object_position + 1)) < 0:
                return - 100
            elif (second_subject_position - 1 - (object_position + 1)) == 0:
                return object_position + 1
            else:
                return slice(object_position + 1, second_subject_position - 1)
        if i == 9: # first token of second subject
            return second_subject_position
        if i == 10: # between first and second subject
            last_subject = second_subject_position + subject_lengths
            if (last_subject -1 - (second_subject_position + 1)) < 0:
                return - 100
            elif (last_subject -1 - (second_subject_position + 1)) == 0:
                return second_subject_position + 1
            else:
                return slice(second_subject_position + 1, second_subject_position + last_subject -1)
        if i == 11: # last token of second subject
            return second_subject_position + subject_lengths
        if i == 12: # from subject to last token
            last_token = length - 1
            if (last_token - 1 - (second_subject_position + subject_lengths + 1)) < 0:
                return - 100
            elif (last_token - 1 - (second_subject_position + subject_lengths + 1)) == 0:
                return second_subject_position + subject_lengths + 1
            else:
                return slice(second_subject_position + subject_lengths + 1, last_token)
        if i == 13: # last token
            return length - 1
        
    # def _get_position_to_aggregate_contextVSfact(self, i:int, object_position:int, length:int, subj_position:int):
    #     subject_1 = subj_position
    #     subject_2 = subj_position + 1 if (length - subj_position) > 14 else subject_1
    #     subject_3 = subj_position + 2 if (length - subj_position) > 16 else subject_2
    #     subject_pos_pre = subj_position - 1 
    #     last_position = length - 1
        
    #     if i == 0:
    #         return slice(0, object_position)
    #     if i == 1:
    #         return object_position
    #     if i == 2:
    #         slice_object = slice(object_position + 1, subject_pos_pre)
    #         if slice_object.start > slice_object.stop:
    #             print("ERROR")
    #         if object_position + 1 > subj_position:
    #             return object_position
    #         elif object_position + 1 == subject_pos_pre:
    #             return object_position + 1
    #         else: 
    #             return slice(object_position + 1, subject_pos_pre)
    #     if i == 3:
    #         return subject_pos_pre
    #     if i == 4:
    #         return subject_1
    #     if i == 5:
    #         return subject_2
    #     if i == 6:
    #         return subject_3
    #     if i == 7:
    #         if subject_3 + 1 == last_position:
    #             return subject_3
    #         else:
    #             return slice(subject_3 + 1, last_position)
    #     if i == 8:
    #         return last_position

        
        
    def _aggregate_pattern(self, 
                        pattern:torch.Tensor, 
                        object_position:torch.Tensor,
                        first_subject_position:torch.Tensor,
                        second_subject_position:torch.Tensor,
                        subject_lengths,
                        length:int
                        ) -> torch.Tensor:    
        """
        pattern shape: (batch_size, seq_len, seq_len)
        return shape:(batch_size, 13, 13)
        """

        length = pattern.shape[-1]

        if "contextVSfact" in self.experiment:
            raise NotImplementedError("Only copyVSfact and contextVSfact are supported")
        elif "copyVSfact" in self.experiment:
            return_pattern = torch.zeros((pattern.shape[0], AGGREGATED_DIMS, AGGREGATED_DIMS))
            for i in range(pattern.shape[0]):
                return_pattern[i] = self._aggregate_pattern_copyVSfact(pattern[i],
                                                object_position[i],
                                                first_subject_position[i],
                                                second_subject_position[i],
                                                subject_lengths[i],
                                                length 
                                                )
            return return_pattern
        else:
            raise NotImplementedError("Only copyVSfact and contextVSfact are supported")
        
    
    def _aggregate_pattern_copyVSfact(self,
                                    pattern:torch.Tensor,
                                    object_position:torch.Tensor,
                                    first_subject_position:torch.Tensor,
                                    second_subject_position:torch.Tensor,
                                    subject_lengths:torch.Tensor,
                                    length:int,
                                    ) -> torch.Tensor:
        
        assert pattern.ndim == 2, "pattern should be 2D, (seq_len, seq_len) NOT (batch_size, seq_len, seq_len)"
        assert object_position.ndim == 0, "object_position should be 0D, NOT 1D. Not (batch_size) but ()"
        assert first_subject_position.ndim == 0, "first_subject_position should be 0D, NOT 1D. Not (batch_size) but ()"
        assert second_subject_position.ndim == 0, "second_subject_position should be 0D, NOT 1D. Not (batch_size) but ()"
        assert subject_lengths.ndim == 0, "subject_lengths should be 0D, NOT 1D. Not (batch_size) but ()"
        

        aggregate_result = torch.zeros((AGGREGATED_DIMS, AGGREGATED_DIMS))
        
        for i in range(AGGREGATED_DIMS):
            position_to_aggregate_row = self._get_position_to_aggregate_copyVSfact(i, 
                                                                                object_position = object_position,
                                                                                first_subject_position = first_subject_position,
                                                                                second_subject_position = second_subject_position,
                                                                                subject_lengths = subject_lengths,    
                                                                                length = length)
            for j in range(AGGREGATED_DIMS):
                position_to_aggregate_col = self._get_position_to_aggregate_copyVSfact(j, 
                                                                                    object_position = object_position,
                                                                                    first_subject_position = first_subject_position, 
                                                                                    second_subject_position = second_subject_position,
                                                                                    subject_lengths = subject_lengths,
                                                                                    length= length,)
                if position_to_aggregate_row == -100 or position_to_aggregate_col == -100:
                    aggregate_result[i, j] = 0
                    continue
                value_to_aggregate = pattern[position_to_aggregate_row, position_to_aggregate_col]
                if value_to_aggregate.ndim == 2:
                    value_to_aggregate = value_to_aggregate.mean(dim=(0, 1))
                elif value_to_aggregate.ndim == 1:
                    value_to_aggregate = value_to_aggregate.mean(dim=0)
                aggregate_result[i, j] = value_to_aggregate
        return aggregate_result
            
        
        
    def store(self,     
            layer:int, 
            head:int, 
            pattern:torch.Tensor, 
            object_position:torch.Tensor,
            first_subject_position:torch.Tensor,
            second_subject_position:torch.Tensor,
            subject_lengths:torch.Tensor,
            length:int):
        
        aggregate_pattern = self._aggregate_pattern(
                        pattern = pattern,
                        object_position = object_position,
                        first_subject_position = first_subject_position,
                        second_subject_position = second_subject_position,
                        subject_lengths = subject_lengths,
                        length = length
                        ) # (batch_size, 13, 13)
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
        self, dataset: BaseDataset, model: BaseModel, batch_size: int, experiment: Literal["copyVSfact", "contextVSfact"],
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
            _, cache = self.model.run_with_cache(batch["prompt"], prepend_bos=False)
            for layer in range(self.model.cfg.n_layers):
                for head in range(self.model.cfg.n_heads):
                    pattern = self._extract_pattern(cache, layer, head)
                    storage.store(
                        layer=layer,
                        head=head,
                        pattern=pattern.cpu(),
                        object_position=batch["obj_pos"],
                        first_subject_position=batch["1_subj_pos"],
                        second_subject_position=batch["2_subj_pos"],
                        subject_lengths=batch["subj_len"],
                        length=length
                    )
                            
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
        if "contextVSfact" in self.experiment:
            n_grid = 9
        elif  "copyVSfact" in self.experiment:
            n_grid = AGGREGATED_DIMS
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
