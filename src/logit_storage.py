import torch
from typing import Tuple, Optional
import einops

    
class LogitStorage:
    """
    store logits and return shape (layers, position, examples) or (layers, position, heads, examples)
    """
    def __init__(self, n_layers: int, length: int, n_heads: int = 1):
        self.n_layers = n_layers
        self.length = length
        self.n_heads = n_heads
        self.size = n_layers * length * n_heads

        # Using a dictionary to store logit values
        self.logits = {
            'mem_logit': [[] for _ in range(self.size)],
            'cp_logit': [[] for _ in range(self.size)],
        }

    def _get_index(self, layer: int, position: int, head: int = 0):
        return (layer * self.length + position) * self.n_heads + head
    
    def _reshape_tensor(self, tensor:torch.Tensor):
        return einops.rearrange(tensor, "layer position examples -> examples layer position")
    
    def _reshape_tensor_back(self, tensor:torch.Tensor):
        return einops.rearrange(tensor, "examples layer position -> layer position examples")
    
    def _aggregate_result(self, object_positions:int, pattern:torch.Tensor, dim:int=-1) -> torch.Tensor:
        
        pattern = self._reshape_tensor(pattern)
        subject_1_1 = 5
        subject_1_2 = 6 if self.length > 15 else 5
        subject_1_3 = 7 if self.length > 17 else subject_1_2
        subject_2_1 = object_positions + 2
        subject_2_2 = object_positions + 3 if self.length > 15 else subject_2_1
        subject_2_3 = object_positions + 4 if self.length > 17 else subject_2_2
        subject_2_2 = subject_2_2 if subject_2_2 < self.length else subject_2_1
        subject_2_3 = subject_2_3 if subject_2_3 < self.length else subject_2_2
        last_position = self.length - 1
        object_positions_pre = object_positions - 1
        object_positions_next = object_positions + 1
        *leading_dims, pen_len, last_len = pattern.shape
        
        result_aggregate = torch.zeros((*leading_dims, 13,13))
        intermediate_aggregate = torch.zeros((*leading_dims, pen_len, 13))
        #aggregate for pre-last dimension
        intermediate_aggregate[..., 0] = pattern[..., :subject_1_1].mean(dim=-1)
        intermediate_aggregate[..., 1] = pattern[..., subject_1_1]
        intermediate_aggregate[..., 2] = pattern[..., subject_1_2]
        intermediate_aggregate[..., 3] = pattern[..., subject_1_3]
        intermediate_aggregate[..., 4] = pattern[..., subject_1_3 + 1:object_positions_pre].mean(dim=-1)
        intermediate_aggregate[..., 5] = pattern[..., object_positions_pre]
        intermediate_aggregate[..., 6] = pattern[..., object_positions]
        intermediate_aggregate[..., 7] = pattern[..., object_positions_next]
        intermediate_aggregate[..., 8] = pattern[..., subject_2_1]
        intermediate_aggregate[..., 9] = pattern[..., subject_2_2]
        intermediate_aggregate[..., 10] = pattern[..., subject_2_3]
        intermediate_aggregate[..., 11] = pattern[..., subject_2_3 + 1:last_position].mean(dim=-1)
        intermediate_aggregate[..., 12] = pattern[..., last_position]
        if dim == -1:
            return self._reshape_tensor_back(intermediate_aggregate)

        result_aggregate[..., 0,:] = intermediate_aggregate[..., :subject_1_1,:].mean(dim=-2)
        result_aggregate[..., 1,:] = intermediate_aggregate[..., subject_1_1,:]
        result_aggregate[..., 2,:] = intermediate_aggregate[..., subject_1_2,:]
        result_aggregate[..., 3,:] = intermediate_aggregate[..., subject_1_3,:]
        result_aggregate[..., 4,:] = intermediate_aggregate[..., subject_1_3 + 1:object_positions_pre,:].mean(dim=-2)
        result_aggregate[..., 5,:] = intermediate_aggregate[..., object_positions_pre,:]
        result_aggregate[..., 6,:] = intermediate_aggregate[..., object_positions,:]
        result_aggregate[..., 7,:] = intermediate_aggregate[..., object_positions_next,:]
        result_aggregate[..., 8,:] = intermediate_aggregate[..., subject_2_1,:]
        result_aggregate[..., 9,:] = intermediate_aggregate[..., subject_2_2,:]
        result_aggregate[..., 10,:] = intermediate_aggregate[..., subject_2_3,:]
        result_aggregate[..., 11,:] = intermediate_aggregate[..., subject_2_3+ 1:last_position,:].mean(dim=-2)
        result_aggregate[..., 12,:] = intermediate_aggregate[..., last_position,:]
        print(result_aggregate.shape)
        return result_aggregate


    def store(self, layer: int, position: int, logit:Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]], head: int = 0):
        mem_logit, cp_logit, _, _ = logit
        index = self._get_index(layer, position, head)
        self.logits['mem_logit'][index].append(mem_logit)
        self.logits['cp_logit'][index].append(cp_logit)

    

    def _reshape_logits(self, logits_list, shape):
        return torch.stack([torch.cat(logits, dim=0) for logits in logits_list]).view(shape)

    def get_logit(self):
        shape = (self.n_layers, self.length, -1)
        return tuple(self._reshape_logits(self.logits[key], shape) for key in self.logits)
    
    def get_aggregate_logit(self, object_position:int):
        return_tuple = self.get_logit()
        aggregate_tensor = []
        for elem in return_tuple:
            aggregate_tensor.append(self._aggregate_result(object_position, elem))
        
        return tuple(aggregate_tensor)
    
class IndexLogitStorage(LogitStorage):
    def __init__(self, n_layers: int, length: int, n_heads: int = 1):
        super().__init__(n_layers, length, n_heads)
        self.logits.update({
            'mem_logit_idx': [[] for _ in range(self.size)],
            'cp_logit_idx': [[] for _ in range(self.size)]
        })
        
    def store(self, layer: int, position: int, logit:Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]],  head: int = 0):
        mem_logit, cp_logit, mem_logit_idx, cp_logit_idx = logit
        index = self._get_index(layer, position, head)
        self.logits['mem_logit'][index].append(mem_logit)
        self.logits['cp_logit'][index].append(cp_logit)
        # Handle None values for mem_logit_idx and cp_logit_idx
        self.logits['mem_logit_idx'][index].append(mem_logit_idx if mem_logit_idx is not None else torch.tensor([]))
        self.logits['cp_logit_idx'][index].append(cp_logit_idx if cp_logit_idx is not None else torch.tensor([]))
        



class HeadLogitStorage(IndexLogitStorage, LogitStorage):
    def __init__(self, n_layers: int, length: int, n_heads: int):
        super().__init__(n_layers, length, n_heads)
        
    @classmethod
    def from_logit_storage(cls, logit_storage: LogitStorage, n_heads: int):
        return cls(logit_storage.n_layers, logit_storage.length, n_heads)
    
    @classmethod
    def from_index_logit_storage(cls, index_logit_storage: IndexLogitStorage, n_heads: int):
        return cls(index_logit_storage.n_layers, index_logit_storage.length, n_heads)
    
    def _reshape_tensor(self, tensor: torch.Tensor):
        return einops.rearrange(tensor, "layer position head examples -> examples layer head position")
    
    def _reshape_tensor_back(self, tensor: torch.Tensor):
        return einops.rearrange(tensor, "examples layer head position -> layer head position examples")

    def get_logit(self):
        shape = (self.n_layers, self.length, self.n_heads, -1)
        return tuple(self._reshape_logits(self.logits[key], shape) for key in self.logits)



#