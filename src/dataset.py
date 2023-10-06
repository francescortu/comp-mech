import torch
import transformer_lens
from src.model import WrapHookedTransformer
import random


class Dataset:
    def __init__(self, target_dataset, orthogonal_dataset, model:WrapHookedTransformer):
        self.target_dataset = target_dataset
        self.orthogonal_dataset = orthogonal_dataset
        self.target_dataset_per_length, self.orthogonal_dataset_per_length = self.split_for_lenght()

    def random_sample(self, n, choose_lenght=None):
        possible_lengths = []
        for length in self.target_dataset_per_length.keys():
            if len(self.target_dataset_per_length[length]) >= n and len(self.orthogonal_dataset_per_length[length]) >= n:
                possible_lengths.append(length)
        print(f"possible_lengths for sampling {n}: {possible_lengths}")
        if length is None:
            length = random.choice(possible_lengths)
        else:
            length = choose_lenght
      
        self.dataset_per_length = {length: random.sample(self.target_dataset_per_length[length], n) + random.sample(self.orthogonal_dataset_per_length[length], n)}
        
    def split_for_lenght(self):
        target_dataset_per_length = {}
        for d in self.target_dataset:
            length = d["length"]
            if length not in target_dataset_per_length:
                target_dataset_per_length[length] = []
            target_dataset_per_length[length].append(d)
            
        orthogonal_dataset_per_length = {}
        for d in self.orthogonal_dataset:
            length = d["length"]
            if length not in orthogonal_dataset_per_length:
                orthogonal_dataset_per_length[length] = []
            orthogonal_dataset_per_length[length].append(d)
        return target_dataset_per_length, orthogonal_dataset_per_length
    
    def logits(self, model:WrapHookedTransformer):
        logits_per_length = {}
        for length, dataset in self.dataset_per_length.items():
            input_ids = model.to_tokens([d["premise"] for d in dataset])
            logits_per_length[length] = model(input_ids)
        return logits_per_length
  
    def get_tensor_token(self,model):
        tensor_token_per_length = {}
        for length, dataset in self.dataset_per_length.items():
            if length not in tensor_token_per_length:
                tensor_token_per_length[length] = {}
            tensor_token_per_length[length]["target"] = model.to_tokens([d["target"] for d in dataset], prepend_bos=False)
            tensor_token_per_length[length]["orthogonal_token"] = model.to_tokens([d["orthogonal_token"] for d in dataset], prepend_bos=False)
        
        for length, tensor in tensor_token_per_length.items():
            tensor_token_per_length[length]["target"] = tensor_token_per_length[length]["target"].squeeze(1)
            if len(tensor_token_per_length[length]["orthogonal_token"].shape) > 1 :
                tensor_token_per_length[length]["orthogonal_token"] = tensor_token_per_length[length]["orthogonal_token"][:,0]
            tensor_token_per_length[length]["orthogonal_token"] = tensor_token_per_length[length]["orthogonal_token"]
        return tensor_token_per_length