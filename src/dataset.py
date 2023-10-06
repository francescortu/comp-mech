import torch
import transformer_lens
from src.model import WrapHookedTransformer
import random
from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    def __init__(self, target_dataset, orthogonal_dataset, model:WrapHookedTransformer):
        self.target_dataset = target_dataset
        self.orthogonal_dataset = orthogonal_dataset
        self.target_dataset_per_length, self.orthogonal_dataset_per_length = self.split_for_lenght()

    def __len__(self):
        return len(self.pos_dataset)
    def __getitem__(self, idx):
        return {"pos_dataset": self.pos_dataset[idx], "neg_dataset": self.neg_dataset[idx]}

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
      
      
        self.pos_dataset = self.target_dataset_per_length[length]
        self.neg_dataset = self.orthogonal_dataset_per_length[length]

        
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
    

