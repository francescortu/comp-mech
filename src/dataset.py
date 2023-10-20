import torch
import transformer_lens
from src.model import WrapHookedTransformer
import random
from torch.utils.data import Dataset as TorchDataset
import json
import copy
import einops
torch.manual_seed(0)

class Dataset(TorchDataset):
    def __init__(self, dataset_path):
        self.dataset = json.load(open(dataset_path))
        

    def __len__(self):
        if self.dataset_key == "mem_dataset":
            return len(self.mem_dataset)
        if self.dataset_key == "cp_dataset":
            return len(self.cp_dataset)
    def __getitem__(self, idx):
        assert self.dataset_key is not None, "dataset_key not set"
        if self.dataset_key == "mem_dataset":
            return self.mem_dataset[idx]
        if self.dataset_key == "cp_dataset":
            return self.cp_dataset[idx]

    def random_sample(self, n=None, choose_lenght=None):
        random.seed(43)
        self.target_dataset_per_length, self.orthogonal_dataset_per_length = self.split_for_lenght()
        possible_lengths = []
        for length in self.target_dataset_per_length.keys():
            if len(self.target_dataset_per_length[length]) >= n and len(self.orthogonal_dataset_per_length[length]) >= n:
                possible_lengths.append(length)
        print(f"possible_lengths for sampling {n}: {possible_lengths}")
        assert len(possible_lengths) > 0, "No possible lengths for sampling"
        assert choose_lenght is None or choose_lenght in possible_lengths, "choose_lenght not in possible_lengths"
        if length is None:
            length = random.choice(possible_lengths)
        else:
            length = choose_lenght
        
        #random sample
        self.pos_dataset = random.sample(self.target_dataset_per_length[length], n)
        self.neg_dataset = random.sample(self.orthogonal_dataset_per_length[length], n)
        
    def select_lenght(self, n:int, lenght):
        random.seed(1002)
        self.mem_dataset_per_length, self.cp_dataset_per_length = self.split_for_lenght()
        if n > len(self.mem_dataset_per_length[lenght]):
            self.mem_dataset = random.sample(self.mem_dataset_per_length[lenght], len(self.mem_dataset_per_length[lenght]))
        else:
            self.mem_dataset = random.sample(self.mem_dataset_per_length[lenght], n)
        if n > len(self.cp_dataset_per_length[lenght]):
            self.cp_dataset = random.sample(self.cp_dataset_per_length[lenght], len(self.cp_dataset_per_length[lenght]))
        else:
            self.cp_dataset = random.sample(self.cp_dataset_per_length[lenght], n)
        print(f"mem_dataset lenght: {len(self.mem_dataset)}")
        print(f"cp_dataset lenght: {len(self.cp_dataset)}")
        
    def select_dataset(self, dataset:str):
        assert dataset in ["mem", "cp"], "dataset must be mem or cp"
        if dataset == "mem":
            self.dataset_key = "mem_dataset"
        else:
            self.dataset_key = "cp_dataset"
        
        
    def split_for_lenght(self):
        target_dataset_per_length = {}
        for d in self.dataset["memorizing_win"]:
            length = d["length"]
            if length not in target_dataset_per_length:
                target_dataset_per_length[length] = []
            target_dataset_per_length[length].append(d)
            
        orthogonal_dataset_per_length = {}
        for d in self.dataset["copying_win"]:
            length = d["length"]
            if length not in orthogonal_dataset_per_length:
                orthogonal_dataset_per_length[length] = []
            orthogonal_dataset_per_length[length].append(d)
        return target_dataset_per_length, orthogonal_dataset_per_length
    
    def mean_var(self):
        target_prob = [d["target_probs"] for d in self.dataset["copying_win"]]
        orthogonal_prob = [d["orthogonal_probs"] for d in self.dataset["copying_win"]]

        target_prob = torch.tensor(target_prob)
        orthogonal_prob = torch.tensor(orthogonal_prob)
        print(target_prob.mean(), target_prob.var(), orthogonal_prob.mean(), orthogonal_prob.var())
        return target_prob.mean(), target_prob.var(), orthogonal_prob.mean(), orthogonal_prob.var()

    def compute_noise_level(self, model, num_sample=1000):
        random.seed(43)
        #sample random examples
        target_dataset = random.sample(self.dataset["copying_win"], num_sample)
        orthogonal_dataset = random.sample(self.dataset["copying_win"], num_sample)
        text = "".join([d["premise"] for d in target_dataset])
        text += "".join([d["premise"] for d in orthogonal_dataset])
        #compute noise level
        tokens = model.to_tokens(text)
        input_embeddings = model.embed(tokens)
        self.noise_std = (input_embeddings.std(dim=-2)).squeeze(0)

        
    def add_noise(self, model, prompt, noise_index, target_win=None, noise_mlt=2):
        if not hasattr(self, "noise_std"):
            self.compute_noise_level(model)
        tokens = model.to_tokens(prompt)
        input_embeddings = model.embed(tokens)  # (batch_size, seq_len, emb_dim)

        # noise = torch.normal(mean=0, std=0.04, size=input_embeddings.shape, device=input_embeddings.device)
        # Load noise standard deviation and create noise tensor
        noise_std = self.noise_std * noise_mlt
        torch.manual_seed(0)
        noise_std = einops.repeat(noise_std, 'd -> b s d', b=input_embeddings.shape[0], s=input_embeddings.shape[1])
        # noise_mean = einops.repeat(noise_mean, 'd -> b s d', b=input_embeddings.shape[0], s=input_embeddings.shape[1])
        noise = torch.normal(mean=torch.zeros_like(input_embeddings), std=noise_std)
        # noise = torch.normal(mean=torch.zeros_like(input_embeddings), std=noise_mlt)
        # Create a mask for positions specified in noise_index
        seq_len = input_embeddings.shape[1]
        noise_mask = torch.zeros(seq_len, device=input_embeddings.device)
        noise_mask[noise_index] = 1

        # If target_win is an integer, modify the noise_mask and noise tensor
        if isinstance(target_win, int):
            for idx in noise_index:
                if idx + target_win < seq_len:
                    noise_mask[idx + target_win] = 1
                    noise[:, idx + target_win, :] = noise[:, idx, :]

        # Expand the mask dimensions to match the noise tensor shape
        noise_mask = noise_mask.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1)
        noise_mask = noise_mask.expand_as(input_embeddings)  # (batch_size, seq_len, emb_dim)

        noise = noise.to(input_embeddings.device)
        noise_mask = noise_mask.to(input_embeddings.device)
        # Apply the mask to the noise tensor
        masked_noise = noise * noise_mask

        # Add the masked noise to the input embeddings
        corrupted_embeddings = input_embeddings + masked_noise
        # print("CORRUPTED EMBEDDINGS", corrupted_embeddings)
        return corrupted_embeddings
    
    def print_statistics(self):
        print("----------------------------")
        print("Memorizing win number of examples:", len(self.dataset["memorizing_win"]))
        print("Copying win number of examples:", len(self.dataset["copying_win"]))
        print("Mean of of memorized token probs in memorizing win:", torch.tensor([d["target_probs"] for d in self.dataset["memorizing_win"]]).mean())
        print("Mean of of memorized token probs in copying win:", torch.tensor([d["target_probs"] for d in self.dataset["copying_win"]]).mean())
        print("Mean of of orthogonal token probs in copying win:", torch.tensor([d["orthogonal_probs"] for d in self.dataset["copying_win"]]).mean())
        print("Mean of of orthogonal token probs in memorizing win:", torch.tensor([d["orthogonal_probs"] for d in self.dataset["memorizing_win"]]).mean())
        #print len per length
        print("----------------------------")
        mem_dataset_per_length, cp_dataset_per_length = self.split_for_lenght()
        print("Memorizing win number of examples per length:")
        for length in mem_dataset_per_length.keys():
            print(f"length {length}: {len(mem_dataset_per_length[length])}")
        print("Copying win number of examples per length:")
        for length in cp_dataset_per_length.keys():
            print(f"length {length}: {len(cp_dataset_per_length[length])}")
    def filter(self, filter_key="cp", filter_interval=(0,1)):
        if filter_key == "cp":
            self.dataset["copying_win"] = [d for d in self.dataset["copying_win"] if d["orthogonal_probs"] < filter_interval[1] and d["orthogonal_probs"] > filter_interval[0]]
        elif filter_key == "mem":
            self.dataset["memorizing_win"] = [d for d in self.dataset["memorizing_win"] if d["target_probs"] < filter_interval[1] and d["target_probs"] > filter_interval[0]]