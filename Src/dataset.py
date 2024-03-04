import torch
from torch.utils.data import Dataset
import json
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Literal
from Src.model import BaseModel

def load_dataset(path:str, model_name:str, start:Optional[int], end:Optional[int]) -> List[Dict]:
    data = json.load(open(path, "r"))
    if start is None:
        start = 0
    if end is None:
        end = len(data)
    return data[start:end]

def load_similarity_score_dict(model_name:str) -> Dict:
    family_name = get_family_name(model_name)
    return torch.load(f"../data/similarity_score_{family_name}.pt")

def get_family_name(model_name:str) -> str:
    if "gpt2" in model_name:
        return "gpt2"
    elif "llama" in model_name:
        return "llama"
    elif "pythia" in model_name:
        return "pythia"
    else:
        raise NotImplementedError(f"Model {model_name} is not supported")

class BaseDataset(Dataset):
    def __init__(
        self,
        path:str,
        model:BaseModel,
        experiment: Literal["copyVSfact", "contextVSfact"],
        start: Optional[int] = None,
        end: Optional[int] = None,
        similarity: Tuple[bool, int, Literal["self-similarity"]] = (False, 0, "self-similarity"),
        premise:str = "Redefine"
    ):
        self.model = model
        self.experiment = experiment
        self.similarity = similarity
        self.premise = premise
        if similarity[0]:
            if similarity[2] == "self-similarity":
                self.similarity_score_dict = load_similarity_score_dict(self.model.cfg.model_name)
                self.full_data = self.generate_similarity_data(similarity[2])
        else:
            self.full_data = load_dataset(path, self.model.cfg.model_name,start, end)
            
        self.lengths = self.__get_lenghts_and_tokenize__()
        self.prompts = []
        self.tokenized_prompts = []
        self.targets = []
        self.obj_pos = []
        self.subj_pos = []
        
    def reset(
        self,
        new_similarity_level:Optional[int] = None,
    ):
        if self.similarity[0] == True:
            self.similarity = (self.similarity[0], new_similarity_level, self.similarity[2])

        self.lengths = self.__get_lenghts_and_tokenize__()
        self.prompts = []
        self.tokenized_prompts = []
        self.targets = []
        self.obj_pos = []
        self.subj_pos = []
        
    def update(
        self,
        premise:str,
        new_similarity_level:Optional[int] = None,
    ):
        print(
            f"Updating the dataset from {self.premise} to {premise} and the similarity level from {self.similarity[1]} to {new_similarity_level}"
        )
        self.similarity = (self.similarity[0], new_similarity_level, self.similarity[2])
        self.premise = premise
        self.prompts = []
        self.tokenized_prompts = []
        self.targets = []
        self.obj_pos = []
        self.lengths = []
        self.lengths = self.__get_lenghts_and_tokenize__()
        
    def __len__(self):
        if len(self.prompts) == 0:
            raise ValueError("The dataset is empty: please call set_len() first")
        return len(self.prompts)
    
    def __getitem__(self, idx):
        if len(self.prompts) == 0:
            raise ValueError("Dataset is empty: please call set_len() first")
        else:
            return {
                "prompt": self.prompts[idx],
                "input_ids": self.tokenized_prompts[idx],
                "target": self.targets[idx],
                "obj_pos": self.obj_pos[idx],
                "subj_pos": self.subj_pos[idx],
            }
            
    def get_lengths(self):
        return self.lengths
    
    def __get_lenghts_and_tokenize__(self):
        lengths = []
        log_data = []
        