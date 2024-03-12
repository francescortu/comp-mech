import re
import torch
import gensim.downloader as api
from torch.utils.data import Dataset
import json
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Literal
from Src.model import BaseModel

REDC = "\033[91m"
ENDC = "\033[0m"

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
        self.first_subj_pos = []
        self.second_subj_pos = []
        self.subj_len = []
        
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
        self.first_subj_pos = []
        self.second_subj_pos = []
        
        self.subj_len = []
        
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
        self.subj_len = []
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
                "1_subj_pos": self.first_subj_pos[idx],
                "2_subj_pos": self.second_subj_pos[idx],
                "subj_len": self.subj_len[idx],
            }
            
    def get_lengths(self):
        return self.lengths
    
    def __get_prompt__(self, d:Dict) -> str:
        if self.experiment == "copyVSfact":
            return d["template"].format(self.premise, d["target_new"])
        elif self.experiment == "contextVSfact":
            if d["prompt"][0] == " ":
                return d["prompt"]
            else:
                return " " + d["prompt"]
        else:
            raise NotImplementedError(f"Experiment {self.experiment} is not supported")
        
    def __find_first_occurence__(self, prompt:torch.Tensor, target:torch.Tensor) -> int:
        position = -1
        for i in range(prompt.shape[0] - target.shape[0] + 1):
            if torch.all(prompt[i:i+target.shape[0]] == target):
                position = i
                break
        return position
    
    def __find_second_occurence__(self, prompt:torch.Tensor, target:torch.Tensor, first_occurence:int) -> int:
        position = -1
        for i in range(first_occurence + 1, prompt.shape[0] - target.shape[0] + 1):
            if torch.all(prompt[i:i+target.shape[0]] == target):
                position = i
                break
        return position
    
    def find_occurence(self, prompt:torch.Tensor, target:torch.Tensor) -> Tuple[int,int]:
        first_occurence = self.__find_first_occurence__(prompt, target)
        second_occurence = self.__find_second_occurence__(prompt, target, first_occurence)
        return first_occurence, second_occurence

    
    def __find_subj_pos__(self, d:Dict) -> Tuple[int,int, int]:
        subject_string = " " + d["subject"]
        subject_token = self.model.tokenize(subject_string).squeeze(0).cuda()
        prompt_token = d["tokenized_prompt"]
        
        subject_token_len = subject_token.shape[0]
        #find the first occurence of the subject tokens in the prompt tokens
        first_subj_pos, second_subj_pos = self.find_occurence(prompt_token, subject_token)
            
        if first_subj_pos == -1:
            # try with removing the last token
            subject_token = subject_token[:-1]
            subject_token_len = subject_token.shape[0]
            first_subj_pos, second_subj_pos = self.find_occurence(prompt_token, subject_token)
                
        if first_subj_pos == -1:
            raise ValueError(f"Subject token: {subject_token}")
        
        return first_subj_pos, second_subj_pos, subject_token_len-1
    
    def __find_obj_pos__(self, d:Dict) -> int:
        object_string = d["target_new"]
        object_token = self.model.tokenize(object_string).cuda()
        prompt_token = d["tokenized_prompt"]
        
        #find the first occurence of the subject tokens in the prompt tokens
        obj_pos = -1
        for i in range(prompt_token.shape[0] - object_token.shape[0] + 1):
            if torch.all(prompt_token[i:i+object_token.shape[0]] == object_token):
                obj_pos = i
                break
        return obj_pos
    def one_token(self, token:torch.Tensor) -> torch.Tensor:
        if token.shape[0] == 1:
            return token
        else:
            return token[0].unsqueeze(0)
    def __get_lenghts_and_tokenize__(self):
        lengths = []
        log_data = []
        to_remove = []
        
        for d in tqdm(self.full_data, desc="Tokenizing and computing lengths"):
            d["prompt"] = self.__get_prompt__(d)
            d["tokenized_prompt"] = self.model.tokenize(d["prompt"]).squeeze(0).cuda()
            d["target_new_token"] = self.one_token(self.model.tokenize(d["target_new"]).squeeze(0).cuda())
            d["target_true_token"] = self.one_token(self.model.tokenize(d["target_true"]).squeeze(0).cuda())
            d["targets"] = torch.cat(
                (d["target_true_token"], d["target_new_token"]), dim=0
            )
            # if find_subj_pod raises an error, add the point to the log data and continue
            try:
                first_subj_pos, second_subj_pos, subj_len = self.__find_subj_pos__(d)
                d["1_subj_pos"] = first_subj_pos
                d["2_subj_pos"] = second_subj_pos
                d["subj_len"] = subj_len
                
            except ValueError as e:
                for key in d.keys():
                    if isinstance(d[key], torch.Tensor):
                        d[key] = d[key].tolist() 
                log_data.append((d, str(e)))
                # remove the point from the full data
                to_remove.append(d)
                continue

            d["obj_pos"] = self.__find_obj_pos__(d)
            d["length"] = d["tokenized_prompt"].shape[0]
            if d["length"] not in lengths:
                lengths.append(d["length"])
                
        if len(log_data) > 0:
            print(f"{REDC} Found {len(log_data)} errors while tokenizing the prompts. Check the logs for more details... {ENDC}")
            #save in a json file
            with open(f"../logs/tokenization_errors_{self.model.cfg.model_name}.json", "w") as f:
                json.dump(log_data, f, indent=4)
            
        for d in to_remove:
            self.full_data.remove(d)
            
        return lengths
    
    def generate_similarity_data(self, similarity_type:Literal["self-similarity"]):
        if similarity_type == "self-similarity":
            return self.__generate_self_similarity_data__()
        else:
            raise NotImplementedError(f"Similarity type {similarity_type} is not supported")
        
    def __generate_self_similarity_data__(self):
        word2vec = api.load("word2vec-google-news-300")
        similarity_score_list = []
        for d in tqdm(
            self.full_data,
            desc="Generating self similarity tokens (word2vec)",
            total=len(self.full_data),
        ):
            base_target = d["target_true"]
            other_target = d["target_new"]
            # remove first space if present
            if other_target[0] == " ":
                other_target = other_target[1:]
            if base_target[0] == " ":
                base_target = base_target[1:]

            # compute similarity
            try:
                similarity_score = word2vec.similarity(base_target, other_target)  # type: ignore
                similarity_score_list.append(similarity_score)
            except:
                similarity_score = -100
            # save the similarity score
            d["similarity_score"] = similarity_score

        # sort full data by similarity score
        self.full_data = sorted(self.full_data, key=lambda x: x["similarity_score"], reverse=True)
        # split into bins, with the highest similarity scores first
        near_high_similarity_bins = [
            self.full_data[i * 50: 500 + (i) * 50] for i in range(10)
        ]
        high_similarity_bin = [self.full_data[500:1000]]
        remaining_bins = [
            self.full_data[i: i + 1000] for i in range(1000, len(self.full_data), 1000)
        ]
        similarity_score_bins = high_similarity_bin + near_high_similarity_bins + remaining_bins

        # Assign similarity_group to each data point based on the bin it falls into
        for i, bin in enumerate(similarity_score_bins):
            for d in bin:
                d["similarity_group"] = i

        # Count the number of points in each group in full data
        similarity_group_count = {}
        for d in self.full_data:
            similarity_group_count[d["similarity_group"]] = similarity_group_count.get(
                d["similarity_group"], 0
            ) + 1
        print(similarity_group_count)
        return self.full_data
    
    def filter_similarity_data(self):
        similarity_group = self.similarity[1]
        return [d for d in self.full_data if d["similarity_group"] == similarity_group]
    
    def set_len(self, length:int):
        self.len = length
        
        #filter for similarity group
        if self.similarity[0]:
            data = self.filter_similarity_data()
        else:
            data = self.full_data
            
        #filter for length
        data = [d for d in data if d["length"] == length]
        self.prompts = [d["prompt"] for d in data]
        self.tokenized_prompts = [d["tokenized_prompt"] for d in data]
        self.targets = [d["targets"] for d in data]
        self.obj_pos = [d["obj_pos"] for d in data]
        self.first_subj_pos = [d["1_subj_pos"] for d in data]
        self.second_subj_pos = [d["2_subj_pos"] for d in data]
        self.subj_len = [d["subj_len"] for d in data]
        
        self.original_index = [
            i for i, d in enumerate(self.full_data) if d["length"] == length
        ]
        self.check_duplicates()
        
    def check_duplicates(self):
        seen = set()
        for i, d in enumerate(self.full_data):
            if d["prompt"] in seen:
                for j, d2 in enumerate(self.full_data):
                    if j == i:
                        continue
                    if d["prompt"] == d2["prompt"]:
                        if d["target_new"] == d2["target_new"]:
                            if d["target_true"] == d2["target_true"]:
                                print(f"duplicate found: {d}, {d2}")
                                return False
            seen.add(d["prompt"])
        return True
        