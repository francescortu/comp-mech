import sys
sys.path.append('..')
sys.path.append('../src')
sys.path.append('../data')

import json
from tqdm import tqdm

import json
import torch
from torch.utils.data import Dataset
import einops

class MyDataset(Dataset):
    def __init__(self, path, tokenizer, slice=None):
        with open(path, 'r') as file:
            self.full_data = json.load(file)
        
        if slice is not None:
            self.full_data = self.full_data[:slice]
        # Initialize variables to avoid AttributeError before calling set_len
        self.prompts = []
        self.target = []
        self.obj_pos = []
        self.tokenizer = tokenizer
        self.lenghts = self.get_lengths()
        
    def __len__(self):
        return len(self.data) 
    
    def __getitem__(self, idx):
        if not self.prompts:
            raise ValueError("Set length using set_len before fetching items.")
        return {
            "prompt": self.prompts[idx],
            "input_ids": self.input_ids[idx],
            "target": self.target[idx],
            "obj_pos": self.obj_pos[idx],
        }
    
    def set_len(self, length):
        self.data = [d for d in self.full_data if d["length"] == length]
        self.original_index = [i for i, d in enumerate(self.full_data) if d["length"] == length]
        self.prompts = [d["prompt"] for d in self.data]
        self.obj_pos = [d["position"] for d in self.data]
        self.input_ids = [torch.tensor(d["input_ids"]) for d in self.data]
        # target1 = [torch.tensor(model.to_tokens(d["true"], prepend_bos=False)) for d in self.data]
        # target2 = [torch.tensor(model.to_tokens(d["false"], prepend_bos=False)) for d in self.data]
        target1 = torch.tensor([d["token_true"] for d in self.data])
        target2 = torch.tensor([d["token_false"] for d in self.data])
        # target1, target2 = [torch.zeros(10)], [torch.zeros(10)]
        self.target = torch.stack([target1, target2], dim=1)

        
    def slice(self, end, start=0):
        self.data   = self.data[start:end]
        self.target = self.target[start:end]
        self.clean_prompts = self.clean_prompts[start:end]
        self.corrupted_prompts = self.corrupted_prompts[start:end]
        self.obj_pos = self.obj_pos[start:end]
        
    def get_lengths(self):
        # return all the possible lengths in the dataset
        for d in tqdm(self.full_data, desc="Tokenizing prompts"):
            tokenized_prompt = self.tokenizer([d["prompt"], d["target_true"], d["target_new"]], return_length=True)
            d["length"] = tokenized_prompt["length"][0]
            # find the position of d["false"] in the tokenized prompt

            assert len(tokenized_prompt["input_ids"][2]) < 3, "False token is too long"
            
            if len(tokenized_prompt["input_ids"][2]) == 2:
                token_position = 1
            if len(tokenized_prompt["input_ids"][2]) == 1:
                token_position = 0
                
            d["position"] = tokenized_prompt["input_ids"][0].index(tokenized_prompt["input_ids"][2][token_position])
            d["token_true"] = tokenized_prompt["input_ids"][1][token_position]
            d["token_false"] = tokenized_prompt["input_ids"][2][token_position]
            d["input_ids"] = tokenized_prompt["input_ids"][token_position]
        return list(set([d["length"] for d in self.full_data]))
    
    def slice_to_fit_batch(self, batch_size):
        maxdatadize = (len(self.data)//batch_size)*batch_size
        self.slice(maxdatadize)
        
    def save_filtered(self):
        self.data_per_len[self.length] = self.data
        
        
from transformers import AutoTokenizer, AutoModelForCausalLM

class EvaluateMechanism:
    def __init__(self, model_name:str, dataset:MyDataset, device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model = self.model.to(device)
        self.model_name = model_name
        self.dataset = dataset
        self.lenghts = self.dataset.lenghts
        self.device = device
        print("Model device", self.model.device)
        
    def check_prediction(self, logit, target):
        probs = torch.softmax(logit, dim=-1)[:,-1,:]
        # count the number of times the model predicts the target[:, 0] or target[:, 1]
        num_samples = target.shape[0]
        target_true = 0
        target_false = 0
        other = 0
        target_true_indices = []
        target_false_indices = []
        other_indices = []
        for i in range(num_samples):
            if torch.argmax(probs[i]) == target[i, 0]:
                target_true += 1
                target_true_indices.append(i)
            elif torch.argmax(probs[i]) == target[i, 1]:
                target_false += 1
                target_false_indices.append(i)
            else:
                other += 1
                other_indices.append(i)
        return  target_true_indices, target_false_indices, other_indices
    
    def evaluate(self, length):
        self.dataset.set_len(length)
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=100, shuffle=True)
        batch_size = 100
        target_true, target_false, other = 0, 0, 0
        n_batch = len(dataloader)
        all_true_indices = []
        all_false_indices = []
        all_other_indices = []
        
        for idx, batch in tqdm(enumerate(dataloader), total=n_batch):
            input_ids = batch["input_ids"].to(self.device)
            logits = self.model(input_ids)["logits"]
            count = self.check_prediction(logits, batch["target"])
            target_true += len(count[0])
            target_false += len(count[1])
            other += len(count[2])

            all_true_indices.extend([self.dataset.original_index[i+idx*batch_size] for i in count[0]])
            all_false_indices.extend([self.dataset.original_index[i+idx*batch_size] for i in count[1]])
            all_other_indices.extend([self.dataset.original_index[i+idx*batch_size] for i in count[2]])

        return target_true, target_false, other, all_true_indices, all_false_indices, all_other_indices
    
    def evaluate_all(self):
        target_true, target_false, other = 0, 0, 0
        all_true_indices = []
        all_false_indices = []
        all_other_indices = []
        for length in self.lenghts:
            result = self.evaluate(length)
            target_true += result[0]
            target_false += result[1]
            other += result[2]
            
            #assert duplicates
            all_index = result[3] + result[4] + result[5]
            assert len(all_index) == len(set(all_index)), "Duplicates in the indices"
            
            all_true_indices.extend(result[3])
            all_false_indices.extend(result[4])
            all_other_indices.extend(result[5])
            
        print(f"Total: Target True: {target_true}, Target False: {target_false}, Other: {other}")
        # index = torch.cat(index, dim=1)
        
        if len(self.model_name.split("/")) > 1:
            save_name = self.model_name.split("/")[1]
        
        #save results
        with open(f"../results/{save_name}_evaluate_mechanism.json", "w") as file:
            json.dump({"target_true": target_true, "target_false": target_false, "other": other, "dataset_len":len(self.dataset.full_data)}, file)
        # torch.save(index, f"../results/{self.model_name}_evaluate_mechanism.pt")
        
        # save indices
        with open(f"../results/{save_name}_evaluate_mechanism_indices.json", "w") as file:
            json.dump({"target_true": all_true_indices, "target_false": all_false_indices, "other": all_other_indices}, file)
        
        return target_true, target_false, other
