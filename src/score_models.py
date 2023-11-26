import sys

from matplotlib.transforms import interval_contains
sys.path.append('..')
sys.path.append('../src')
sys.path.append('../data')

import json
from tqdm import tqdm
import os
from typing import Optional

import json
import torch
from torch.utils.data import Dataset, DataLoader
import einops
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.dataset import HFDataset
from dataclasses import dataclass






class EvaluateMechanism:
    def __init__(self, model_name:str, dataset:HFDataset, device="cpu", batch_size=100, orthogonalize=False, premise="Redefine", interval=None, family_name:Optional[str]=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name) #, load_in_8bit=True, device_map="auto")
        self.model = self.model.to(device)
        self.model_name = model_name
        self.dataset = dataset
        self.lenghts = self.dataset.lenghts
        self.device = device
        self.batch_size = batch_size
        self.orthogonalize = orthogonalize
        self.premise = premise
        self.family_name = family_name
        self.interval = interval
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
        self.dataset.set_len(length, orthogonal = self.orthogonalize, model = self.model)
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
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

            all_true_indices.extend([self.dataset.original_index[i+idx*self.batch_size] for i in count[0]])
            all_false_indices.extend([self.dataset.original_index[i+idx*self.batch_size] for i in count[1]])
            all_other_indices.extend([self.dataset.original_index[i+idx*self.batch_size] for i in count[2]])

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
        else:
            save_name = self.model_name
        if self.orthogonalize:
            save_name += "orth"
        #save results
        
        filename = f"results/{self.family_name}_evaluate_mechanism.csv"
        #if file not exists, create it and write the header
        if not os.path.isfile(filename):
            with open(filename, "w") as file:
                file.write("model_name,orthogonalize,premise,interval,target_true,target_false,other\n")
        
        with open(filename, "a+") as file:
            file.seek(0)
            # if there is aleardy a line with the same model_name and orthogonalize, delete it
            lines = file.readlines()
            # Check if a line with the same model_name and orthogonalize exists
            line_exists = any(line.split(",")[0] == self.model_name and line.split(",")[1] == str(self.orthogonalize) and line.split(",")[2] == self.premise and line.split(",")[3] == self.interval for line in lines)

            # If the line exists, remove it
            if line_exists:
                lines = [line for line in lines if not (line.split(",")[0] == self.model_name and line.split(",")[1] == str(self.orthogonalize and line.split(",")[2] == self.premise and line.split(",")[3] == self.interval))]

                # Rewrite the file without the removed line
                file.seek(0)  # Move the file pointer to the start of the file
                file.truncate()  # Truncate the file (i.e., remove all content)
                file.writelines(lines)  # Write the updated lines back to the file
            file.write(f"{self.model_name},{self.orthogonalize},{self.premise},{self.interval},{target_true},{target_false},{other}\n")
        
        
    
        
        # save indices
        if not os.path.isdir(f"results/{self.family_name}_evaluate_mechs_indices"):
            # if the directory does not exist, create it
            os.makedirs(f"results/{self.family_name}_evaluate_mechs_indices")
        
        with open(f"results/{self.family_name}_evaluate_mechs_indices/{save_name}_evaluate_mechanism_indices.json", "w") as file:
            json.dump({"target_true": all_true_indices, "target_false": all_false_indices, "other": all_other_indices}, file)
        
        return target_true, target_false, other
