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
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.dataset import HFDataset
class EvaluateMechanism:
    def __init__(self, model_name:str, dataset:HFDataset, device="cpu", batch_size=100, orthogonalize=False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name) #, load_in_8bit=True, device_map="auto")
        self.model = self.model.to(device)
        self.model_name = model_name
        self.dataset = dataset
        self.lenghts = self.dataset.lenghts
        self.device = device
        self.batch_size = batch_size
        self.orthogonalize = orthogonalize
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
        self.dataset.set_len(length, self.orthogonalize, self.model)
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
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
        with open(f"../results/{save_name}_evaluate_mechanism.json", "w") as file:
            json.dump({"target_true": target_true, "target_false": target_false, "other": other, "dataset_len":len(self.dataset.full_data)}, file)
        # torch.save(index, f"../results/{self.model_name}_evaluate_mechanism.pt")
        
        # save indices
        with open(f"../results/{save_name}_evaluate_mechanism_indices.json", "w") as file:
            json.dump({"target_true": all_true_indices, "target_false": all_false_indices, "other": all_other_indices}, file)
        
        return target_true, target_false, other
