import einops
import torch
from torch.utils.data import Dataset
import json
from tqdm import tqdm

class MyDataset(Dataset):
    def __init__(self, path, model, slice=None):
        self.data = json.load(open(path))
        if slice is not None:
            self.data = self.data[:slice]
        print("Dataset loaded from", path)
        print("Number of samples:", len(self.data))
        self.pad_token = model.tokenizer.pad_token
        self.data_per_len = self.split_per_len(self.data)
        
    def __len__(self):
        return len(self.data) 
    def __getitem__(self, idx):
        return {
            "clean_prompts": self.clean_prompts[idx],
            "corrupted_prompts": self.corrupted_prompts[idx],
            "target": self.target[idx],
            "obj_pos": self.obj_pos[idx],
        }
        
    def split_per_len(self, data):
        data_per_len = {}
        for sample in data:
            length = sample["length"]
            if length not in data_per_len:
                data_per_len[length] = []
            data_per_len[length].append(sample)
        return data_per_len
    
    def set_len(self, length, model):
        self.length = length
        self.data = self.data_per_len[length]
        self.clean_prompts = [d["template"].format(self.pad_token) for d in self.data]
        self.corrupted_prompts = [d["template"].format(d["target_new"]) for d in self.data]
        self.obj_pos = [d["obj_pos"] for d in self.data]
        target1 = [model.to_tokens(d["target_true"], prepend_bos=False) for d in self.data]
        target2 = [model.to_tokens(d["target_new"], prepend_bos=False) for d in self.data]
        tensor_1 = torch.stack(target1, dim=0)
        tensor_2 = torch.stack(target2, dim=0)
        # stack the tensors
        self.target = torch.stack([tensor_1, tensor_2], dim=1).squeeze()
        
    def filter_from_idx(self, index, exclude=False, save_filtered=False):
        if exclude:
            self.target = [self.target[i] for i in range(len(self.target)) if i not in index]
            self.clean_prompts = [self.clean_prompts[i] for i in range(len(self.clean_prompts)) if i not in index]
            self.corrupted_prompts = [self.corrupted_prompts[i] for i in range(len(self.corrupted_prompts)) if i not in index]
            self.obj_pos = [self.obj_pos[i] for i in range(len(self.obj_pos)) if i not in index]
            self.data = [self.data[i] for i in range(len(self.data)) if i not in index]
        else:
            self.target = [self.target[i] for i in index]
            self.clean_prompts = [self.clean_prompts[i] for i in index]
            self.corrupted_prompts = [self.corrupted_prompts[i] for i in index]
            self.obj_pos = [self.obj_pos[i] for i in index]
            self.data = [self.data[i] for i in index]
            
        if save_filtered:
            self.save_filtered()
    
    def slice(self, end, start=0):
        self.data   = self.data[start:end]
        self.target = self.target[start:end]
        self.clean_prompts = self.clean_prompts[start:end]
        self.corrupted_prompts = self.corrupted_prompts[start:end]
        self.obj_pos = self.obj_pos[start:end]
        
    def get_lengths(self):
        return list(self.data_per_len.keys())
    
    def slice_to_fit_batch(self, batch_size):
        maxdatadize = (len(self.data)//batch_size)*batch_size
        self.slice(maxdatadize)
        
    def save_filtered(self):
        self.data_per_len[self.length] = self.data
       

    
class DatasetGenerator():
    def __init__(self, path):
        self.data = json.load(open(path))
    
    def generate_dataset(self, model, lenghts=[17,19,23]):
    
        my_data = []
        for i,d in tqdm(enumerate(self.data), total=len(self.data), desc="Generating dataset"):
            target_new = " " + d["requested_rewrite"]["target_true"]["str"]
            target_true = " " + d["requested_rewrite"]["target_new"]["str"]
            if i % 50 == 0:
                unique_strs = set(json.dumps(d) for d in my_data)
                my_data = [json.loads(s) for s in unique_strs]
                print(len(my_data))
                # if len(my_data) > 1000:
                #     break
            for p in d["attribute_prompts"]:
                template = "Redefine: " + p + "{}" + ". " + p
                #find position of {} in template
                if len(model.to_str_tokens(template.format(model.tokenizer.pad_token))) not in lenghts:
                    continue
                try:
                    obj_pos = model.to_str_tokens(template.format(model.tokenizer.pad_token)).index(".") - 1
                except:
                    continue
                if target_true in template:
                    continue
                position = template.find("{}")
                prediction = model.predict(template.format(model.tokenizer.pad_token))[1][0]
                copy_prediction = model.predict(template.format(target_new))[1][0]
                if prediction == target_true and copy_prediction == target_new:
                    my_data.append({
                        "prompt": p,
                        "template": template,
                        "prediction": prediction,
                        "copy_prediction": copy_prediction,
                        "target_true": target_true,
                        "target_new": target_new,
                        "length": len(model.to_str_tokens(template.format(model.tokenizer.pad_token))),
                        "lenght_copy": len(model.to_str_tokens(template.format(target_new))),
                        "obj_pos": obj_pos,
                    
                    })
            for p in d["neighborhood_prompts"]:
                template = "Redefine: " + p + "{}" + ". " + p
                #find position of {} in template
                if len(model.to_str_tokens(template.format(model.tokenizer.pad_token))) not in lenghts:
                    continue
                try:
                    obj_pos = model.to_str_tokens(template.format(model.tokenizer.pad_token)).index(".") - 1
                except:
                    continue
                if target_true in template:
                    continue
                position = template.find("{}")
                prediction = model.predict(template.format(model.tokenizer.pad_token))[1][0]
                copy_prediction = model.predict(template.format(target_new))[1][0]
                if prediction == target_true and copy_prediction == target_new:
                    # check if is a duplicate
                    
                    my_data.append({
                        "prompt": p,
                        "template": template,
                        "prediction": prediction,
                        "copy_prediction": copy_prediction,
                        "target_true": target_new,
                        "target_new": target_true,
                        "length": len(model.to_str_tokens(template.format(model.tokenizer.pad_token))),
                        "lenght_copy": len(model.to_str_tokens(template.format(target_new))),
                        "obj_pos": obj_pos,
                    })
                    
        print("Number of examples:", len(my_data), "Number of possible lengths:", lenghts)
        self.my_data = my_data
        
    def save(self, path):
        json.dump(self.my_data, open(path, "w"), indent=2)
