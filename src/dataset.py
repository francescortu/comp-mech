from click import Option
import torch
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import random
from typing import Optional,  List
from src.model import WrapHookedTransformer
from transformer_lens import HookedTransformer
from functools import partial
from transformers import  AutoModelForCausalLM 

class TlensDataset(Dataset):
    def __init__(self, path:str, model:HookedTransformer, slice:Optional[int] = None):
        self.data = json.load(open(path))
        if slice is not None:
            self.data = self.data[:slice]
        print("Dataset loaded from", path)
        print("Number of samples:", len(self.data))
        self.model = model
        self.pad_token = model.tokenizer.pad_token if model.tokenizer is not None else ValueError("You must pass a tokenizer to the model.")
        self.data_per_len = self.split_per_len()
        
    def __len__(self):
        return len(self.corrupted_prompts) 
    
    def __getitem__(self, idx):
        return {
            # "clean_prompts": self.clean_prompts[idx],
            "corrupted_prompts": self.corrupted_prompts[idx],
            "target": self.target[idx],
            "obj_pos": self.obj_pos[idx],
        }
        
    def split_per_len(self):
        for d in self.data:
            string_tokens = self.model.to_str_tokens(d["prompt"])
            d["length"] = len(string_tokens)
            for i, token in enumerate(string_tokens):
                # print(token, d["target_true"])
                if token == d["target_new"]:
                    d["obj_pos"] = i
                    break
        data_per_len = {}
        for sample in self.data:
            length = sample["length"]
            if length not in data_per_len:
                data_per_len[length] = []
            data_per_len[length].append(sample)
            
        # remove the lengths that have less than 100 samples
        # for length in list(data_per_len.keys()):
        #     if len(data_per_len[length]) < 100:
        #         del data_per_len[length]
        return data_per_len
    
    def set_len(self, length:int):
        self.length = length
        data = self.data_per_len[length]
        # self.clean_prompts = [d["template"].format(self.pad_token) for d in self.data]
        # self.corrupted_prompts = [d["template"].format(d["target_new"]) for d in self.data]
        self.corrupted_prompts = [d["prompt"] for d in data]
        self.obj_pos = [d["obj_pos"] for d in data]
        target1 = [self.model.to_tokens(d["target_true"], prepend_bos=False) for d in data]
        target2 = [self.model.to_tokens(d["target_new"], prepend_bos=False) for d in data]
        tensor_1 = torch.stack(target1, dim=0)
        tensor_2 = torch.stack(target2, dim=0)
        # stack the tensors
        self.target = torch.stack([tensor_1, tensor_2], dim=1).squeeze()
        if len(self.target.shape) < 2:
            self.target = self.target.unsqueeze(0)
        assert len(self.target.shape) == 2, "The target should be a tensor of shape (batch_size, 2)"
        assert self.target.shape[1] == 2, "The target should be a tensor of shape (batch_size, 2)"
        
    def filter_from_idx_all(self, index:List[int]):
        self.data = [self.data[i] for i in range(len(self.data)) if i in index]
        self.data_per_len = self.split_per_len()
        
    def filter_from_idx(self, index:List[int], exclude:bool=False, save_filtered:bool=False):
        if exclude:
            self.target = [self.target[i] for i in range(len(self.target)) if i not in index]
            # self.clean_prompts = [self.clean_prompts[i] for i in range(len(self.clean_prompts)) if i not in index]
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
    
    def slice(self, end:int, start:int=0):
        assert end <= len(self.corrupted_prompts), "End index is greater than the dataset size"
        self.target = self.target[start:end]
        self.corrupted_prompts = self.corrupted_prompts[start:end]
        self.obj_pos = self.obj_pos[start:end]
        
    def get_lengths(self):
        return list(self.data_per_len.keys())
    
    def slice_to_fit_batch(self, batch_size:int):
        maxdatadize = (len(self.corrupted_prompts)//batch_size)*batch_size
        self.slice(maxdatadize)
        
    def save_filtered(self):
        self.data_per_len[self.length] = self.data

class HFDataset(Dataset):
    def __init__(self, path:str, tokenizer, premise= "Redefine", slice=None):
        with open(path, 'r') as file:
            self.full_data = json.load(file)
        
        if slice is not None:
            self.full_data = self.full_data[:slice]
        # Initialize variables to avoid AttributeError before calling set_len
        self.prompts = []
        self.target = []
        self.obj_pos = []
        self.tokenizer = tokenizer
        self.premise = premise
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

    def generate_random_vector(self, embedding, similarity_range = (0,0.2):
        min_sim, max_sim = similarity_range
        random_vector = torch.randn(embedding.shape).cuda()
        random_vector = random_vector / torch.norm(random_vector)

        # Adjust the vector to fall within the desired similarity range
        cos_angle = torch.rand(1).cuda() * (max_sim - min_sim) + min_sim
        adjusted_vector = cos_angle * embedding + torch.sqrt(1 - cos_angle**2) * random_vector

        # Check if the adjusted vector falls within the similarity range
        similarity = torch.nn.functional.cosine_similarity(embedding.unsqueeze(0), adjusted_vector.unsqueeze(0), dim=1).item()
        if min_sim <= similarity <= max_sim:
            return adjusted_vector

    def compute_orthogonal(self, string_token:str, model):
        while True:
            token = self.tokenizer.encode(string_token, return_tensors="pt", add_special_tokens=True)
            if token.shape[1] > 1:
                token = token[0,1]
            else:
                token = token[0,0]
                
            token = token.to(model.device)
            with torch.no_grad():
                embedding = model.get_input_embeddings()(token)
                embedding = embedding.cuda()
            
            random_vector = torch.randn(embedding.shape[0]).cuda()
            # gram_schmidt
            random_vector = random_vector - torch.dot(random_vector, embedding) * embedding / torch.dot(embedding, embedding)
            # normalize
            # x = random_vector / torch.norm(random_vector)
            #project the embedding in the closest token
            
                # Get all embeddings from the model
            embeddings_matrix = model.get_input_embeddings().weight
            # Compute cosine similarity with all embeddings
            similarities = torch.nn.functional.cosine_similarity(random_vector.unsqueeze(0), embeddings_matrix, dim=1)
            # Find the index of the most similar token
            most_similar_token_idx = torch.argmax(similarities).item()
            # Decode the token
            most_similar_token = self.tokenizer.decode([most_similar_token_idx])

            # ############# DEBUG #############
            ort_token = self.tokenizer.encode(most_similar_token, return_tensors="pt")
            base_token = self.tokenizer.encode(string_token, return_tensors="pt")
            ort_embedding = model.get_input_embeddings()(ort_token.cuda()).cuda()
            base_embedding = model.get_input_embeddings()(base_token.cuda()).cuda()
            if len(ort_embedding.squeeze().shape) > 1:
                ort_embedding = ort_embedding.squeeze()[-1,:]
            if len(base_embedding.squeeze().shape) > 1:
                base_embedding = base_embedding.squeeze()[-1,:]
            if torch.cosine_similarity(ort_embedding.squeeze(), base_embedding.squeeze(), dim=0).item() < 0.3:
                break
            #print(most_similar_token, most_similar_token_idx)
        return most_similar_token
        
    def set_len(self, length:int, orthogonal:bool=False, model:Optional[AutoModelForCausalLM]=None):
        self.data = [d for d in self.full_data if d["length"] == length]
        self.original_index = [i for i, d in enumerate(self.full_data) if d["length"] == length]
     
        
        if orthogonal:
            assert model is not None, "You must pass a model to compute the orthogonal prompt."
            compute_orthogonal = partial(self.compute_orthogonal, model=model)
            orthogonal_target_new = [compute_orthogonal(string_token=d["target_true"]) for d in tqdm(self.data)]
            target_new = orthogonal_target_new
            token_false = []
            for tn in target_new:
                token = self.tokenizer.encode(tn, return_tensors="pt", add_special_tokens=True)
                if token.shape[1] > 1:
                    token = token[0,1]
                else:
                    token = token[0,0]
                token_false.append(token)
        else:
            target_new = [d["target_new"] for d in self.data]
            token_false = [d["token_false"] for d in self.data]
        self.prompts = []
        for d, tn in zip(self.data, target_new):
            self.prompts.append(d["template"].format(self.premise, tn))
        #     self.prompts.append(d["template"].format("Redefine", d["target_new"]))
        self.obj_pos = [d["position"] for d in self.data]
        self.input_ids = [torch.tensor(d["input_ids"]) for d in self.data]
        if orthogonal:
            for idx, _ in enumerate(self.input_ids):                    
                self.input_ids[idx][self.obj_pos[idx]] = token_false[idx]
                
        
        target1 = torch.tensor([d["token_true"] for d in self.data])
        target2 = torch.tensor(token_false)
        
        # target1 = [torch.tensor(model.to_tokens(d["true"], prepend_bos=False)) for d in self.data]
        # target2 = [torch.tensor(model.to_tokens(d["false"], prepend_bos=False)) for d in self.data]
        # target1, target2 = [torch.zeros(10)], [torch.zeros(10)]
        self.target = torch.stack([target1, target2], dim=1)

        assert self.check_duplicate() == True, "Duplicate prompts"
        assert self.check_index_mapping() == True, "Index mapping is wrong"
        assert len(self.data) == len(set(d['prompt'] for d in self.data)), "There are duplicate prompts after filtering."
        
        # check if the index mapping is unique
        assert len(self.original_index) == len(set(self.original_index)), "Original indices are not unique."
        assert all(self.full_data[idx]["prompt"] == self.data[i]["prompt"] for i, idx in enumerate(self.original_index)), "Index mapping mismatch."

        
    def check_index_mapping(self):
        # check if the index mapping is correct
        for i, d in enumerate(self.data):
            if d["prompt"] != self.full_data[self.original_index[i]]["prompt"]:
                return False
        return True
    
    def check_duplicate(self):
        # check if full_data has duplicate prompts
        #find duplicate in two lists
        seen = set()
        for i,d in enumerate(self.full_data):
            if d["prompt"] in seen:
                #check the other fields
                for j,d2 in enumerate(self.full_data):
                    if j == i:
                        continue
                    if d["prompt"] == d2["prompt"]:
                        if d["true"] != d2["true"] or d["false"] != d2["false"]:
                            continue
                        else:
                            return False
            seen.add(d["prompt"])
        return True
        
    def slice(self, end:int, start:int=0):
        self.data   = self.data[start:end]
        self.target = self.target[start:end]
        self.clean_prompts = self.clean_prompts[start:end]
        self.corrupted_prompts = self.corrupted_prompts[start:end]
        self.obj_pos = self.obj_pos[start:end]
        
    def get_lengths(self):
        # return all the possible lengths in the dataset
        for d in tqdm(self.full_data, desc="Tokenizing prompts"):
            prompt = d["template"].format(self.premise, d["target_new"])
            tokenized_prompt = self.tokenizer([prompt, d["target_true"], d["target_new"]], return_length=True)
            d["length"] = tokenized_prompt["length"][0]
            d["input_ids"] = tokenized_prompt["input_ids"][0]
            # find the position of d["false"] in the tokenized prompt

            assert len(tokenized_prompt["input_ids"][2]) < 3, "False token is too long"
            
            if len(tokenized_prompt["input_ids"][2]) == 2:
                token_position = 1
            if len(tokenized_prompt["input_ids"][2]) == 1:
                token_position = 0
                
            d["position"] = tokenized_prompt["input_ids"][0].index(tokenized_prompt["input_ids"][2][token_position])
            d["token_true"] = tokenized_prompt["input_ids"][1][token_position] 
            d["token_false"] = tokenized_prompt["input_ids"][2][token_position]
        return list(set([d["length"] for d in self.full_data]))
    
    def slice_to_fit_batch(self, batch_size):
        maxdatadize = (len(self.data)//batch_size)*batch_size
        self.slice(maxdatadize)
        
    def save_filtered(self):
        self.data_per_len[self.length] = self.data

       
class SampleDataset:
    def __init__(self, path:str, model, save_path:str, tokenizer:Optional[object]):
        self.data = json.load(open(path))
        self.model = model
        self.save_path = save_path
        if type(model) == WrapHookedTransformer:
            self.model_type = "WrapHookedTransformer"
            self.tokenizer = model.tokenizer  
        else:
            self.model_type = "AutoModelForCausalLM"
            try:
                self.tokenizer = tokenizer
            except:  
                raise ValueError("With HuggingFace models, you must pass a tokenizer")

    def sample(self, size:int=10000):
        if type(self.model) == WrapHookedTransformer:
            self.sample_dataset_tlens(size)
        else:
            self.sample_dataset_hf(size)
    
    def sample_dataset_tlens(self, size:int):
        random.seed(42)
        new_data = []
        random.shuffle(self.data)
        with tqdm(total=size) as pbar:
            for i,d in enumerate(self.data):

                # empty_prompt = d["template"].format("Redefine", self.model.tokenizer.pad_token)
                empty_prompt = d["base_prompt"]
                if self.model.predict(empty_prompt)[1][0] == d["target_true"]:
                    new_data.append(d)
                    if len(new_data) == size:
                        break
                pbar.update(len(new_data)-pbar.n)
            self.data = new_data
            
    def sample_dataset_hf(self, size:int):
        random.seed(42)
        new_data = []
        random.shuffle(self.data)
        with tqdm(total=size) as pbar:
            for i,d in enumerate(self.data):
                empty_prompt = d["base_prompt"]
                #encode the prompt
                input_ids = self.tokenizer.encode(empty_prompt, return_tensors="pt")
                input_ids = input_ids.to(self.model.device)
                target_true = self.tokenizer.encode(d["target_true"], return_tensors="pt", add_special_tokens=False)
                #predict the next token
                logits = self.model(input_ids)["logits"][0, -1, :].cpu()
                #get the index of the predicted token
                index = logits.argmax()
                # check if the predicted token is the target

                if index in target_true:
                    new_data.append(d)
                    if len(new_data) == size:
                        break
                pbar.update(len(new_data)-pbar.n)
            self.data = new_data
    
    def save(self):
        json.dump(self.data, open(self.save_path, "w"), indent=2)
    
    
    
    
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
