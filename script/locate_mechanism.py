import sys
import torch
import json
from dataclasses import dataclass
import einops

# Add paths
for path in ['..', '../src', '../data']:
    sys.path.append(path)

from transformer_lens import HookedTransformer
from src.model import WrapHookedTransformer
from src.dataset import Dataset
from src.locate_mechanism import construct_result_dict, indirect_effect
from src.utils import list_of_dicts_to_dict_of_lists
import argparse 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--interval", type=int, default=1)
    return parser.parse_args()

@dataclass
class Config:
    num_samples: int
    batch_size: int
    model_name:str
    name_dataset:str
    name_save_file:str
    mem_win_noise_position = [1,2,3,9,10,11]
    mem_win_noise_mlt = 1.4
    cp_win_noise_position = [1,2,3,8,9,10,11]
    cp_win_noise_mlt = 1.4
    max_len = 15
    keys_to_compute = [
        "logit_lens_mem",
        "logit_lens_cp",
        "resid_pos",
        "attn_head_out",
        # "attn_head_by_pos", !WARNING! this is not implemented
        # "per_block", !WARNING! this is not implemented 
        "mlp_out",
        "attn_out_by_pos"
    ]
    filter_interval = (0,1)
    interval:int = 10
    @classmethod
    def from_args(cls, args):
        return cls(
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            interval=args.interval,
            model_name=args.model,
            name_save_file=f"1000{args.model}_full_result" if args.interval == 1 else f"{args.model}_full_result_{args.interval}",
            name_dataset = f"dataset_{args.model}.json",
        )



def dict_of_lists_to_dict_of_tensors(dict_of_lists):
    dict_of_tensors = {}
    for key, tensor_list in dict_of_lists.items():
        # If the key is "example_str_token", keep it as a list of strings
        if key == "example_str_tokens":
            dict_of_tensors[key] = tensor_list
            continue
        
        # Check if the first element of the list is a tensor
        if isinstance(tensor_list[0], torch.Tensor):
            dict_of_tensors[key] = torch.stack(tensor_list)
        # If the first element is a list, convert each inner list to a tensor and then stack
        elif isinstance(tensor_list[0], list):
            tensor_list = [torch.tensor(item) for item in tensor_list]
            dict_of_tensors[key] = torch.stack(tensor_list)
        else:
            print(f"Unsupported data type for key {key}: {type(tensor_list[0])}")
            raise ValueError(f"Unsupported data type for key {key}: {type(tensor_list[0])}")
    return dict_of_tensors

torch.set_grad_enabled(False)



def load_model_and_data(config):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = WrapHookedTransformer.from_pretrained(config.model_name, device=DEVICE)
    dataset = Dataset(f"../data/{config.name_dataset}")
    dataset.filter(filter_key="cp", filter_interval=config.filter_interval)
    dataset.filter(filter_key="mem", filter_interval=config.filter_interval)
    dataset.random_sample(config.num_samples, config.max_len)
    dataset.compute_noise_level(model)
    return model, dataset


def process_batch(batch, model, dataset, config):
    mem_batch = batch["mem_dataset"]
    cp_batch = batch["cp_dataset"]
    mem_target_ids = {
        "mem_token": model.to_tokens(mem_batch["target"], prepend_bos=False),
        "cp_token": model.to_tokens(mem_batch["orthogonal_token"], prepend_bos=False),
    }
    cp_target_ids = {
        "mem_token": model.to_tokens(cp_batch["target"], prepend_bos=False),
        "cp_token": model.to_tokens(cp_batch["orthogonal_token"], prepend_bos=False),
    }
    mem_input_ids = model.to_tokens(mem_batch["premise"], prepend_bos=True)
    cp_input_ids = model.to_tokens(cp_batch["premise"], prepend_bos=True)
    mem_embs_corrupted = dataset.add_noise(
        model,
        mem_batch["premise"],
        noise_index = torch.tensor(config.mem_win_noise_position),
        target_win=8,
        noise_mlt=config.mem_win_noise_mlt
    )
    cp_embs_corrupted = dataset.add_noise(
        model,
        cp_batch["premise"],
        noise_index = torch.tensor(config.cp_win_noise_position),
        target_win=8,
        noise_mlt=config.cp_win_noise_mlt
    )

    mem_corrupted_logit, mem_corrupted_cache = model.run_with_cache_from_embed(mem_embs_corrupted)
    mem_clean_logit, mem_clean_cache = model.run_with_cache(mem_batch["premise"])
    cp_corrupted_logit, cp_corrupted_cache = model.run_with_cache_from_embed(cp_embs_corrupted)
    cp_clean_logit, cp_clean_cache = model.run_with_cache(cp_batch["premise"])
    
    if torch.equal(mem_clean_logit, mem_corrupted_logit):
        raise ValueError("Logits are the same, no corruption happened")
    
    def check_reversed_probs( corrupted_logits, target_pos, orthogonal_pos):
        corrupted_logits = torch.softmax(corrupted_logits, dim=-1)
        print("corrupted_logits", corrupted_logits[:,-1,:].shape, target_pos.shape)
        target_probs = corrupted_logits[:,-1,:].gather(-1, index=target_pos).squeeze(-1)
        orthogonal_probs = corrupted_logits[:,-1,:].gather(-1, index=orthogonal_pos).squeeze(-1)
        return (target_probs - orthogonal_probs).mean()

    print("Traget - Orthogonal", check_reversed_probs( mem_corrupted_logit,  mem_target_ids["mem_token"], mem_target_ids["cp_token"],))
    print("Target - orthogonal", check_reversed_probs( cp_corrupted_logit, cp_target_ids["mem_token"], cp_target_ids["cp_token"]))
    
    

    def mem_metric(logits):
        improved = indirect_effect(
            logits=logits,
            corrupted_logits=mem_corrupted_logit,
            first_ids_pos=mem_target_ids["mem_token"],
            clean_logits=mem_clean_logit,
        )
        # improved = improved/POS_BASELINE
        return improved
        
    def cp_metric(logits):
        improved = indirect_effect(
            logits=logits,
            corrupted_logits=cp_corrupted_logit,
            first_ids_pos=cp_target_ids["cp_token"],
            clean_logits=cp_clean_logit,
        )
        return improved
    
    

    
    print("mem metric", mem_metric(logits=mem_clean_logit))
    print("cp metric", cp_metric(logits=cp_clean_logit))
    
    
    shared_args = {
        "model": model,
        "input_ids": mem_input_ids,
        "clean_cache": mem_clean_cache,
        "metric": mem_metric,
        "embs_corrupted": mem_embs_corrupted,
        "interval": config.interval,
        "target_ids": mem_target_ids,
    }
    mem_result = construct_result_dict(shared_args, config.keys_to_compute)
    mem_result["example_str_tokens"] = model.to_str_tokens(mem_batch["premise"][0])
    
    mem_clean_probs = torch.softmax(mem_clean_logit, dim=-1)[:,-1,:]
    mem_corrupted_logit = torch.softmax(mem_corrupted_logit, dim=-1)[:,-1,:]
    
    
    mem_clean_probs_mem_token = mem_clean_probs.gather(-1, index=mem_target_ids["mem_token"]).squeeze(-1)
    mem_corrupted_probs_mem_token = mem_corrupted_logit.gather(-1, index=mem_target_ids["mem_token"]).squeeze(-1)
    
    mem_clean_probs_orthogonal_token = mem_clean_probs.gather(-1, index=mem_target_ids["cp_token"]).squeeze(-1)
    mem_corrupted_probs_orthogonal_token = mem_corrupted_logit.gather(-1, index=mem_target_ids["cp_token"]).squeeze(-1)
    
    mem_result["clean_logit_mem"] = mem_clean_probs_mem_token.cpu()
    mem_result["corrupted_logit_mem"] = mem_corrupted_probs_mem_token.cpu()
    mem_result["clean_logit_cp"] = mem_clean_probs_orthogonal_token.cpu()
    mem_result["corrupted_logit_cp"] = mem_corrupted_probs_orthogonal_token.cpu()
    mem_result["premise"] = mem_batch["premise"]

    
    shared_args = {
        "model": model,
        "input_ids": cp_input_ids,
        "clean_cache": cp_clean_cache,
        "metric": cp_metric,
        "embs_corrupted": cp_embs_corrupted,
        "interval": config.interval,
        "target_ids": cp_target_ids,
    }
    
    cp_result = construct_result_dict(shared_args, config.keys_to_compute)
    cp_result["example_str_tokens"] = model.to_str_tokens(cp_batch["premise"][0])

    cp_clean_probs = torch.softmax(cp_clean_logit, dim=-1)[:,-1,:]
    cp_corrputed_probs = torch.softmax(cp_corrupted_logit, dim=-1)[:,-1,:]
    
    cp_clean_probs_mem_token = cp_clean_probs.gather(-1, index=cp_target_ids["mem_token"]).squeeze(-1)
    cp_corrupted_probs_mem_token = cp_corrputed_probs.gather(-1, index=cp_target_ids["mem_token"]).squeeze(-1)
    cp_clean_probs_orthogonal_token = cp_clean_probs.gather(-1, index=cp_target_ids["cp_token"]).squeeze(-1)
    cp_corrupted_probs_orthogonal_token = cp_corrputed_probs.gather(-1, index=cp_target_ids["cp_token"]).squeeze(-1)
    
    cp_result["clean_logit_mem"] = cp_clean_probs_mem_token.cpu()
    cp_result["corrupted_logit_mem"] = cp_corrupted_probs_mem_token.cpu()
    cp_result["clean_logit_cp"] = cp_clean_probs_orthogonal_token.cpu()
    cp_result["corrupted_logit_cp"] = cp_corrupted_probs_orthogonal_token.cpu()
    cp_result["premise"] = cp_batch["premise"]

    return mem_result, cp_result

def main():
    args = get_args()
    config = Config.from_args(args)

    print("Starting with config", config)
    torch.set_grad_enabled(False)
    model, dataset = load_model_and_data(config)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    
    mem_results, cp_results = [], []
    for batch in dataloader:
        mem_result, cp_result = process_batch(batch, model, dataset, config)
        mem_results.append(mem_result)
        cp_results.append(cp_result)

    full_result = {
        "mem": list_of_dicts_to_dict_of_lists(mem_results),
        "cp": list_of_dicts_to_dict_of_lists(cp_results)
    }
    for key in full_result.keys():
        for subkey in full_result[key].keys():
            if subkey in ["resid_pos", "attn_head_out", "attn_head_by_pos", "per_block", "mlp_out", "attn_out_by_pos"]:
                full_result[key][subkey] = {k: [d[k] for d in full_result[key][subkey]] for k in full_result[key][subkey][0].keys()}
                for subsubkey in full_result[key][subkey].keys():
                    if subsubkey in ['patched_logits_mem', 'patched_logits_cp', 'full_delta']:
                        # here we have list of tensor of shape (component, component, batch)
                        full_result[key][subkey][subsubkey] = torch.cat(full_result[key][subkey][subsubkey], dim=-1)
                        full_result[key][subkey][subsubkey] = einops.rearrange(full_result[key][subkey][subsubkey], 'c1 c2 b -> b c1 c2')
                    else:
                        #here we have list of tensor of shape (component, component)
                        full_result[key][subkey][subsubkey] = torch.stack(full_result[key][subkey][subsubkey])
            if subkey in ["clean_logit_mem", "corrupted_logit_mem", "clean_logit_cp", "corrupted_logit_cp", "logit_lens_mem", "logit_lens_cp"]:
                full_result[key][subkey] = torch.cat(full_result[key][subkey], dim=0)
       
    
    torch.save(full_result, f"../results/locate_mechanism/{config.name_save_file}.pt")

if __name__ == "__main__":
    main()



