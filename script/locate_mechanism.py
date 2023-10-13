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

@dataclass
class Config:
    num_samples: int = 10
    batch_size: int = 5
    mem_win_noise_position = [1,2,3,9,10,11]
    mem_win_noise_mlt = 1.4
    cp_win_noise_position = [1,2,3,8,9,10,11]
    cp_win_noise_mlt = 0.8
    name_save_file = "prova_gpt2"
    name_dataset = "dataset_gpt2.json"
    max_len = 15
    keys_to_compute = [
        # "logit_lens_mem",
        # "logit_lens_cp",
        # "resid_pos",
        # "attn_head_out",
        # "attn_head_by_pos",
        # "per_block",
        "mlp_out",
        "attn_out_by_pos"
    ]
    interval:int = 1
config = Config()



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



def load_model_and_data():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = WrapHookedTransformer.from_pretrained("gpt2", device=DEVICE)
    dataset = Dataset(f"../data/{config.name_dataset}")
    dataset.filter(filter_key="cp", filter_interval=(0.2, 0.25))
    dataset.filter(filter_key="mem", filter_interval=(0.2, 0.25))
    dataset.random_sample(config.num_samples, config.max_len)
    dataset.compute_noise_level(model)
    return model, dataset


def process_batch(batch, model, dataset):
    pos_batch = batch["pos_dataset"]
    neg_batch = batch["neg_dataset"]
    pos_target_ids = {
        "target": model.to_tokens(pos_batch["target"], prepend_bos=False),
        "orthogonal": model.to_tokens(pos_batch["orthogonal_token"], prepend_bos=False),
    }
    neg_target_ids = {
        "target": model.to_tokens(neg_batch["target"], prepend_bos=False),
        "orthogonal": model.to_tokens(neg_batch["orthogonal_token"], prepend_bos=False),
    }
    pos_input_ids = model.to_tokens(pos_batch["premise"], prepend_bos=True)
    neg_input_ids = model.to_tokens(neg_batch["premise"], prepend_bos=True)
    pos_embs_corrupted = dataset.add_noise(
        model,
        pos_batch["premise"],
        noise_index = torch.tensor(config.mem_win_noise_position),
        target_win=8,
        noise_mlt=config.mem_win_noise_mlt
    )
    neg_embs_corrupted = dataset.add_noise(
        model,
        neg_batch["premise"],
        noise_index = torch.tensor(config.cp_win_noise_position),
        target_win=8,
        noise_mlt=config.cp_win_noise_mlt
    )

    pos_corrupted_logit, pos_corrupted_cache = model.run_with_cache_from_embed(pos_embs_corrupted)
    pos_clean_logit, pos_clean_cache = model.run_with_cache(pos_batch["premise"])
    neg_corrupted_logit, neg_corrupted_cache = model.run_with_cache_from_embed(neg_embs_corrupted)
    neg_clean_logit, neg_clean_cache = model.run_with_cache(neg_batch["premise"])
    
    if torch.equal(pos_clean_logit, pos_corrupted_logit):
        raise ValueError("Logits are the same, no corruption happened")
    
    def check_reversed_probs( corrupted_logits, target_pos, orthogonal_pos):
        corrupted_logits = torch.softmax(corrupted_logits, dim=-1)
        target_probs = corrupted_logits[:,-1,:].gather(-1, index=target_pos).squeeze(-1)
        orthogonal_probs = corrupted_logits[:,-1,:].gather(-1, index=orthogonal_pos).squeeze(-1)
        return (target_probs - orthogonal_probs).mean()

    print("Traget - Orthogonal", check_reversed_probs( pos_corrupted_logit,  pos_target_ids["target"], pos_target_ids["orthogonal"],))
    print("Target - orthogonal", check_reversed_probs( neg_corrupted_logit, neg_target_ids["target"], neg_target_ids["orthogonal"]))
    
    

    def pos_metric(logits):
        improved = indirect_effect(
            logits=logits,
            corrupted_logits=pos_corrupted_logit,
            first_ids_pos=pos_target_ids["target"]
        )
        # improved = improved/POS_BASELINE
        return improved
        
    def neg_metric(logits):
        improved = indirect_effect(
            logits=logits,
            corrupted_logits=neg_corrupted_logit,
            first_ids_pos=neg_target_ids["orthogonal"]
        )
        return improved
    
    

    
    print("pos metric", pos_metric(logits=pos_clean_logit))
    print("neg metric", neg_metric(logits=neg_clean_logit))
    
    
    shared_args = {
        "model": model,
        "input_ids": pos_input_ids,
        "clean_cache": pos_clean_cache,
        "metric": pos_metric,
        "embs_corrupted": pos_embs_corrupted,
        "interval": config.interval,
        "target_ids": pos_target_ids,
    }
    pos_result = construct_result_dict(shared_args, config.keys_to_compute)
    pos_result["example_str_tokens"] = model.to_str_tokens(pos_batch["premise"][0])
    
    pos_clean_probs = torch.softmax(pos_clean_logit, dim=-1)[:,-1,:]
    pos_corrupted_logit = torch.softmax(pos_corrupted_logit, dim=-1)[:,-1,:]
    pos_target_probs_clean = pos_clean_probs.gather(-1, index=pos_target_ids["target"]).squeeze(-1)
    pos_target_probs_corrupted = pos_corrupted_logit.gather(-1, index=pos_target_ids["target"]).squeeze(-1)
    pos_orthogonal_probs_clean = pos_clean_probs.gather(-1, index=pos_target_ids["orthogonal"]).squeeze(-1)
    pos_orthogonal_probs_corrupted = pos_corrupted_logit.gather(-1, index=pos_target_ids["orthogonal"]).squeeze(-1)
    
    pos_result["clean_logit_mem"] = pos_target_probs_clean.cpu()
    pos_result["corrupted_logit_mem"] = pos_target_probs_corrupted.cpu()
    pos_result["clean_logit_cp"] = pos_orthogonal_probs_clean.cpu()
    pos_result["corrupted_logit_cp"] = pos_orthogonal_probs_corrupted.cpu()


    
    shared_args = {
        "model": model,
        "input_ids": neg_input_ids,
        "clean_cache": neg_clean_cache,
        "metric": neg_metric,
        "embs_corrupted": neg_embs_corrupted,
        "interval": config.interval,
        "target_ids": neg_target_ids,
    }
    
    neg_result = construct_result_dict(shared_args, config.keys_to_compute)
    neg_result["example_str_tokens"] = model.to_str_tokens(neg_batch["premise"][0])

    neg_clean_probs = torch.softmax(neg_clean_logit, dim=-1)[:,-1,:]
    neg_corrputed_probs = torch.softmax(neg_corrupted_logit, dim=-1)[:,-1,:]
    neg_target_probs_clean = neg_clean_probs.gather(-1, index=neg_target_ids["target"]).squeeze(-1)
    neg_target_probs_corrupted = neg_corrputed_probs.gather(-1, index=neg_target_ids["target"]).squeeze(-1)
    neg_orthogonal_probs_clean = neg_clean_probs.gather(-1, index=neg_target_ids["orthogonal"]).squeeze(-1)
    neg_orthogonal_probs_corrupted = neg_corrputed_probs.gather(-1, index=neg_target_ids["orthogonal"]).squeeze(-1)
    
    neg_result["clean_logit_mem"] = neg_target_probs_clean.cpu()
    neg_result["corrupted_logit_mem"] = neg_target_probs_corrupted.cpu()
    neg_result["clean_logit_cp"] = neg_orthogonal_probs_clean.cpu()
    neg_result["corrupted_logit_cp"] = neg_orthogonal_probs_corrupted.cpu()

    return pos_result, neg_result

def main():
    torch.set_grad_enabled(False)
    model, dataset = load_model_and_data()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    
    pos_results, neg_results = [], []
    for batch in dataloader:
        pos_result, neg_result = process_batch(batch, model, dataset)
        pos_results.append(pos_result)
        neg_results.append(neg_result)

    full_result = {
        "pos": list_of_dicts_to_dict_of_lists(pos_results),
        "neg": list_of_dicts_to_dict_of_lists(neg_results)
    }
    for key in full_result.keys():
        for subkey in full_result[key].keys():
            if subkey not in ["clean_logit_mem", "corrupted_logit_mem", "clean_logit_cp", "corrupted_logit_cp", "example_str_tokens"]:
                full_result[key][subkey] = {k: [d[k] for d in full_result[key][subkey]] for k in full_result[key][subkey][0].keys()}
                for subsubkey in full_result[key][subkey].keys():
                    if subsubkey in ['patched_logits_mem', 'patched_logits_cp', 'full_delta']:
                        # here we have list of tensor of shape (component, component, batch)
                        full_result[key][subkey][subsubkey] = torch.cat(full_result[key][subkey][subsubkey], dim=-1)
                        full_result[key][subkey][subsubkey] = einops.rearrange(full_result[key][subkey][subsubkey], 'c1 c2 b -> b c1 c2')
                    else:
                        #here we have list of tensor of shape (component, component)
                        full_result[key][subkey][subsubkey] = torch.stack(full_result[key][subkey][subsubkey])
            
        full_result[key]["clean_logit_mem"] = torch.cat(full_result[key]["clean_logit_mem"], dim=0)
        full_result[key]["corrupted_logit_mem"] = torch.cat(full_result[key]["corrupted_logit_mem"], dim=0)
        full_result[key]["clean_logit_cp"] = torch.cat(full_result[key]["clean_logit_cp"], dim=0)
        full_result[key]["corrupted_logit_cp"] = torch.cat(full_result[key]["corrupted_logit_cp"], dim=0)

    
    torch.save(full_result, f"../results/locate_mechanism/{config.name_save_file}_full_result.pt")

if __name__ == "__main__":
    main()



