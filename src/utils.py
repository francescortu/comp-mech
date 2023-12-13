import torch
import warnings
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.table import Table
import time
import os
import torch.nn.functional as F


def check_dataset_and_sample(dataset_path, model_name):
    if os.path.exists(dataset_path):
        return 
    else:
        from transformers import AutoModelForCausalLM
        from transformers import AutoTokenizer
        print("Dataset not found, creating it:")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.eval()
        from src.dataset import SampleDataset
        sampler = SampleDataset(
            "../data/full_data.json",
            model=model,
            save_path=dataset_path,
            tokenizer=tokenizer,
        )
        sampler.sample()
        sampler.save()
        del model
        del sampler
        torch.cuda.empty_cache()
        return 


def display_experiments(experiments, status):
    table = Table(show_header=True, header_style="bold magenta", expand=True)
    table.add_column("Experiment", style="dim", width=3)
    table.add_column("Status", width=1)
    table.add_column("Live Output")  # New column for future live output
    for experiment, stat in zip(experiments, status):
        table.add_row(experiment.__name__, stat, "")  # Empty string for future live output
    return table


def display_config(config):
    config_items = [
        Text.assemble(("Model Name: ", "bold"), str(config.model_name)),
        Text.assemble(("Batch Size: ", "bold"), str(config.batch_size)),
        Text.assemble(("Dataset Path: ", "bold"), config.dataset_path),
        Text.assemble(("Dataset Slice: ", "bold"), str(config.dataset_slice)),
        Text.assemble(("Produce Plots: ", "bold"), str(config.produce_plots)),
        Text.assemble(("Normalize Logit: ", "bold"), str(config.normalize_logit)),
        Text.assemble(("Std Dev: ", "bold"), str(config.std_dev)),
    ]

    columns = Columns(config_items, equal=True, expand=True)
    panel = Panel(columns, title="Configuration", border_style="green")
    return panel

def update_status(i, status):
    try:
        dots = "."
        while status[i] == "Running" or status[i].startswith("Running."):
            status[i] = "Running" + dots + " " * (3 - len(dots))  # Pad with spaces
            dots = dots + "." if len(dots) < 3 else "."
            time.sleep(0.5)
    except Exception as e:
        raise e

def update_live(live, experiments, status):
    while True:
        live.update(display_experiments(experiments, status))
        time.sleep(0.1)




def get_predictions(model, logits, k, return_type):
    if return_type == "probabilities":
        logits = torch.softmax(logits, dim=-1)
    if return_type == "logprob":
        logits = torch.log_softmax(logits, dim=-1)

    prediction_tkn_ids = logits[0, -1, :].topk(k).indices.cpu().detach().numpy()
    prediction_tkns = [model.to_string(tkn_id) for tkn_id in prediction_tkn_ids]
    best_logits = logits[0, -1, prediction_tkn_ids]

    return best_logits, prediction_tkns


def squeeze_last_dims(tensor):
    if len(tensor.shape) == 3 and tensor.shape[1] == 1 and tensor.shape[2] == 1:
        return tensor.squeeze(-1).squeeze(-1)
    if len(tensor.shape) == 2 and tensor.shape[1] == 1:
        return tensor.squeeze(-1)
    else:
        return tensor


def suppress_warnings(fn):
    def wrapper(*args, **kwargs):
        # Save the current warnings state
        current_filters = warnings.filters[:]
        warnings.filterwarnings("ignore")
        try:
            return fn(*args, **kwargs)
        finally:
            # Restore the warnings state
            warnings.filters = current_filters

    return wrapper


def embs_to_tokens_ids(noisy_embs, model):
    input_embedding_norm = F.normalize(noisy_embs, p=2, dim=2)
    embedding_matrix_norm = F.normalize(model.W_E, p=2, dim=1)
    similarity = torch.matmul(input_embedding_norm, embedding_matrix_norm.T)
    corrupted_tokens = torch.argmax(similarity, dim=2)
    return corrupted_tokens





def list_of_dicts_to_dict_of_lists(list_of_dicts):
    # Initialize an empty dictionary to store the result
    dict_of_lists = {}
    
    # Iterate over each dictionary in the list
    for d in list_of_dicts:
        # Iterate over each key-value pair in the dictionary
        for key, value in d.items():
            # If the key is not already in the result dictionary, add it with an empty list as its value
            if key not in dict_of_lists:
                dict_of_lists[key] = []
            # Append the value to the list corresponding to the key in the result dictionary
            dict_of_lists[key].append(value)
    
    return dict_of_lists

def dict_of_lists_to_dict_of_tensors(dict_of_lists):
    dict_of_tensors = {}
    for key, tensor_list in dict_of_lists.items():
        dict_of_tensors[key] = torch.stack(tensor_list)
    return dict_of_tensors

def aggregate_result(pattern:torch.Tensor, object_positions:int, length:int) -> torch.Tensor:
    subject_1_1 = 5
    subject_1_2 = 6 if length > 15 else 5
    subject_1_3 = 7 if length > 17 else subject_1_2
    subject_2_1 = object_positions + 2
    subject_2_2 = object_positions + 3 if length > 15 else subject_2_1
    subject_2_3 = object_positions + 4 if length > 17 else subject_2_2
    subject_2_2 = subject_2_2 if subject_2_2 < length else subject_2_1
    subject_2_3 = subject_2_3 if subject_2_3 < length else subject_2_2
    last_position = length - 1
    object_positions_pre = object_positions - 1
    object_positions_next = object_positions + 1
    *leading_dims, pen_len, last_len = pattern.shape
    

    intermediate_aggregate = torch.zeros((*leading_dims, pen_len, 13))
    #aggregate for pre-last dimension
    intermediate_aggregate[..., 0] = pattern[..., :subject_1_1].mean(dim=-1)
    intermediate_aggregate[..., 1] = pattern[..., subject_1_1]
    intermediate_aggregate[..., 2] = pattern[..., subject_1_2]
    intermediate_aggregate[..., 3] = pattern[..., subject_1_3]
    intermediate_aggregate[..., 4] = pattern[..., subject_1_3 + 1:object_positions_pre].mean(dim=-1)
    intermediate_aggregate[..., 5] = pattern[..., object_positions_pre]
    intermediate_aggregate[..., 6] = pattern[..., object_positions]
    intermediate_aggregate[..., 7] = pattern[..., object_positions_next]
    intermediate_aggregate[..., 8] = pattern[..., subject_2_1]
    intermediate_aggregate[..., 9] = pattern[..., subject_2_2]
    intermediate_aggregate[..., 10] = pattern[..., subject_2_3]
    intermediate_aggregate[..., 11] = pattern[..., subject_2_3 + 1:last_position].mean(dim=-1)
    intermediate_aggregate[..., 12] = pattern[..., last_position]
    return intermediate_aggregate