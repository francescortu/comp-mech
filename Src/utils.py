from ast import FunctionType
from math import e
from re import sub
from numpy import object_
from sympy import Function, fu
import torch
import warnings
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.table import Table
import time
import os
import torch.nn.functional as F
from typing import Literal, Union


def check_dataset_and_sample(dataset_path, model_name, hf_model_name):
    if os.path.exists(dataset_path):
        print("Dataset found!")
        return
    else:
        raise FileNotFoundError("Dataset not found, please create it first")


def display_experiments(experiments, status):
    table = Table(show_header=True, header_style="bold magenta", expand=True)
    table.add_column("Experiment", style="dim", width=3)
    table.add_column("Status", width=1)
    table.add_column("Live Output")  # New column for future live output
    for experiment, stat in zip(experiments, status):
        table.add_row(
            experiment.__name__, stat, ""
        )  # Empty string for future live output
    return table


def display_config(config):
    config_items = [
        Text.assemble(("Model Name: ", "bold"), str(config.model_name)),
        Text.assemble(("Batch Size: ", "bold"), str(config.batch_size)),
        Text.assemble(("Dataset Path: ", "bold"), config.dataset_path),
        Text.assemble(("Dataset End: ", "bold"), str(config.dataset_end)),
        Text.assemble(("Produce Plots: ", "bold"), str(config.produce_plots)),
        Text.assemble(("Normalize Logit: ", "bold"), str(config.normalize_logit)),
        Text.assemble(("Std Dev: ", "bold"), str(config.std_dev)),
        Text.assemble(("Total Effect: ", "bold"), str(config.total_effect)),
        Text.assemble(("Up-to-layer: ", "bold"), str(config.up_to_layer)),
        Text.assemble(("Experiment Name: ", "bold"), str(config.experiment)),
        Text.assemble(("Flag: ", "bold"), str(config.flag)),
        Text.assemble(("HF Model: ", "bold"), str(config.hf_model))
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


def aggregate_result(
    experiment: Literal["copyVSfact", "contextVSfact", "copyVSfact_factual"],
    pattern: torch.Tensor,
    object_positions: torch.Tensor,
    first_subject_positions: torch.Tensor,
    second_subject_positions: torch.Tensor,
    subject_lengths: torch.Tensor,
    length: int,
) -> torch.Tensor:
    if "copyVSfact" in experiment:
        return aggregate_result_copyVSfact(
            pattern,
            object_positions,
            first_subject_positions,
            second_subject_positions,
            subject_lengths,
            length,
        )
    elif experiment == "contextVSfact":
        raise NotImplementedError("Not implemented yet")
    else:
        raise ValueError("Invalid experiment name")

AGGREGATED_DIMS = 14
def aggregate_result_copyVSfact(
    pattern: torch.Tensor,
    object_positions: torch.Tensor,
    first_subject_positions: torch.Tensor,
    second_subject_positions: torch.Tensor,
    subject_lengths: torch.Tensor,
    length: int,
) -> torch.Tensor:
    batch_size = pattern.shape[1]
    assert batch_size == first_subject_positions.shape[0], "Batch size mismatch"
    
    *leading_dims, pen_len, last_len = pattern.shape
    intermediate_aggregate = torch.zeros((*leading_dims, pen_len, AGGREGATED_DIMS))
    
    for i in range(batch_size):
        intermediate_aggregate[:,i,:] = aggregate_single_result_copyVSfact(
            pattern[:,i,:],
            object_positions[i],
            first_subject_positions[i],
            second_subject_positions[i],
            subject_lengths[i],
            length,
        )
    return intermediate_aggregate
    
def aggregate_single_result_copyVSfact(
    pattern: torch.Tensor,
    object_positions: torch.Tensor,
    first_subject_positions: torch.Tensor,
    second_subject_positions: torch.Tensor,
    subject_lengths: torch.Tensor,
    length: int,
) -> torch.Tensor:
    #assert the shape of the object_positions, subject_positions and subject_lengths
    assert object_positions.shape == first_subject_positions.shape == subject_lengths.shape, "Shape mismatch"
    assert len(object_positions.shape) == 0, "Shape mismatch"
    *leading_dims, pen_len, last_len = pattern.shape
    intermediate_aggregate = torch.zeros((*leading_dims, pen_len, AGGREGATED_DIMS))
    
    
    # pre-subject
    pre_subject = slice(0, first_subject_positions-1)
    intermediate_aggregate[..., 0] = pattern[..., pre_subject].mean(dim=-1)
    
    # first token subject
    first_token_subject = first_subject_positions
    intermediate_aggregate[..., 1] = pattern[..., first_token_subject]
    
    # intermediate tokens subject
    last_subject = first_subject_positions + subject_lengths
    
    if (last_subject - 1 - (first_token_subject + 1)) < 0:
        intermediate_aggregate[..., 2] = 0
    elif (last_subject - 1 - (first_token_subject + 1)) == 0:
        intermediate_aggregate[..., 2] = pattern[..., first_token_subject + 1]
    else:
        intermediate_token_subject = slice(first_subject_positions +1, first_subject_positions + subject_lengths - 1)
        intermediate_aggregate[..., 2] = pattern[..., intermediate_token_subject].mean(dim=-1)
    
    # last token subject
    last_token_subject = first_subject_positions + subject_lengths
    intermediate_aggregate[..., 3] = pattern[..., last_token_subject]
    
    # between subject and object
    if ((object_positions -1) - (last_token_subject + 1)) < 0:
            intermediate_aggregate[..., 4] = 0
    elif ((object_positions -1) - (last_token_subject + 1)) == 0:
        intermediate_aggregate[..., 4] = pattern[..., last_token_subject + 1]
    else:
        intermediate_aggregate[..., 4] = pattern[..., last_token_subject + 1 : object_positions - 1].mean(dim=-1) 
    
    # pre-object
    pre_object = object_positions - 1
    intermediate_aggregate[..., 5] = pattern[..., pre_object].mean(dim=-1)
    
    # object
    intermediate_aggregate[..., 6] = pattern[..., object_positions]
    
    # next object
    next_object = object_positions + 1
    intermediate_aggregate[..., 7] = pattern[..., next_object]
    
    # from next object to second subject
    if ((second_subject_positions - 1) - (next_object + 1)) < 0:
        intermediate_aggregate[..., 8] = 0
    elif ((second_subject_positions - 1) - (next_object + 1)) == 0:
        intermediate_aggregate[..., 8] = pattern[..., next_object + 1]
    else:
        intermediate_aggregate[..., 8] = pattern[..., next_object + 1 : second_subject_positions - 1].mean(dim=-1)
        
    # first token second subject
    first_token_second_subject = second_subject_positions
    intermediate_aggregate[..., 9] = pattern[..., first_token_second_subject]
    
    # intermediate tokens second subject
    last_second_subject = second_subject_positions + subject_lengths
    if (last_second_subject - 1 - (first_token_second_subject + 1)) < 0:
        intermediate_aggregate[..., 10] = 0
    elif (last_second_subject - 1 - (first_token_second_subject + 1)) == 0:
        intermediate_aggregate[..., 10] = pattern[..., first_token_second_subject + 1]
    else:
        intermediate_aggregate[..., 10] = pattern[..., first_token_second_subject + 1 : last_second_subject - 1].mean(dim=-1)
        
    # last token second subject
    last_token_second_subject = second_subject_positions + subject_lengths
    intermediate_aggregate[..., 11] = pattern[..., last_token_second_subject]
    
    # second subject to last token
    last_token = length - 1
    if (last_token - 1 - (last_token_second_subject + 1)) < 0:
        intermediate_aggregate[..., 12] = 0
    elif (last_token - 1 - (last_token_second_subject + 1)) == 0:
        intermediate_aggregate[..., 12] = pattern[..., last_token_second_subject + 1]
    else:
        intermediate_aggregate[..., 12] = pattern[..., last_token_second_subject + 1 : last_token - 1].mean(dim=-1)
    
    # last token
    last_token = length - 1
    intermediate_aggregate[..., 13] = pattern[..., last_token]
    
    return intermediate_aggregate
    
    
    



def aggregate_result_contextVSfact(
    pattern: torch.Tensor,
    object_positions: Union[torch.Tensor, int],
    length: int,
    subj_positions: int,
    batch_dim: int,
) -> torch.Tensor:
    raise NotImplementedError("Not implemented yet")


def aggregate_single_result_contextVSfact(
    pattern: torch.Tensor, object_positions: int, length: int, subj_position: int
) -> torch.Tensor:
    raise NotImplementedError("Not implemented yet")
