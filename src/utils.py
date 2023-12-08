import torch
from transformer_lens import HookedTransformer

# from src.model import WrapHookedTransformer
import transformer_lens.patching as patching

from typing import List
import warnings
import einops


class C:
    """Color class for printing colored text in terminal"""

    RED = "\033[91m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    END = "\033[0m"
    
    



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
    input_embedding_norm = torch.functional.F.normalize(noisy_embs, p=2, dim=2)
    embedding_matrix_norm = torch.functional.F.normalize(model.W_E, p=2, dim=1)
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