import torch
from transformer_lens import HookedTransformer

from typing import List
import warnings





class C:
    """Color class for printing colored text in terminal"""
    RED = "\033[91m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    END = "\033[0m"
    
    
class ModelWrapper:
    """Wrapper class for the HookedTransformer model to make it easier to use in the notebook"""
    def __init__(self, model_name: str):
        self.model = HookedTransformer.from_pretrained(model_name)

    def __call__(self, prompt: str):
        return self.predict(prompt)

    def predict(self, prompt: str):
        logits = self.model(prompt)
        return self.model.to_string(logits[0, -1, :].argmax(dim=-1))

    def get_probs(self, prompt: str, list_of_words: List[str], logit: bool = False):
        
        
        
        logits = self.model(prompt)
        probas = torch.softmax(logits[0, -1, :], dim=-1)
        tkns = self.model.to_tokens(list_of_words, prepend_bos=False)
        if logit:
            return logits[0, -1, tkns].mean().item()
        return probas[tkns].mean().item()
        
        
        # return probas[self.model.to_tokens(word, prepend_bos=False)].item()
    
    def print_top(self, prompt:str, k:int = 10, print_pred:bool = False):
        logits = self.model(prompt)
        probas = torch.softmax(logits[0, -1, :], dim=-1)
        pred_ids = logits[0, -1, :].topk(k).indices
        pred_ids = pred_ids.detach().numpy()
        if print_pred:
            for i in range(k):
                print(
                    f" {i} .{self.model.to_string(pred_ids[i])}  {logits[0,-1,pred_ids[i]].item():5.2f} {C.GREEN}{probas[pred_ids[i]].item():6.2%}{C.END} "
                )
        return [self.model.to_string(pred_ids[i]) for i in range(k)]
    
    
    
    

def get_predictions(model, logits, k, return_type):
    if return_type == "probabilities":
        logits = torch.softmax(logits, dim=-1)

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