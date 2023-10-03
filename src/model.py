import torch
from transformer_lens import HookedTransformer
from typing import List
from functools import partial

from .utils import C, get_predictions

torch.set_grad_enabled(False)

class WrapHookedTransformer(HookedTransformer):
    """
    Wrapper class for the HookedTransformer model to make it easier to use in the notebook and add some functionality
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, prompt: str, k: int = 1, return_type: str = "logits"):
        """
        Predict the next token(s) given a prompt, return a tuple of (logits/probas, tokens)
        
        Args:
            prompt (str): The prompt to predict from
            k (int): The number of tokens to return in descending order of probability
            return_type (str): Either "logits" or "probabilities"
        
        Returns:
            Tuple of (logits/probas, tokens)
        """
        logits = self(prompt)
        return get_predictions(self, logits, k, return_type)
    


    def show_predictions(self, prompt: str, n_tokens: int = 10, return_type: str = "logits"):
        """
        Print the next token(s) given a prompt
        
        Args:
            prompt (str): The prompt to predict from
            n_tokens (int): The number of tokens to return in descending order of probability
            return_type (str): Either "logits" or "probabilities"
        """
        logits, prediction_tkns = self.predict(prompt, k=n_tokens, return_type=return_type)
        for i in range(n_tokens):
            if return_type == "probabilities":
                print(f"{i} {prediction_tkns[i]} {C.GREEN}{logits[i].item():6.2%}{C.END}")
            else:
                print(f"{i} {prediction_tkns[i]} {C.GREEN}{logits[i].item():5.2f}{C.END}")
                
                
    # def add_noise(self, prompt: List[str], max_pos_noise):
    #     tokens = self.to_tokens(prompt)
    #     input_embeddings = self.embed(tokens) # (batch_size, seq_len, emb_dim)
        
    #     # add noise to the input embeddings
    #     noise = torch.normal(mean=0, std=0.8, size=input_embeddings.shape, device=input_embeddings.device)
    #     # remove noise from tokens in position >= max_pos_noise
    #     noise_index = torch.arange(input_embeddings.shape[0], device=input_embeddings.device)
    #     # put noise to 0 except for the first token and the max_pos_noise token
    #     noise_index = torch.where((noise_index != 0) & (noise_index != max_pos_noise), torch.tensor(0, device=input_embeddings.device), noise_index)
        
    #     # add noise to the input embeddings
    #     corrupted_embeddings = input_embeddings + noise
    #     return corrupted_embeddings
    def add_noise(self, prompt: List[str], noise_index):
        tokens = self.to_tokens(prompt)
        input_embeddings = self.embed(tokens) # (batch_size, seq_len, emb_dim)

        # create noise
        noise = torch.normal(mean=0, std=0.8, size=input_embeddings.shape, device=input_embeddings.device)

        # create a mask for positions 0 and max_pos_noise
        seq_len = input_embeddings.shape[1]
        noise_mask = torch.zeros(seq_len, device=input_embeddings.device)
        noise_mask[noise_index] = 1

        # noise_mask[max_pos_noise] = 1

        # expand the mask dimensions to match the noise tensor shape
        noise_mask = noise_mask.unsqueeze(0).unsqueeze(2) # (1, seq_len, 1)
        noise_mask = noise_mask.expand_as(input_embeddings) # (batch_size, seq_len, emb_dim)

        # apply the mask to the noise tensor
        masked_noise = noise * noise_mask

        # add the masked noise to the input embeddings
        corrupted_embeddings = input_embeddings + masked_noise
        return corrupted_embeddings

    
    
    def run_with_cache_from_embed(self, input_embeddings , hook_fn=[], return_cache=True, *args,  **kwargs ):
        """
        Run the model with the cache enabled
        """

        def embed_hook(cache, hook, input_embeddings):
            cache[:,:,:] = input_embeddings
            return cache

        placeholder = torch.zeros(input_embeddings.shape[:-1], dtype=torch.long)
        hook_embed = partial(embed_hook, input_embeddings=input_embeddings)
        hooks = [("hook_embed", hook_embed)] + hook_fn
        return self.run_with_hooks(placeholder,
                                    fwd_hooks=hooks,
                                    return_cache=return_cache,
                                    return_type="logits")
        

        
            

        
    # def to_orthogonal_tokens(self, string_token: str, alpha: float = 1):
    #     """
    #     Convert a token to its orthogonal representation
        
    #     Args:
    #         string_token (str): The token to convert
    #         alpha (float): The amount of orthogonalization to apply
    #     """
    #     token = self.to_tokens(string_token, prepend_bos=False)

        
    #     token = token[0]
    #     embedding = self.W_E[token].mean(dim=0).squeeze(0)

    #     random_embedding = torch.randn_like(embedding)
    #     orthogonal_embedding = random_embedding - (random_embedding @ embedding) / (embedding @ embedding) * embedding
    
    #     new_embedding = embedding + alpha * (orthogonal_embedding - embedding)
    #     # from the orthogonal embedding, find the closest token
    #     new_embedding = torch.argmin(torch.norm(self.W_E - orthogonal_embedding, dim=-1))
    #     return self.to_string(new_embedding.item())
    
    def to_orthogonal_tokens(self, string_token: str, alpha: float = 1):
        """
        Convert a token to its orthogonal representation
        
        Args:
            string_token (str): The token to convert
            alpha (float): The amount of orthogonalization to apply
        """
        token = self.to_tokens(string_token, prepend_bos=False)

        
        token = token[0]
        embedding = self.W_E[token].mean(dim=0).squeeze(0)

        random_embedding = torch.randn_like(embedding)
        orthogonal_embedding = random_embedding - (random_embedding @ embedding) / (embedding @ embedding) * embedding
    
        new_embedding = embedding + alpha * (orthogonal_embedding - embedding)
        # from the orthogonal embedding, find the closest token
        new_embedding_normalize = torch.functional.F.normalize(new_embedding, dim=-1)
        embedding_normalize = torch.functional.F.normalize(self.W_E, dim=-1)
        similarity = embedding_normalize @ new_embedding_normalize
        new_embedding = torch.argmax(similarity)

        return self.to_string(new_embedding.item())