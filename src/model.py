import torch
from transformer_lens import HookedTransformer
from typing import List
from functools import partial
import einops

from .utils import C, get_predictions

torch.set_grad_enabled(False)

class WrapHookedTransformer(HookedTransformer):
    """
    Wrapper class for the HookedTransformer model to make it easier to use in the notebook and add some functionality
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, prompt: str, k: int = 1, return_type: str = "logits"):
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
                

    def add_noise(self, prompt: List[str], noise_index, target_win=None, noise_mlt=1):
        tokens = self.to_tokens(prompt)
        input_embeddings = self.embed(tokens)  # (batch_size, seq_len, emb_dim)

        # noise = torch.normal(mean=0, std=0.04, size=input_embeddings.shape, device=input_embeddings.device)
        # Load noise standard deviation and create noise tensor
        noise_mean = torch.load("../data/noise_mean.pt")
        noise_std = torch.load("../data/noise_std.pt") * noise_mlt
        noise_std = einops.repeat(noise_std, 'd -> b s d', b=input_embeddings.shape[0], s=input_embeddings.shape[1])
        noise_mean = einops.repeat(noise_mean, 'd -> b s d', b=input_embeddings.shape[0], s=input_embeddings.shape[1])
        noise = torch.normal(mean=noise_mean, std=noise_std)

        # Create a mask for positions specified in noise_index
        seq_len = input_embeddings.shape[1]
        noise_mask = torch.zeros(seq_len, device=input_embeddings.device)
        noise_mask[noise_index] = 1

        # If target_win is an integer, modify the noise_mask and noise tensor
        if isinstance(target_win, int):
            for idx in noise_index:
                if idx + target_win < seq_len:
                    noise_mask[idx + target_win] = 1
                    noise[:, idx + target_win, :] = noise[:, idx, :]

        # Expand the mask dimensions to match the noise tensor shape
        noise_mask = noise_mask.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1)
        noise_mask = noise_mask.expand_as(input_embeddings)  # (batch_size, seq_len, emb_dim)

        noise = noise.to(input_embeddings.device)
        noise_mask = noise_mask.to(input_embeddings.device)
        # Apply the mask to the noise tensor
        masked_noise = noise * noise_mask

        # Add the masked noise to the input embeddings
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
        orthogonal_embedding = random_embedding - ((random_embedding @ embedding) / (embedding @ embedding)) * embedding

        new_embedding = embedding + alpha * (orthogonal_embedding - embedding)
        new_embedding_normalize = torch.functional.F.normalize(new_embedding, dim=-1)
        embedding_normalize = torch.functional.F.normalize(self.W_E, dim=-1)
        similarity = embedding_normalize @ new_embedding_normalize

        # Exclude the original token and find the closest token
        sorted_indices = torch.argsort(similarity, descending=True)
        sorted_indices = sorted_indices[sorted_indices != token]
        new_token = sorted_indices[0]

        # print(string_token, self.to_string(new_token.item()), alpha)
        return self.to_string(new_token.item())
