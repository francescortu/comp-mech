import torch
from transformer_lens import HookedTransformer
from functools import partial


from .utils import  get_predictions

torch.set_grad_enabled(False)

class WrapHookedTransformer(HookedTransformer):
    """
    Wrapper class for the HookedTransformer model to make it easier to use in the notebook and add some functionality
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = self.cfg.device
    def predict(self, prompt: str, k: int = 1, return_type: str = "logits"):
        logits = self(prompt)
        return get_predictions(self, logits, k, return_type)
    
    # @classmethod
    # def from_pretrained(cls, *args, **kwargs):
    #     # Call the superclass's from_pretrained method
    #     hooked_transformer = super().from_pretrained(*args, **kwargs)

    #     # Create a new instance of WrapHookedTransformer that wraps the HookedTransformer instance
    #     return cls(hooked_transformer)
    
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
    
    def to_orthogonal_tokens2(self, string_token: str, alpha: float = 1):
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

        # Use linear interpolation between the original embedding and the orthogonal embedding
        new_embedding = (1 - alpha) * embedding + alpha * orthogonal_embedding
        new_embedding_normalize = torch.functional.F.normalize(new_embedding, dim=-1)
        embedding_normalize = torch.functional.F.normalize(self.W_E, dim=-1)
        similarity = embedding_normalize @ new_embedding_normalize

        # Exclude the original token and find the closest token
        sorted_indices = torch.argsort(similarity, descending=True)
        sorted_indices = sorted_indices[sorted_indices != token]
        new_token = sorted_indices[0]

        return self.to_string(new_token.item())
