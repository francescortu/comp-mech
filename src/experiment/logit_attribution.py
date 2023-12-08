import torch
from torch.utils.data import DataLoader
import einops
from tqdm import tqdm
from src.dataset import TlensDataset
from src.model import WrapHookedTransformer
from src.base_experiment import BaseExperiment, to_logit_token, LogitStorage, IndexLogitStorage, HeadLogitStorage
from transformer_lens import ActivationCache
from typing import Optional, List, Tuple, Union, Dict, Any


class LogitAttribution(BaseExperiment):
    def __init__(self, dataset: TlensDataset, model: WrapHookedTransformer, batch_size:int):
        super().__init__(dataset, model, batch_size)
        
    def attribute_single_len(self, length:int, component:str):
        self.set_len(length, slice_to_fit_batch=False)
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        num_batches = len(dataloader)
        
        # Create a storage object to store the logits
        for batch in tqdm(dataloader, total=num_batches):
            logits, cache = self.model.run_with_cache(batch["corrupted_prompts"], return_cache=True)
            