import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from script.run_all import logit_attribution
from src.dataset import TlensDataset
from src.model import WrapHookedTransformer
from src.experiment import LogitAttribution
from src.base_experiment import BaseExperiment, to_logit_token
from typing import Optional,  Tuple,  Dict, Any, Literal
import pandas as pd
from src.experiment import LogitStorage, HeadLogitStorage
from functools import partial
from copy import deepcopy


class AblationAttribution(LogitAttribution):
    def __init__(self, target_head:Tuple[int, int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_layer, self.target_head = target_head
    
    def get_hook(self, layer:int, head:int, position:int):
        