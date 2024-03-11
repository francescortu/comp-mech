from re import sub
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from Src.dataset import BaseDataset
from Src.model import BaseModel
from Src.base_experiment import BaseExperiment, to_logit_token
from typing import Optional, Tuple, Dict, Any, Literal, Union, List
import pandas as pd
from Src.experiment import LogitStorage, HeadLogitStorage
from functools import partial
from copy import deepcopy


class Ablator(BaseExperiment):
    def __init__(
        self,
        dataset: BaseDataset,
        model: BaseModel,
        batch_size: int,
        experiment: Literal["copyVSfact"],
        total_effect: bool = True
    ):
        super().__init__(dataset, model, batch_size, experiment)
        self.total_effect = total_effect
        
        