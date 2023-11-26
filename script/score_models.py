import sys
import os

from pydantic import conint


# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory (..) to sys.path
sys.path.append(os.path.join(script_dir, '..'))

# Optionally, add the 'src' directory directly
sys.path.append(os.path.join(script_dir, '..', 'src'))

from src.score_models import EvaluateMechanism
from src.dataset import SampleDataset, HFDataset, HFDatasetConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from argparse import ArgumentParser
from dataclasses import dataclass, field
from typing import List, Tuple


FAMILY_NAME = "gpt2"

@dataclass
class Options:
    models_name: List[str] = field(default_factory=lambda: ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"])
    premise: List[str] = field(default_factory=lambda: ["Redefine", "Assume", "Suppose", "Context"])
    orthogonalize: List[bool] = field(default_factory=lambda: [True, False])
    alpha: List[float] = field(default_factory=lambda: [1, 0.75, 0.5, 0.25])
    interval: List[Tuple[float, float]] = field(default_factory=lambda: [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1)])
    #alpha: List[float] = field(default_factory=lambda: [ 0.5, 0.25])
    #interval: List[Tuple[float, float]] = field(default_factory=lambda: [(0.5, 0.75), (0.75, 1)])

@dataclass
class LaunchConfig:
    model_name: str
    orthogonalize: bool
    alpha: float
    interval: tuple[float, float]
    family_name: str
    premise: str = "Redefine"
    batch_size:int = 50
    
    
def launch_evaluation(config:LaunchConfig):
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Loading model", config.model_name)
    print("Launch config", config)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
    )

    if len(config.model_name.split("/")) > 1:
        save_name = config.model_name.split("/")[1]

    else:
        save_name = config.model_name
    dataset_path = f"data/full_data_sampled_{save_name}.json"
    if os.path.exists(dataset_path) == False:
        print("Creating sampled data")
        model = AutoModelForCausalLM.from_pretrained(config.model_name)
        model = model.to(DEVICE)
        model.eval()
        sampler = SampleDataset("../data/full_data.json", model=model, save_path=dataset_path, tokenizer=tokenizer)
        sampler.sample()
        sampler.save()
        del model
        del sampler
        torch.cuda.empty_cache()        
    dataset = HFDataset(dataset_path, tokenizer=tokenizer,config=HFDatasetConfig(premise = config.premise, alpha=config.alpha, interval=config.interval), slice=10000)

    evaluator = EvaluateMechanism(
        model_name=config.model_name, dataset=dataset, device=DEVICE, batch_size=config.batch_size, orthogonalize=config.orthogonalize, premise=config.premise, interval=config.interval, family_name=config.family_name)
    evaluator.evaluate_all()
    
def main():
    parser = ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--similarity", action="store_true")
    parser.add_argument("--input-variation", action="store_true")
    args = parser.parse_args()

    options = Options()
    
    if args.all:
        for model_name in options.models_name:
            for premise in options.premise:
                for orthogonalize in options.orthogonalize:
                    if orthogonalize is False:
                        config = LaunchConfig(model_name, orthogonalize, 1, (0,0.25), FAMILY_NAME, premise)
                    elif orthogonalize is True:
                        for idx in range(len(options.alpha)):
                            config = LaunchConfig(model_name, orthogonalize, options.alpha[idx], options.interval[idx], FAMILY_NAME, premise)
                            launch_evaluation(config)
    
    if args.similarity:
        for model_name in options.models_name:
                for orthogonalize in options.orthogonalize:
                    if orthogonalize is True:
                        config = LaunchConfig(model_name,  orthogonalize, 1, (0,0.25), FAMILY_NAME) # premise default
                    elif orthogonalize is False:
                        for idx in range(len(options.alpha)):
                            config = LaunchConfig(model_name, orthogonalize, options.alpha[idx], options.interval[idx], FAMILY_NAME)
                            launch_evaluation(config)
                            
                    
    if args.input_variation:
        for model_name in options.models_name:
            for premise in options.premise:
                for orthogonalize in options.orthogonalize:
                        config = LaunchConfig(model_name, False, 1, (0,0.25), FAMILY_NAME, premise)

    
    


if __name__ == "__main__":
    main()
