import sys
import torch
import json
from dataclasses import dataclass
import einops

# Add paths
for path in ['..', '../src', '../data']:
    sys.path.append(path)

from src.ablation import Ablator
import argparse 


def get_name_file(args):
    base_name = f"{args.model}_result_len_{args.length}"
    if args.interval != 1:
        base_name = base_name + f"_{args.interval}"
    return base_name

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--name_dataset", type=str, default="dataset_gpt2_f.json")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--length", type=int, default=16)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=50)
    return parser.parse_args()


@dataclass
class Config:
    model_name: str
    name_dataset: str
    n:str
    length: int
    patch_interval: int
    batch_size: int
    name_save_file: str
    @classmethod
    def from_args(cls, args):
        return cls(
            model_name=args.model,
            name_dataset=args.name_dataset,
            n=args.n,
            length=args.length,
            patch_interval=args.interval,
            batch_size=args.batch_size,
            name_save_file=get_name_file(args)
        )

def main():
    args = get_args()
    config = Config.from_args(args)
    print("Run ablation with config:")
    print(config)
    ablator = Ablator(config)
    ablator.ablate()
    
if __name__ == "__main__":
    main()
