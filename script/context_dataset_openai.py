
import sys
import os
sys.path.append(os.path.abspath(os.path.join("..")))
sys.path.append(os.path.abspath(os.path.join("../src")))
sys.path.append(os.path.abspath(os.path.join("../data")))
from src.context_dataset_generator import ContextDatasetGPT
import argparse


def main(model_name: str, n_samples: int):
    context_dataset = ContextDatasetGPT(model_name, n_samples)
    context_dataset.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--n_samples", type=int, default=100)
    args = parser.parse_args()
    main(args.model_name, args.n_samples)

