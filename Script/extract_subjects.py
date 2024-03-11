
import sys
import os
sys.path.append(os.path.abspath(os.path.join("..")))
sys.path.append(os.path.abspath(os.path.join("../src")))
sys.path.append(os.path.abspath(os.path.join("../data")))
from Src.subject import AutoSubjectGenerator
import argparse


def main(path: str):
    context_dataset = AutoSubjectGenerator(path)
    context_dataset.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="pythia-6.9b")
    args = parser.parse_args()
    path = f"../data/full_data_sampled_{args.model_name}.json"
    main(path)

