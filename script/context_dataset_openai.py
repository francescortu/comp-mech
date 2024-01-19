
import sys
import os
sys.path.append(os.path.abspath(os.path.join("..")))
sys.path.append(os.path.abspath(os.path.join("../src")))
sys.path.append(os.path.abspath(os.path.join("../data")))
from src.context_dataset_generator import ContextDatasetGPT

def main():
    model_name = "gpt2"
    n_samples = 10000
    context_dataset = ContextDatasetGPT(model_name, n_samples)
    context_dataset.run()


if __name__ == "__main__":
    main()
