import sys
sys.path.append('..')
sys.path.append('../src')
sys.path.append('../data')

from src.score_models import EvaluateMechanism
from src.dataset import SampleDataset, HFDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
models_name = ["facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3B", "facebook/opt-2.7B"]
#models_name = ["EleutherAI/gpt-j-6b"]
#models_name = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
#models_name = ["facebook/opt-350m"]

for model_name in models_name:
    print("Loading model", model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
    )
    
    if len(model_name.split("/")) > 1:
        save_name = model_name.split("/")[1]

    else:
        save_name = model_name
    dataset_path = f"../data/full_data_sampled_{save_name}.json"
    if os.path.exists(dataset_path) == False:
        print("Creating sampled data")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = model.to(DEVICE)
        model.eval()
        sampler = SampleDataset("../data/full_data.json", model=model, save_path=dataset_path, tokenizer=tokenizer)
        sampler.sample()
        sampler.save()
        del model
        del sampler
        torch.cuda.empty_cache()        
    dataset = HFDataset(dataset_path, tokenizer=tokenizer, slice=10000)
    
    evaluator = EvaluateMechanism(
        model_name, dataset, device=DEVICE, batch_size=50, orthogonalize=True, family_name="opt")
    evaluator.evaluate_all()
