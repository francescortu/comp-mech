import sys
sys.path.append('..')
sys.path.append('../src')
sys.path.append('../data')

from src.score_models import MyDataset, EvaluateMechanism
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
models_name = ["facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3B", "facebook/opt-2.7B"]

for model_name in models_name:
    print("Loading model", model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
    )
    dataset = MyDataset("../data/full_data.json", tokenizer=tokenizer, slice=10000)
    
    evaluator = EvaluateMechanism(
        model_name, dataset, device=DEVICE, batch_size=50)
    evaluator.evaluate_all()
