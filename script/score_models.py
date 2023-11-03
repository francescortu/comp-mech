import sys
sys.path.append('..')
sys.path.append('../src')
sys.path.append('../data')

from src.score_models import MyDataset, EvaluateMechanism
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


models_name = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]

for model_name in models_name:
    print("Loading model", model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
    )

    dataset = MyDataset("../data/full_data.json", tokenizer=tokenizer)

    evaluator = EvaluateMechanism(
        model_name, dataset)
    evaluator.evaluate_all()