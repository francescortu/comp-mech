import sys
sys.path.append('..')
sys.path.append('../src')
sys.path.append('../data')

from src.score_models import HFDataset, EvaluateMechanism
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import matplotlib.pyplot as plt

plt.style.use('ggplot')
#generate random data


#data = np.random.normal(0, 20, 1000)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
models_name = ["gpt2"]



for model_name in models_name:
    print("Loading model", model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
    )
    dataset = HFDataset("data/full_data_sampled_gpt2.json", tokenizer=tokenizer, slice=10000)
    
    evaluator = EvaluateMechanism(
        model_name, dataset, device=DEVICE, batch_size=20)
    evaluator.evaluate_all()
