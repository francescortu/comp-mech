import sys
sys.path.append('..')
sys.path.append('../src')
sys.path.append('../data')

from src.ablate_heads import AblateMultiLen
from src.dataset import TlensDataset
from src.model import WrapHookedTransformer
import torch
torch.set_grad_enabled(False)

model = WrapHookedTransformer.from_pretrained("gpt2", device="cuda", refactor_factored_attn_matrices=True)
dataset = TlensDataset("../data/full_data_sampled_gpt2.json", model)

ablator = AblateMultiLen(dataset, model, 100)
examples_mem, examples_cp = ablator.ablate_multi_len()

torch.save(examples_mem, "../results/examples_mem.pt")
torch.save(examples_cp, "../results/examples_cp.pt")

