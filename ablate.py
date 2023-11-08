from src.ablate_heads import Ablate, AblateMultiLen, OVCircuit
from src.dataset import MyDataset
# from src.dataset import MyDataset
from src.model import WrapHookedTransformer
from src.myplot import plot_heatmaps, barplot_head
import torch
torch.set_grad_enabled(False)

model = WrapHookedTransformer.from_pretrained("gpt2", device="cuda", refactor_factored_attn_matrices=True)


dataset = MyDataset("data/full_data.json", model, slice=1000)
# dataset = MyDataset("../data/full_data.json", model.tokenizer, slice=1000)
print(dataset.get_lengths())
ablate_multi = AblateMultiLen(dataset, model, 40)

ablate_multi = AblateMultiLen(dataset, model, 40)
examples_mem, examples_cp = ablate_multi.ablate_multi_len(filter_outliers=False, )
