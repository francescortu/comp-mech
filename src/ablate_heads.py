from src.dataset import MyDataset
from src.model import WrapHookedTransformer
import einops
import torch
from tqdm import tqdm
from functools import partial
from copy import deepcopy



def to_logit_token(logit, target):
    logit = torch.log_softmax(logit, dim=-1)
    logit_mem = torch.zeros(target.shape[0])
    logit_cp = torch.zeros(target.shape[0])
    for i in range(target.shape[0]):
        logit_mem[i] = logit[i, target[i,0]]
        logit_cp[i] = logit[i, target[i,1]]
    return logit_mem, logit_cp
    
    

class Ablate():
    def __init__(self, dataset:MyDataset, model:WrapHookedTransformer,  batch_size, filter_outliers=False):
        self.dataset = dataset
        self.model = model
        self.model.eval()
        self.batch_size = batch_size
        self.filter_outliers = filter_outliers

    def set_len(self, length):
        self.dataset.set_len(length, self.model)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
    
    def compute_logit(self):
        clean_logit = []
        corrupted_logit = []
        target = []
        for batch in tqdm(self.dataloader):
            clean_logit.append(self.model(batch["clean_prompts"])[:,-1,:].cpu())
            corrupted_logit.append(self.model(batch["corrupted_prompts"])[:,-1,:].cpu())
            target.append(batch["target"].cpu())
        clean_logit = torch.cat(clean_logit, dim=0)
        corrupted_logit = torch.cat(corrupted_logit, dim=0)
        target = torch.cat(target, dim=0)
        return clean_logit, corrupted_logit, target
    
    def create_dataloader(self, filter_outliers=False):
        if filter_outliers:
            self.filter_outliers()
        else:
            self.slice_to_fit_batch()
    
    def filter_outliers(self):
        print("Number of examples before outliers:", len(self.dataset))
        clean_logit, corrupted_logit, target = self.compute_logit()
        clean_logit_mem, clean_logit_cp = to_logit_token(clean_logit, target)
        corrupted_logit_mem, corrupted_logit_cp = to_logit_token(corrupted_logit, target)
        
        outliers_under = torch.where(corrupted_logit_mem < (corrupted_logit_mem.mean() - corrupted_logit_mem.std()))[0]
        outliers_over = torch.where(corrupted_logit_cp > (corrupted_logit_cp.mean() + corrupted_logit_cp.std()))[0]
        outliers_indexes = torch.cat([outliers_under, outliers_over], dim=0).tolist()
        
        maxdatasize = ((len(self.dataset) - len(outliers_indexes))//self.batch_size)*self.batch_size
        
        self.dataset.filter_from_idx(outliers_indexes, exclude=True)
        self.dataset.slice(maxdatasize)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        self.num_batches = len(self.dataloader)
        print("Number of examples after outliers:", len(self.dataloader)*self.batch_size)
        
    def slice_to_fit_batch(self):
        self.dataset.slice_to_fit_batch(self.batch_size)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        self.num_batches = len(self.dataloader)
        print("Number of examples after slicing:", len(self.dataloader)*self.batch_size)
        
    def get_normalize_metric(self):
        clean_logit, corrupted_logit, target = self.compute_logit()
        clean_logit_mem, clean_logit_cp = to_logit_token(clean_logit, target)
        corrupted_logit_mem, corrupted_logit_cp = to_logit_token(corrupted_logit, target)
        
        def normalize_logit_token(logit_mem, logit_cp,  baseline="corrupted",):
            # logit_mem, logit_cp = to_logit_token(logit, target)
            # percentage increase or decrease of logit_mem
            if baseline == "clean":
                logit_mem = 100 * (logit_mem - clean_logit_mem) / clean_logit_mem
                # percentage increase or decrease of logit_cp
                logit_cp = 100 * (logit_cp - clean_logit_cp) / clean_logit_cp
                return -logit_mem, -logit_cp
            elif baseline == "corrupted":
                logit_mem = 100 * (logit_mem - corrupted_logit_mem) / corrupted_logit_mem
                # percentage increase or decrease of logit_cp
                logit_cp = 100 * (logit_cp - corrupted_logit_cp) / corrupted_logit_cp
                return -logit_mem, -logit_cp
        
        return normalize_logit_token
    
    def ablate_heads(self):
        normalize_logit_token = self.get_normalize_metric()
        def heads_hook(activation, hook, head,  pos1=None, pos2=None):
            activation[:, head, -1, :] = 0
            return activation

        def freeze_hook(activation, hook,  clean_activation, pos1=None, pos2=None):
            activation = clean_activation
            return activation
        
        examples_mem = torch.zeros((self.model.cfg.n_layers, self.model.cfg.n_heads, self.num_batches, self.batch_size), device="cpu")
        examples_cp = torch.zeros((self.model.cfg.n_layers, self.model.cfg.n_heads, self.num_batches, self.batch_size), device="cpu")
        
        
        for idx, batch in tqdm(enumerate(self.dataloader), total=self.num_batches, desc="Ablating batches"):
            _, clean_cache = self.model.run_with_cache(batch["corrupted_prompts"])
            hooks = {}
            for layer in range(self.model.cfg.n_layers):
                # for head in range(self.model.cfg.n_heads):
                hooks[f"L{layer}"] = (f"blocks.{layer}.hook_attn_out",
                                partial(
                                    freeze_hook,
                                    clean_activation=clean_cache[f"blocks.{layer}.hook_attn_out"],
                                    )
                                )
                    
            for layer in range(self.model.cfg.n_layers):
                for head in range(self.model.cfg.n_heads):
                    tmp_hooks = deepcopy(hooks)
                    tmp_hooks[f"L{layer}"] = (f"blocks.{layer}.attn.hook_pattern",
                                    partial(
                                        heads_hook,
                                        head=head,
                                        )
                                    )
                    
                    list_hooks = list(tmp_hooks.values())
                    self.model.reset_hooks()
                    logit = self.model.run_with_hooks(
                        batch["corrupted_prompts"],
                        fwd_hooks=list_hooks,
                    )[:,-1,:]
                    mem, cp = to_logit_token(logit, batch["target"])
                    # norm_mem, norm_cp = normalize_logit_token(mem, cp, baseline="corrupted")
                    examples_mem[layer, head, idx, :] = mem.cpu()
                    examples_cp[layer, head, idx, :] = cp.cpu()
                #remove the hooks for the previous layer

                hooks.pop(f"L{layer}")
                
        examples_mem = einops.rearrange(examples_mem, "l h b s -> l h (b s)")
        examples_cp = einops.rearrange(examples_cp, "l h b s -> l h (b s)")
        for layer in range(self.model.cfg.n_layers):
            for head in range(self.model.cfg.n_heads):
                norm_mem, norm_cp = normalize_logit_token(examples_mem[layer, head, :], examples_cp[layer, head, :], baseline="corrupted")
                examples_mem[layer, head, :] = norm_mem
                examples_cp[layer, head, :] = norm_cp
                
        return examples_mem, examples_cp
    
    
class AblateMultiLen():
    def __init__(self, dataset:MyDataset, model:WrapHookedTransformer,  batch_size):
        self.dataset = dataset
        self.model = model
        self.batch_size = batch_size
        self.ablate = Ablate(dataset, model, batch_size)
        
    def ablate_single_len(self, length, filter_outliers=False):
        self.dataset.set_len(length, self.model)
        self.ablate.set_len(length)
        self.ablate.create_dataloader(filter_outliers=filter_outliers)
        return self.ablate.ablate_heads()
    
    def ablate_multi_len(self, filter_outliers=False):
        lenghts = self.dataset.get_lengths()
        
        result_cp_per_len = {}
        result_mem_per_len = {}
        for l in lenghts:
            print("Ablating examples of length", l, "...")
            result_cp_per_len[l], result_mem_per_len[l] = self.ablate_single_len(l, filter_outliers=filter_outliers)
        
        # concatenate the results
        result_cp = torch.cat(list(result_cp_per_len.values()), dim=-1)
        result_mem = torch.cat(list(result_mem_per_len.values()), dim=-1)
        print("result_cp.shape", result_cp.shape)
        
        return result_mem, result_cp
    
    

class OVCircuit:
    def __init__(self, model:WrapHookedTransformer, dataset:MyDataset, batch_size):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        
    def ov_single_len(self, resid_layer_input, resid_pos:str, layer, head, length, disable_tqdm=False):
        self.dataset.set_len(length, self.model)
        self.dataset.slice_to_fit_batch(self.batch_size)
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        num_batches = len(dataloader)
        
        if resid_pos == "1_1_subject":
            position = 5
        elif resid_pos == "1_1_subject":
            position = 6
        elif resid_pos == "1_1_subject":
            position = 7
        elif resid_pos == "definition":
            position = self.dataset.obj_pos[0]
        elif resid_pos == "2_1_subject":
            position = self.dataset.obj_pos[0] + 2
        elif resid_pos == "2_2_subject":
            position = self.dataset.obj_pos[0] + 3
        elif resid_pos == "2_3_subject":
            position = self.dataset.obj_pos[0] + 4
        else:
            raise ValueError("resid_pos not recognized: should be one of 1_1_subject, 1_2_subject, 1_3_subject, definition, 2_1_subject, 2_2_subject, 2_3_subject")
        
        logit_ov = torch.zeros((num_batches, self.batch_size, self.model.cfg.d_vocab))
        for idx, batch in tqdm(enumerate(dataloader), total=num_batches, desc=f"OV circuit at ({layer},{head})", disable=disable_tqdm):
            _, cache = self.model.run_with_cache(batch["corrupted_prompts"])
            residual_stream = cache["resid_post", resid_layer_input]
            W_OV = (self.model.blocks[layer].attn.W_V @ self.model.blocks[layer].attn.W_O)[head]
            logit_output = einops.einsum(self.model.W_U, (residual_stream[:,position,:] @ W_OV), "d d_v, b d -> b d_v")
            logit_ov[idx] = self.model.ln_final(logit_output)
        
        logit_ov = einops.rearrange(logit_ov, "b s d -> (b s) d")
        return logit_ov
    
    def ov_single_copy_score(self, resid_layer_input, resid_pos:str, layer, head, length, target="copy", disable_tqdm=False):
        if target == "copy":
            target_idx = 1
        elif target == "mem":
            target_idx = 0
        logit_ov = self.ov_single_len(resid_layer_input, resid_pos, layer, head, length, disable_tqdm=disable_tqdm)
        num_examples = logit_ov.shape[0]
        # get the top 10 tokens for each example
        topk = 10
        topk_tokens = torch.topk(logit_ov, k=topk, dim=-1).indices
        target_list = [self.model.to_string(self.dataset.target[i,target_idx]) for i in range(num_examples)]
    
        count = 0
        for i in range(num_examples):
            if target_list[i] in [self.model.to_string(topk_tokens[i,j]) for j in range(topk)]:
                count += 1
        
        percentage = 100* count/num_examples
        return percentage
    
    def ov_multi_len(self, resid_layer_input, resid_pos:str, layer, head, disable_tqdm=False):
        lenghts = self.dataset.get_lengths()
        logit = {}
        for l in lenghts:
            logit[l] = self.ov_single_len(resid_layer_input, resid_pos, layer, head, l, disable_tqdm=disable_tqdm)
        logit = torch.cat(list(logit.values()), dim=0)
        return logit
    
    def ov_multi_copy_score(self, resid_layer_input, resid_pos, layer, head, target="copy", disable_tqdm=False):
        lenghts = self.dataset.get_lengths()
        copy_score = {}
        for l in lenghts:
            copy_score[l] = self.ov_single_copy_score(resid_layer_input, resid_pos, layer, head, l, target=target, disable_tqdm=disable_tqdm)
        
        # mean of the copy score
        copy_score = torch.tensor(list(copy_score.values())).mean().item()
        return copy_score
    
    def ov_single_len_all_heads_score(self, resid_layer_input, resid_pos:str, length, target="copy", disable_tqdm=False, plot=False):
        self.dataset.set_len(length, self.model)
        self.dataset.slice_to_fit_batch(self.batch_size)
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        num_batches = len(dataloader)
        if target == "copy":
            target_idx = 1
        elif target == "mem":
            target_idx = 0
        if resid_pos == "1_1_subject":
            position = 5
        elif resid_pos == "1_1_subject":
            position = 6
        elif resid_pos == "1_1_subject":
            position = 7
        elif resid_pos == "definition":
            position = self.dataset.obj_pos[0]
        elif resid_pos == "2_1_subject":
            position = self.dataset.obj_pos[0] + 2
        elif resid_pos == "2_2_subject":
            position = self.dataset.obj_pos[0] + 3
        elif resid_pos == "2_3_subject":
            position = self.dataset.obj_pos[0] + 4
        else:
            raise ValueError("resid_pos not recognized: should be one of 1_1_subject, 1_2_subject, 1_3_subject, definition, 2_1_subject, 2_2_subject, 2_3_subject")
        topk = 10
        count_target = torch.zeros(( self.model.cfg.n_layers, self.model.cfg.n_heads,  num_batches))
        for idx, batch in tqdm(enumerate(dataloader), total=num_batches, desc=f"OV circuit at all heads", disable=disable_tqdm):
            _, cache = self.model.run_with_cache(batch["corrupted_prompts"])
            for layer in range(self.model.cfg.n_layers):
                for head in range(self.model.cfg.n_heads):
                    residual_stream = cache["resid_post", resid_layer_input]
                    W_OV = (self.model.blocks[layer].attn.W_V @ self.model.blocks[layer].attn.W_O)[head]
                    logit_output = einops.einsum(self.model.W_U, (residual_stream[:,position,:] @ W_OV), "d d_v, b d -> b d_v")
                    logit_output = self.model.ln_final(logit_output)
                    topk_tokens = torch.topk(logit_output, k=topk, dim=-1).indices
                    target_list = [self.model.to_string(batch["target"][i,target_idx]) for i in range(self.batch_size)]
                    
                    count = 0
                    for i in range(self.batch_size):
                        if target_list[i] in [self.model.to_string(topk_tokens[i,j]) for j in range(topk)]:
                            count += 1
                    count_target[layer, head, idx] = count
                    

        total_num_examples = self.batch_size * num_batches
        count_target = einops.reduce(count_target, "l h b -> l h", reduction="sum")
        # percentage
        count_target = 100 * count_target / total_num_examples
        if plot:
            self.plot_heatmap(count_target)
        return count_target       
    
    def plot_heatmap(self, copy_score, xlabel="head", ylabel="layer", x_ticks=None, y_ticks=None):
            import seaborn as sns
            import matplotlib.pyplot as plt
            sns.set()
            sns.set_style("whitegrid", {"axes.grid": False})
            fig, ax = plt.subplots(figsize=(15, 10))
            ax.set_title(f"Copy score for all heads")
            sns.heatmap(copy_score, annot=True, ax=ax, cmap="YlGnBu", fmt=".1f")
            if x_ticks:
                ax.set_xticklabels(x_ticks)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            
    def compute_copy_score_all_heads(self, resid_layer_input, resid_pos, target="copy", plot=False):
        lenghts = self.dataset.get_lengths()
        copy_score = {} # key: length, value: copy score (layer, head)
        for l in lenghts:
            copy_score[l] = self.ov_single_len_all_heads_score(resid_layer_input, resid_pos, l, target=target, plot=False, disable_tqdm=False)
        
        # copy_score is a dict of tensors of shape (n_layers, n_heads). Convert to a tensor of shape (n_layers, n_heads) with the mean of the copy score
        copy_score = torch.stack(list(copy_score.values()), dim=0).mean(dim=0)
        
        if plot:
            self.plot_heatmap(copy_score)
        else:
            return copy_score
    

    def residual_stram_track_target(self,  resid_pos:str, length, target="copy", disable_tqdm=False, plot=False):
        self.dataset.set_len(length, self.model)
        self.dataset.slice_to_fit_batch(self.batch_size)
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        num_batches = len(dataloader)
    
        logit_target = torch.zeros(( self.model.cfg.n_layers, length,  num_batches, self.batch_size))
        for idx, batch in tqdm(enumerate(dataloader), total=num_batches, desc=f"OV circuit at all heads", disable=disable_tqdm):
            _, cache = self.model.run_with_cache(batch["corrupted_prompts"])
            for layer in range(self.model.cfg.n_layers):
                for pos in range(length):
                    residual_stream = cache["resid_post", layer]
                    logit_output = einops.einsum(self.model.W_U, residual_stream[:,pos,:], "d d_v, b d -> b d_v")
                    logit_output = self.model.ln_final(logit_output)
                    mem_logit, cp_logit = to_logit_token(logit_output, batch["target"])
                    if target == "copy":
                        logit_target[layer, pos, idx] = cp_logit.cpu()
                    elif target == "mem":
                        logit_target[layer, pos, idx] = mem_logit.cpu()
        logit_target = einops.rearrange(logit_target, "l p b s -> l p (b s)")
        # compute avg logit_target for each layer accross al the position and examples
        avg_mean = logit_target.mean(dim=-1).mean(dim=-1)
        #mean over all examples
        logit_target = logit_target.mean(dim=-1)
        # for each layer, compute the percentage increase or decrease of logit_target
        for layer in range(self.model.cfg.n_layers):
            logit_target[layer] = -100 * (logit_target[layer] - avg_mean[layer]) / avg_mean[layer]
        print("logit_target.shape", logit_target.shape)
        print("logit_target", logit_target)
        # aggregate for positions
        object_positions = self.dataset.obj_pos[0]
        subject_1_1 = 5
        subject_1_2 = 6
        subject_1_3 = 7
        subject_2_1 = object_positions + 2
        subject_2_2 = object_positions + 3
        subject_2_3 = object_positions + 4
        last_position = length - 1
        object_positions_pre = object_positions - 1
        object_positions_next = object_positions + 1
        
        result_aggregate = torch.zeros((self.model.cfg.n_layers, 14))
        result_aggregate[:,0] = logit_target[:,:subject_1_1].mean(dim=1)
        result_aggregate[:,1] = logit_target[:,subject_1_1]
        result_aggregate[:,2] = logit_target[:,subject_1_2]
        result_aggregate[:,3] = logit_target[:,subject_1_3]
        result_aggregate[:,4] = logit_target[:,subject_1_3+1:object_positions_pre].mean(dim=1)
        result_aggregate[:,5] = logit_target[:,object_positions_pre]
        result_aggregate[:,6] = logit_target[:,object_positions]
        result_aggregate[:,7] = logit_target[:,object_positions_next]
        result_aggregate[:,8] = logit_target[:,subject_2_1]
        result_aggregate[:,9] = logit_target[:,subject_2_2]
        result_aggregate[:,10] = logit_target[:,subject_2_3]
        result_aggregate[:,11] = logit_target[:,subject_2_3+1:last_position-1].mean(dim=1)
        result_aggregate[:,12] = logit_target[:,last_position-1]
        result_aggregate[:,13] = logit_target[:,last_position]
        
        self.plot_heatmap(
            result_aggregate,
            xlabel="position",
            ylabel="layer",
            x_ticks=["--", "1_1", "1_2", "1_3", "--", "o_pre", "o", "o_next", "2_1", "2_2", "2_3", "--", "l_pre", "last"],
        )
        
        return result_aggregate    