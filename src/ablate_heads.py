from src.dataset import TlensDataset
from src.model import WrapHookedTransformer
import einops
import torch
from tqdm import tqdm
from functools import partial
from copy import deepcopy

torch.set_grad_enabled(False)


def to_logit_token(logit, target, normalize="logsoftmax"):
    assert len(logit.shape) in [2,
                                3], "logit should be of shape (batch_size, d_vocab) or (batch_size, seq_len, d_vocab)"
    if len(logit.shape) == 3:
        logit = logit[:, -1, :]  # batch_size, d_vocab
    if normalize == "logsoftmax":
        logit = torch.log_softmax(logit, dim=-1)
    elif normalize == "softmax":
        logit = torch.softmax(logit, dim=-1)
    elif normalize == "none":
        pass
    logit_mem = torch.zeros(target.shape[0])
    logit_cp = torch.zeros(target.shape[0])
    for i in range(target.shape[0]):
        logit_mem[i] = logit[i, target[i, 0]]
        logit_cp[i] = logit[i, target[i, 1]]
    return logit_mem, logit_cp


class BaseExperiment():
    def __init__(self, dataset: TlensDataset, model: WrapHookedTransformer, batch_size, filter_outliers=False):
        self.dataset = dataset
        self.model = model
        self.model.eval()
        self.batch_size = batch_size
        self.filter_outliers = filter_outliers

    def set_len(self, length, slice_to_fit_batch=True):
        self.dataset.set_len(length, self.model)
        if self.filter_outliers:
            self._filter_outliers()
        elif slice_to_fit_batch:
            print("WARNING: slicing to fit batch")
            self.dataset.slice_to_fit_batch(self.batch_size)
        

    def _filter_outliers(self, save_filtered=False):
        print("save filtered:", save_filtered)
        print("Number of examples before outliers:", len(self.dataset))
        clean_logit, corrupted_logit, target = self.compute_logit()
        # clean_logit_mem, clean_logit_cp = to_logit_token(clean_logit, target)
        corrupted_logit_mem, corrupted_logit_cp = to_logit_token(corrupted_logit, target)

        outliers_under = torch.where(corrupted_logit_mem < (corrupted_logit_mem.mean() - corrupted_logit_mem.std()))[0]
        outliers_over = torch.where(corrupted_logit_cp > (corrupted_logit_cp.mean() + corrupted_logit_cp.std()))[0]
        outliers_indexes = torch.cat([outliers_under, outliers_over], dim=0).tolist()

        maxdatasize = ((len(self.dataset) - len(outliers_indexes)) // self.batch_size) * self.batch_size

        self.dataset.filter_from_idx(outliers_indexes, exclude=True, save_filtered=save_filtered)
        self.dataset.slice(maxdatasize)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        self.num_batches = len(self.dataloader)
        print("Number of examples after outliers:", len(self.dataloader) * self.batch_size)

    def compute_logit(self):
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        clean_logit = []
        corrupted_logit = []
        target = []
        for batch in tqdm(dataloader):
            # clean_logit.append(self.model(batch["clean_prompts"])[:,-1,:].cpu())
            corrupted_logit.append(self.model(batch["corrupted_prompts"])[:, -1, :].cpu())
            target.append(batch["target"].cpu())
        # clean_logit = torch.cat(clean_logit, dim=0)
        corrupted_logit = torch.cat(corrupted_logit, dim=0)
        target = torch.cat(target, dim=0)
        return clean_logit, corrupted_logit, target

    def get_batch(self, len, **kwargs):
        self.set_len(len, **kwargs)
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        return next(iter(self.dataloader))

    def get_position(self, resid_pos: str):
        positions = {
            "1_1_subject": 5,
            "1_2_subject": 6,
            "1_3_subject": 7,
            "definition": self.dataset.obj_pos[0],
            "2_1_subject": self.dataset.obj_pos[0] + 2,
            "2_2_subject": self.dataset.obj_pos[0] + 3,
            "2_3_subject": self.dataset.obj_pos[0] + 4,
            "last_pre": -2,
            "last": -1,
        }
        return positions.get(resid_pos, ValueError(
            "resid_pos not recognized: should be one of 1_1_subject, 1_2_subject, 1_3_subject, definition, 2_1_subject, 2_2_subject, 2_3_subject"))


class Ablate(BaseExperiment):
    def __init__(self, dataset: TlensDataset, model: WrapHookedTransformer, batch_size, filter_outliers=False):
        super().__init__(dataset, model, batch_size, filter_outliers)

    def create_dataloader(self, filter_outliers=False, **kwargs):
        if filter_outliers:
            print(self._filter_outliers)
            self._filter_outliers(**kwargs)
        else:
            self.slice_to_fit_batch()

    def slice_to_fit_batch(self):
        self.dataset.slice_to_fit_batch(self.batch_size)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        self.num_batches = len(self.dataloader)
        print("Number of examples after slicing:", len(self.dataloader) * self.batch_size)

    def get_normalize_metric(self):
        clean_logit, corrupted_logit, target = self.compute_logit()
        # clean_logit_mem, clean_logit_cp = to_logit_token(clean_logit, target)
        corrupted_logit_mem, corrupted_logit_cp = to_logit_token(corrupted_logit, target)

        def normalize_logit_token(logit_mem, logit_cp, baseline="corrupted", ):
            # logit_mem, logit_cp = to_logit_token(logit, target)
            # percentage increase or decrease of logit_mem
            if baseline == "clean":
                pass
                # logit_mem = 100 * (logit_mem - clean_logit_mem) / clean_logit_mem
                # # percentage increase or decrease of logit_cp
                # logit_cp = 100 * (logit_cp - clean_logit_cp) / clean_logit_cp
                return -logit_mem, -logit_cp
            elif baseline == "corrupted":
                logit_mem = 100 * (logit_mem - corrupted_logit_mem) / corrupted_logit_mem
                # percentage increase or decrease of logit_cp
                logit_cp = 100 * (logit_cp - corrupted_logit_cp) / corrupted_logit_cp
                return -logit_mem, -logit_cp
            elif baseline == "base_logit":
                return corrupted_logit_mem, corrupted_logit_cp

        return normalize_logit_token
    
    
    
    def ablate_heads(self, return_type="diff"):
        normalize_logit_token = self.get_normalize_metric()
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        self.num_batches = len(self.dataloader)

        def heads_hook(activation, hook, head, pos1=None, pos2=None):
            activation[:, head, -1, :] = 0
            return activation

        def freeze_hook(activation, hook, clean_activation, pos1=None, pos2=None):
            activation = clean_activation
            return activation


        examples_mem_list = [[[] for _ in range(self.model.cfg.n_heads)] for _ in range(self.model.cfg.n_layers)]
        examples_cp_list =  [[[] for _ in range(self.model.cfg.n_heads)] for _ in range(self.model.cfg.n_layers)]

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
                    )[:, -1, :]
                    mem, cp = to_logit_token(logit, batch["target"])

                    examples_mem_list[layer][head].append(mem.cpu())
                    examples_cp_list[layer][head].append(cp.cpu())
                # remove the hooks for the previous layer

                hooks.pop(f"L{layer}")
        
        for layer in range(self.model.cfg.n_layers):
            for head in range(self.model.cfg.n_heads):
                examples_mem_list[layer][head] = torch.cat(examples_mem_list[layer][head], dim=0)
                examples_cp_list[layer][head] = torch.cat(examples_cp_list[layer][head], dim=0)
        
        flattened_mem = [tensor for layer in examples_mem_list for head in layer for tensor in head]
        flattened_cp = [tensor for layer in examples_cp_list for head in layer for tensor in head]
        
        examples_mem = torch.stack(flattened_mem).view(self.model.cfg.n_layers, self.model.cfg.n_heads, -1)
        examples_cp = torch.stack(flattened_cp).view(self.model.cfg.n_layers, self.model.cfg.n_heads, -1)

        if return_type == "diff":
            # examples_mem = (einops.einsum(examples_mem, -examples_cp, "l h b, l h b -> l h b")).abs()
            examples_mem = (examples_mem -examples_cp).abs()
            examples_cp = examples_mem
            base_logit_mem, base_logit_cp = normalize_logit_token(examples_mem, examples_cp, baseline="base_logit")
            base_diff = (base_logit_mem - base_logit_cp).abs()
            #percentage increade/decrease
            examples_cp =  (examples_cp - base_diff) 
            examples_mem = (examples_mem - base_diff) 
            
            return examples_mem, examples_cp
        
        base_logit_mem, base_logit_cp = normalize_logit_token(examples_mem, examples_cp, baseline="corrupted")


        return base_logit_mem, base_logit_cp
    
    def count_mem_win(self, mem, cp):
        count = 0
        for i in range(len(mem)):
            if mem[i] > cp[i]:
                count += 1
        print("Number of examples where mem > cp:", count)
        return count
    
    def ablate_attn_block(self):
        normalize_logit_token = self.get_normalize_metric()
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        self.num_batches = len(self.dataloader)

        def attn_hook(activation, hook,  pos1=None, pos2=None):
            activation[:,-1, :] = 0
            return activation

        def freeze_hook(activation, hook, clean_activation, pos1=None, pos2=None):
            activation = clean_activation
            return activation

        # examples_mem = torch.zeros((self.model.cfg.n_layers, self.num_batches, self.batch_size),
        #                            device="cpu")
        # examples_cp = torch.zeros((self.model.cfg.n_layers, self.num_batches, self.batch_size),
        #                           device="cpu")
        examples_mem_list = [[] for _ in range(self.model.cfg.n_layers)]
        examples_cp_list = [[] for _ in range(self.model.cfg.n_layers)]


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
                tmp_hooks = deepcopy(hooks)
                tmp_hooks[f"L{layer}"] = (f"blocks.{layer}.hook_attn_out",
                                            partial(
                                                attn_hook,
                                            )
                                            )

                list_hooks = list(tmp_hooks.values())
                self.model.reset_hooks()
                logit = self.model.run_with_hooks(
                    batch["corrupted_prompts"],
                    fwd_hooks=list_hooks,
                )[:, -1, :]
                if logit.shape[0] != self.batch_size:
                    print("Ops, the batch size is not correct")
                mem, cp = to_logit_token(logit, batch["target"])
                # norm_mem, norm_cp = normalize_logit_token(mem, cp, baseline="corrupted")
                examples_mem_list[layer].append(mem.cpu())
                examples_cp_list[layer].append(cp.cpu())
            # remove the hooks for the previous layer

                hooks.pop(f"L{layer}")

        for layer in range(self.model.cfg.n_layers):
            examples_mem_list[layer] = torch.cat(examples_mem_list[layer], dim=0)
            examples_cp_list[layer] = torch.cat(examples_cp_list[layer], dim=0)
            
        examples_mem = torch.stack(examples_mem_list).view(self.model.cfg.n_layers, -1)
        examples_cp = torch.stack(examples_cp_list).view(self.model.cfg.n_layers, -1)
        
        base_logit_mem, base_logit_cp = normalize_logit_token(examples_mem, examples_cp, baseline="corrupted")
        
        
        # for layer in range(self.model.cfg.n_layers):
        #     for head in range(self.model.cfg.n_heads):
        #         norm_mem, norm_cp, = normalize_logit_token(examples_mem[layer, head, :], examples_cp[layer, head, :],
        #                                                   baseline="raw")
        #         examples_mem[layer, head, :] = norm_mem
        #         examples_cp[layer, head, :] = norm_cp

        return base_logit_mem, base_logit_cp


class AblateMultiLen:
    def __init__(self, dataset: TlensDataset, model: WrapHookedTransformer, batch_size):
        self.model = model
        self.batch_size = batch_size
        self.ablate = Ablate(dataset, model, batch_size)

    def ablate_single_len(self, length, filter_outliers=False, ablate_target="head", **kwargs):
        self.ablate.set_len(length, slice_to_fit_batch=False)
        # n_samples = len(self.ablate.dataset)
        # if n_samples  < self.batch_size:
        #     return None, None
        # self.ablate.create_dataloader(filter_outliers=filter_outliers, **kwargs)
        if ablate_target == "head":
            return self.ablate.ablate_heads()
        if ablate_target == "attn":
            return self.ablate.ablate_attn_block()

    def ablate_multi_len(self, filter_outliers=False, **kwargs):
        lenghts = self.ablate.dataset.get_lengths()
        # lenghts = [11]
        result_cp_per_len = {}
        result_mem_per_len = {}
        result_cp_base_per_len = {}
        result_mem_base_per_len = {}
        for l in lenghts:
            print("Ablating examples of length", l, "...")
            mem, cp = self.ablate_single_len(l, filter_outliers=filter_outliers, **kwargs)
            if mem != None and cp != None:
                result_mem_per_len[l], result_cp_per_len[l] = mem, cp
        # concatenate the results
        result_cp = torch.cat(list(result_cp_per_len.values()), dim=-1)
        result_mem = torch.cat(list(result_mem_per_len.values()), dim=-1)

        print("result_cp.shape", result_cp.shape)
        
        
        return result_mem, result_cp
        
        
        # return result_mem, result_cp, result_mem_base, result_cp_base


class OVCircuit(BaseExperiment):
    def __init__(self, model: WrapHookedTransformer, dataset: TlensDataset, batch_size, filter_outliers=False):
        super().__init__(dataset, model, batch_size, filter_outliers)

    def ov_single_len(self, resid_layer_input, resid_pos: str, layer, head, length, disable_tqdm=False):
        self.set_len(length)
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        num_batches = len(dataloader)
        position = self.get_position(resid_pos)
        logit_ov = torch.zeros((num_batches, self.batch_size, self.model.cfg.d_vocab))
        
        for idx, batch in tqdm(enumerate(dataloader), total=num_batches, desc=f"OV circuit at ({layer},{head})",
                               disable=disable_tqdm):
            _, cache = self.model.run_with_cache(batch["corrupted_prompts"])
            residual_stream = cache["resid_post", resid_layer_input]
            W_OV = (self.model.blocks[layer].attn.W_V @ self.model.blocks[layer].attn.W_O)[head]
            logit_output = einops.einsum(self.model.W_U, (residual_stream[:, position, :] @ W_OV),
                                         "d d_v, b d -> b d_v")
            logit_ov[idx] = self.model.ln_final(logit_output)
        logit_ov = einops.rearrange(logit_ov, "b s d -> (b s) d")
        return logit_ov

    def ov_single_copy_score(self, resid_layer_input, resid_pos: str, layer, head, length, target="copy",
                             disable_tqdm=False):
        target_idx = 1 if target == "copy" else 0
        logit_ov = self.ov_single_len(resid_layer_input, resid_pos, layer, head, length, disable_tqdm=disable_tqdm)
        num_examples = logit_ov.shape[0]
        topk = 10
        topk_tokens = torch.topk(logit_ov, k=topk, dim=-1).indices
        target_list = [self.model.to_string(self.dataset.target[i, target_idx]) for i in range(num_examples)]
        count = sum([1 for i in range(num_examples) if
                     target_list[i] in [self.model.to_string(topk_tokens[i, j]) for j in range(topk)]])
        return 100 * count / num_examples

    def ov_multi_len(self, resid_layer_input, resid_pos: str, layer, head, disable_tqdm=False):
        lenghts = self.dataset.get_lengths()
        logit = {l: self.ov_single_len(resid_layer_input, resid_pos, layer, head, l, disable_tqdm=disable_tqdm) for l in
                 lenghts}
        return torch.cat(list(logit.values()), dim=0)

    def ov_multi_copy_score(self, resid_layer_input, resid_pos, layer, head, target="copy", disable_tqdm=False):
        lenghts = self.dataset.get_lengths()
        copy_score = {l: self.ov_single_copy_score(resid_layer_input, resid_pos, layer, head, l, target=target,
                                                   disable_tqdm=disable_tqdm) for l in lenghts}
        return torch.tensor(list(copy_score.values())).mean().item()

    def ov_single_len_all_heads_score(self, resid_layer_input, resid_pos: str, length, target="copy",
                                      disable_tqdm=False, plot=False, logit_score=False, resid_read="resid_post"):
        assert target in ["copy", "mem"], "target should be one of copy or mem"
        assert resid_pos in ["o_pre", "last_pre", "1_1_subject", "1_2_subject", "1_3_subject", "definition",
                             "2_1_subject", "2_2_subject",
                             "2_3_subject"], "resid_pos should be one of 1_1_subject, 1_2_subject, 1_3_subject, definition, 2_1_subject, 2_2_subject, 2_3_subject"
        self.set_len(length)
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        num_batches = len(dataloader)
        if num_batches == 0:
            return None
        
        target_idx = 1 if target == "copy" else 0
        position = self.get_position(resid_pos)
        if logit_score == False:
            topk = 10
            count_target = torch.zeros((self.model.cfg.n_layers, self.model.cfg.n_heads, num_batches))
            for idx, batch in tqdm(enumerate(dataloader), total=num_batches, desc=f"OV circuit at all heads",
                                   disable=disable_tqdm):
                _, cache = self.model.run_with_cache(batch["corrupted_prompts"])
                for layer in range(self.model.cfg.n_layers):
                    for head in range(self.model.cfg.n_heads):
                        residual_stream = cache[resid_read, resid_layer_input]
                        W_OV = (self.model.blocks[layer].attn.W_V @ self.model.blocks[layer].attn.W_O)[head]
                        logit_output = einops.einsum(self.model.W_U, (residual_stream[:, position, :] @ W_OV),
                                                     "d d_v, b d -> b d_v")
                        logit_output = self.model.ln_final(logit_output)
                        topk_tokens = torch.topk(logit_output, k=topk, dim=-1).indices
                        target_list = [self.model.to_string(batch["target"][i, target_idx]) for i in
                                       range(self.batch_size)]
                        count = sum([1 for i in range(self.batch_size) if
                                     target_list[i] in [self.model.to_string(topk_tokens[i, j]) for j in range(topk)]])
                        count_target[layer, head, idx] = count
            total_num_examples = self.batch_size * num_batches
            count_target = einops.reduce(count_target, "l h b -> l h", reduction="sum")
            count_target = 100 * count_target / total_num_examples
            if plot:
                self.plot_heatmap(count_target)
            return count_target

        if logit_score == True:
            logit_target = torch.zeros((self.model.cfg.n_layers, self.model.cfg.n_heads, num_batches, self.batch_size))
            for idx, batch in tqdm(enumerate(dataloader), total=num_batches, desc=f"OV circuit at all heads",
                                   disable=disable_tqdm):
                _, cache = self.model.run_with_cache(batch["corrupted_prompts"])
                for layer in range(self.model.cfg.n_layers):
                    for head in range(self.model.cfg.n_heads):
                        residual_stream = cache[resid_read, resid_layer_input]
                        W_OV = (self.model.blocks[layer].attn.W_V @ self.model.blocks[layer].attn.W_O)[head]
                        logit_output = einops.einsum(self.model.W_U, (residual_stream[:, position, :] @ W_OV),
                                                     "d d_v, b d -> b d_v")
                        logit_output = self.model.ln_final(logit_output)
                        mem_logit, cp_logit = to_logit_token(logit_output, batch["target"])
                        if target == "copy":
                            logit_target[layer, head, idx] = cp_logit.cpu()
                        elif target == "mem":
                            logit_target[layer, head, idx] = mem_logit.cpu()
            logit_target = einops.rearrange(logit_target, "l h b s -> l h (b s)")
            avg_mean = logit_target.mean(dim=-1).mean(dim=-1)
            logit_target_std = logit_target.std(dim=-1)
            logit_target = logit_target.mean(dim=-1)
            for layer in range(self.model.cfg.n_layers):
                logit_target[layer] = -100 * (logit_target[layer] - avg_mean[layer]) / avg_mean[layer]
            if plot:
                self.plot_heatmap(logit_target)
                self.plot_heatmap(logit_target_std)
            return logit_target

    def plot_heatmap(self, copy_score, xlabel="head", ylabel="layer", x_ticks=None, y_ticks=None, title="none"):
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set()
        sns.set_style("whitegrid", {"axes.grid": False})
        fig, ax = plt.subplots(figsize=(13, 8))
        ax.set_title(title)
        sns.heatmap(copy_score, annot=True, ax=ax, cmap="RdBu_r", fmt=".1f", center=0)
        if x_ticks:
            ax.set_xticklabels(x_ticks)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.show()

    def compute_copy_score_all_heads(self, resid_layer_input, resid_pos, target="copy", plot=False, logit_score=False,
                                     **kwargs):
        lenghts = self.dataset.get_lengths()
        copy_score = {l: self.ov_single_len_all_heads_score(resid_layer_input, resid_pos, l, target=target, plot=False,
                                                            disable_tqdm=False, logit_score=logit_score, **kwargs) for l
                      in lenghts}
        # remove the None values
        copy_score = {k: v for k, v in copy_score.items() if v is not None}
        copy_score = torch.stack(list(copy_score.values()), dim=0).mean(dim=0)
        if plot:
            copy_score[copy_score < 0] = 0
            self.plot_heatmap(copy_score)
        return copy_score

    def aggregate_result(self, object_positions, logit_target, length):
        subject_1_1 = 5
        subject_1_2 = 6 if length > 15 else 5
        subject_1_3 = 7 if length > 17 else subject_1_2
        subject_2_1 = object_positions + 2
        subject_2_2 = object_positions + 3 if length > 15 else subject_2_1
        subject_2_3 = object_positions + 4 if length > 17 else subject_2_2
        subject_2_2 = subject_2_2 if subject_2_2 < length else subject_2_1
        subject_2_3 = subject_2_3 if subject_2_3 < length else subject_2_2
        last_position = length - 1
        object_positions_pre = object_positions - 1
        object_positions_next = object_positions + 1
        result_aggregate = torch.zeros((self.model.cfg.n_layers, 13))
        result_aggregate[:, 0] = logit_target[:, :subject_1_1].mean(dim=1)
        result_aggregate[:, 1] = logit_target[:, subject_1_1]
        result_aggregate[:, 2] = logit_target[:, subject_1_2]
        result_aggregate[:, 3] = logit_target[:, subject_1_3]
        result_aggregate[:, 4] = logit_target[:, subject_1_3 + 1:object_positions_pre].mean(dim=1)
        result_aggregate[:, 5] = logit_target[:, object_positions_pre]
        result_aggregate[:, 6] = logit_target[:, object_positions]
        result_aggregate[:, 7] = logit_target[:, object_positions_next]
        result_aggregate[:, 8] = logit_target[:, subject_2_1]
        result_aggregate[:, 9] = logit_target[:, subject_2_2]
        result_aggregate[:, 10] = logit_target[:, subject_2_3]

        result_aggregate[:, 11] = logit_target[:, subject_2_3 + 1:last_position ].mean(dim=1)
        # result_aggregate[:, 12] = logit_target[:, last_position - 1]
        result_aggregate[:, 12] = logit_target[:, last_position]
        return result_aggregate

    def residual_stram_track_target(self, length, target="copy", disable_tqdm=False, plot=False):
        assert target in ["copy", "mem"], "target should be one of copy or mem"
        self.set_len(length, slice_to_fit_batch=False)
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        num_batches = len(dataloader)

        # logit_target = torch.zeros((self.model.cfg.n_layers, length, num_batches, self.batch_size), device="cpu")
        logit_target_list = [[[] for _ in range(length)] for _ in range(self.model.cfg.n_layers)]
        if num_batches == 0:
            return None
        for idx, batch in tqdm(enumerate(dataloader), total=num_batches, desc=f"OV circuit at all heads {length}",
                               disable=disable_tqdm):
            _, cache = self.model.run_with_cache(batch["corrupted_prompts"])
            for layer in range(self.model.cfg.n_layers):
                for pos in range(length):
                    residual_stream = cache["resid_post", layer]
                    logit_output = einops.einsum(self.model.W_U, residual_stream[:, pos, :], "d d_v, b d -> b d_v")
                    logit_output = self.model.ln_final(logit_output)
                    mem_logit, cp_logit = to_logit_token(logit_output, batch["target"])
                    if target == "copy":
                        logit_target_list[layer][pos].append(cp_logit.cpu())
                        # logit_target[layer, pos, idx] = cp_logit.cpu()
                    elif target == "mem":
                        logit_target_list[layer][pos].append(mem_logit.cpu())
                        # logit_target[layer, pos, idx] = mem_logit.cpu()
                        
        for layer in range(self.model.cfg.n_layers):
            for pos in range(length):
                logit_target_list[layer][pos] = torch.cat(logit_target_list[layer][pos], dim=0)
        flattened_logit_target = [tensor for layer in logit_target_list for pos in layer for tensor in pos]
        logit_target = torch.stack(flattened_logit_target).view(self.model.cfg.n_layers, length, -1)
        
        # logit_target = einops.rearrange(logit_target, "l p b s -> l p (b s)")
        # compute avg logit_target for each layer accross al the position and examples
        avg_mean = logit_target.mean(dim=-1).mean(dim=-1)
        # mean over all examples
        logit_target = logit_target.mean(dim=-1)
        # for each layer, compute the percentage increase or decrease of logit_target
        for layer in range(self.model.cfg.n_layers):
            logit_target[layer] = -100 * (logit_target[layer] - avg_mean[layer]) / avg_mean[layer] + 1e-6
        # aggregate for positions
        object_positions = self.dataset.obj_pos[0]

        result_aggregate = self.aggregate_result(object_positions, logit_target, length)

        if plot:
            self.plot_heatmap(
                result_aggregate,
                xlabel="position",
                ylabel="layer",
                x_ticks=["--", "1_1", "1_2", "1_3", "--", "o_pre", "o", "o_next", "2_1", "2_2", "2_3", "--", "l_pre",
                         "last"],
            )

        return result_aggregate

    def residual_stram_track_target_all_len(self, target="copy", disable_tqdm=False, plot=False):
        lenghts = self.dataset.get_lengths()
        result = {}
        for l in lenghts:
            residual = self.residual_stram_track_target(length=l, target=target, disable_tqdm=disable_tqdm, plot=False)
            if residual is not None:
                result[l] = residual
        result_score = torch.stack(list(result.values()), dim=0).mean(dim=0)
        if plot:
            try:
                self.plot_heatmap(
                    result_score,
                    xlabel="position",
                    ylabel="layer",
                    x_ticks=["--", "1_1", "1_2", "1_3", "--", "o_pre", "o", "o_next", "2_1", "2_2", "2_3", "--",
                             "l_pre", "last"],
                    title="Percentage increase or decrease of logit_target respect to the avg per layer"
                )
            except:
                print("Error in plotting, probably due to Matplotlib version (3.5.3 should work)")

        return result_score


class QK_circuit(BaseExperiment):
    def __init__(self, model: WrapHookedTransformer, dataset: TlensDataset, batch_size, filter_outliers=False):
        super().__init__(dataset, model, batch_size, filter_outliers)

    def qk_single_len(self, layer, head, length, disable_tqdm=False):
        self.set_len(length)
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        num_batches = len(dataloader)

        attn_score = torch.zeros((num_batches, self.batch_size))
        mem_logit = torch.zeros((num_batches, self.batch_size))
        cp_logit = torch.zeros((num_batches, self.batch_size))

        W_QK = self.model.blocks[layer].attn.W_Q[head] @ self.model.blocks[layer].attn.W_K[head].T
        for idx, batch in tqdm(enumerate(dataloader), total=num_batches, desc=f"QK circuit at ({layer},{head})",
                               disable=disable_tqdm):
            logit, cache = self.model.run_with_cache(batch["corrupted_prompts"])

            object_token = self.model.to_tokens(batch["corrupted_prompts"])[:, self.dataset.obj_pos[0]]  # batch_size, 1
            # logit of the head
            output_weight_head = self.model.blocks[layer].attn.W_O[head]
            logit_head = einops.einsum(output_weight_head, cache[f"blocks.{layer}.attn.hook_z"][:, :, head, :],
                                       "d_h d, b p d_h -> b p d")

            # W_OV = (self.model.blocks[layer].attn.W_V @ self.model.blocks[layer].attn.W_O)[head]
            logit_last = einops.einsum(self.model.W_U, logit_head[:, -1, :], "d d_v, b d -> b d_v")

            mem_logit[idx], cp_logit[idx] = to_logit_token(logit_last, batch["target"])

            # subject token
            subject_token = self.model.to_tokens(batch["corrupted_prompts"])[:, 5]  # batch_size, 1
            attn_score[idx] = cache["pattern", layer][:, head, -1, object_token[0]]  # batch_size
            # attn_score[idx] = torch.diag(torch.matmul(W_QK, self.model.W_E.T)[:,subject_token][object_token,:]) # batch_size

        attn_score = einops.rearrange(attn_score, "b s -> (b s)")
        mem_logit = einops.rearrange(mem_logit, "b s -> (b s)")
        cp_logit = einops.rearrange(cp_logit, "b s -> (b s)")
        return attn_score, mem_logit, cp_logit


class Investigate_single_head(BaseExperiment):
    def __init__(self, model: WrapHookedTransformer, dataset: TlensDataset, batch_size, filter_outliers=False):
        super().__init__(dataset, model, batch_size, filter_outliers)

    def get_logit_target_single_head_single_len(self, layer, head, length):
        self.set_len(length)
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        num_batches = len(dataloader)

        logit_mem = torch.zeros((num_batches, self.batch_size))
        logit_cp = torch.zeros((num_batches, self.batch_size))
        logit_mem_subj = torch.zeros((num_batches, self.batch_size))
        logit_cp_subj = torch.zeros((num_batches, self.batch_size))
        for idx, batch in enumerate(dataloader):
            logit, cache = self.model.run_with_cache(batch["corrupted_prompts"])
            residual_stream = cache["resid_pre", layer]
            W_OV = (self.model.blocks[layer].attn.W_V @ self.model.blocks[layer].attn.W_O)[head]
            logit_obj = einops.einsum(self.model.W_U, (residual_stream[:, -2, :] @ W_OV), "d d_v, b d -> b d_v")[:, :]
            logit_subj = einops.einsum(self.model.W_U, (residual_stream[:, 5, :] @ W_OV), "d d_v, b d -> b d_v")[:, :]

            logit_obj = self.model.ln_final(logit_obj)
            logit_subj = self.model.ln_final(logit_subj)
            logit_mem[idx], logit_cp[idx] = to_logit_token(logit_obj, batch["target"])
            logit_mem_subj[idx], logit_cp_subj[idx] = to_logit_token(logit_subj, batch["target"])
        logit_mem = einops.rearrange(logit_mem, "b s -> (b s)")
        logit_cp = einops.rearrange(logit_cp, "b s -> (b s)")
        logit_mem_subj = einops.rearrange(logit_mem_subj, "b s -> (b s)")
        logit_cp_subj = einops.rearrange(logit_cp_subj, "b s -> (b s)")

        # # percentage increase or decrease of logit_mem
        # logit_mem = 100 * (logit_mem - logit_mem_resid) / logit_mem_resid
        # # percentage increase or decrease of logit_cp
        # logit_cp = 100 * (logit_cp - logit_mem_subj) / logit_mem_subj

        return logit_mem, logit_cp, logit_mem_subj, logit_cp_subj

    def get_logit_target_single_head(self, layer, head):
        lenghts = self.dataset.get_lengths()
        logit_mem = {}
        logit_cp = {}
        logit_mem_resid = {}
        logit_cp_resid = {}
        for l in lenghts:
            logit_mem[l], logit_cp[l], logit_mem_resid[l], logit_cp_resid[
                l] = self.get_logit_target_single_head_single_len(layer, head, l)

        logit_mem = torch.cat(list(logit_mem.values()), dim=0)
        logit_cp = torch.cat(list(logit_cp.values()), dim=0)
        logit_mem_resid = torch.cat(list(logit_mem_resid.values()), dim=0)
        logit_cp_resid = torch.cat(list(logit_cp_resid.values()), dim=0)
        return logit_mem, logit_cp, logit_mem_resid, logit_cp_resid

    def get_attn_score_per_len(self, lenght):
        self.set_len(lenght)
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        num_batches = len(dataloader)
        object_position = self.dataset.obj_pos[0]
        attn_score_obj = torch.zeros((self.model.cfg.n_layers, self.model.cfg.n_heads, num_batches, self.batch_size))
        attn_score_subj = torch.zeros((self.model.cfg.n_layers, self.model.cfg.n_heads, num_batches, self.batch_size))

        for idx, batch in tqdm(enumerate(dataloader), total=num_batches, desc="Attention score"):
            _, cache = self.model.run_with_cache(batch["corrupted_prompts"])
            for layer in range(self.model.cfg.n_layers):
                for head in range(self.model.cfg.n_heads):
                    attn_score_obj[layer, head, idx] = cache["pattern", layer][:, head, -1, object_position]
                    attn_score_subj[layer, head, idx] = cache["pattern", layer][:, head, -1, 5] + cache[
                                                                                                      "pattern", layer][
                                                                                                  :, head, -1,
                                                                                                  object_position + 2]

        attn_score_obj = einops.rearrange(attn_score_obj, "l h b s -> l h (b s)")
        attn_score_subj = einops.rearrange(attn_score_subj, "l h b s -> l h (b s)")
        return attn_score_obj, attn_score_subj


class LogitLens(BaseExperiment):
    def __init__(self, model: WrapHookedTransformer, dataset: TlensDataset, batch_size, filter_outliers=False):
        super().__init__(dataset, model, batch_size, filter_outliers)

    def logit_lens_single_len(self, l, plot=False):
        self.set_len(l)
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        num_batches = len(dataloader)

        logit_mem = torch.zeros((self.model.cfg.n_layers, num_batches, self.batch_size))
        logit_cp = torch.zeros((self.model.cfg.n_layers, num_batches, self.batch_size))

        for idx, batch in tqdm(enumerate(dataloader), total=num_batches, desc=f"Logit lens at all layers",
                               disable=False):
            logit, cache = self.model.run_with_cache(batch["corrupted_prompts"])
            for layer in range(self.model.cfg.n_layers):
                residual_at_layer = cache["resid_pre", layer]
                logit_at_layer = einops.einsum(self.model.W_U, residual_at_layer[:, -1, :], "d d_v, b d -> b d_v")
                logit_at_layer = self.model.ln_final(logit_at_layer)
                mem_logit, cp_logit = to_logit_token(logit_at_layer, batch["target"], normalize="logsoftmax")
                logit_mem[layer, idx] = mem_logit.cpu()
                logit_cp[layer, idx] = cp_logit.cpu()

        logit_mem = einops.rearrange(logit_mem, "l b s -> l (b s)")
        logit_cp = einops.rearrange(logit_cp, "l b s -> l (b s)")
        # count the examples where the logit_mem is greater than the logit_cp for layer greater the 6 and print the percentage
        print("logit_mem > logit_cp", (logit_mem[2] > logit_cp[2]).sum().item() / logit_mem[2].numel())

        if plot:
            mem_mean_per_layer = logit_mem.mean(dim=-1)  # (l,1)
            cp_mean_per_layer = logit_cp.mean(dim=-1)
            mem_std_per_layer = logit_mem.std(dim=-1)
            cp_std_per_layer = logit_cp.std(dim=-1)

            # plot
            import matplotlib.pyplot as plt
            plt.figure(figsize=(13, 8))
            plt.title("Logit lens")
            plt.plot(mem_mean_per_layer.detach().cpu().numpy(), label="mem")
            plt.fill_between(range(self.model.cfg.n_layers),
                             mem_mean_per_layer.detach().cpu().numpy() - mem_std_per_layer.detach().cpu().numpy(),
                             mem_mean_per_layer.detach().cpu().numpy() + mem_std_per_layer.detach().cpu().numpy(),
                             alpha=0.2)
            plt.plot(cp_mean_per_layer.detach().cpu().numpy(), label="cp")
            plt.fill_between(range(self.model.cfg.n_layers),
                             cp_mean_per_layer.detach().cpu().numpy() - cp_std_per_layer.detach().cpu().numpy(),
                             cp_mean_per_layer.detach().cpu().numpy() + cp_std_per_layer.detach().cpu().numpy(),
                             alpha=0.2)
            plt.legend()
            plt.show()
        return logit_mem, logit_cp

    def logit_lens_multi_len(self, plot=False):
        lenghts = self.dataset.get_lengths()
        logit_mem = {}
        logit_cp = {}
        for l in lenghts:
            logit_mem[l], logit_cp[l] = self.logit_lens_single_len(l, plot=False)

        logit_mem = torch.cat(list(logit_mem.values()), dim=-1)
        logit_cp = torch.cat(list(logit_cp.values()), dim=-1)
        if plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(13, 8))
            mem_mean_per_layer = logit_mem.mean(dim=-1)
            cp_mean_per_layer = logit_cp.mean(dim=-1)
            mem_std_per_layer = logit_mem.std(dim=-1)
            cp_std_per_layer = logit_cp.std(dim=-1)
            plt.plot(mem_mean_per_layer.detach().cpu().numpy(), label="mem")
            plt.fill_between(range(self.model.cfg.n_layers),
                             mem_mean_per_layer.detach().cpu().numpy() - mem_std_per_layer.detach().cpu().numpy(),
                             mem_mean_per_layer.detach().cpu().numpy() + mem_std_per_layer.detach().cpu().numpy(),
                             alpha=0.2)
            plt.plot(cp_mean_per_layer.detach().cpu().numpy(), label="cp")
            plt.fill_between(range(self.model.cfg.n_layers),
                             cp_mean_per_layer.detach().cpu().numpy() - cp_std_per_layer.detach().cpu().numpy(),
                             cp_mean_per_layer.detach().cpu().numpy() + cp_std_per_layer.detach().cpu().numpy(),
                             alpha=0.2)
            plt.legend()
            plt.show()
        return logit_mem, logit_cp


class ResidCorrelation(BaseExperiment):
    def __init__(self, model: WrapHookedTransformer, dataset: TlensDataset, batch_size, filter_outliers=False):
        super().__init__(dataset, model, batch_size, filter_outliers)

    def get_logit_single_len(self, length, layer=0, component="resid_post", position="o_pre"):
        self.set_len(length)
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        num_batches = len(dataloader)
        if num_batches == 0:
            return None, None
        position = self.get_position(position)
        # position = -1

        final_logit = torch.zeros((num_batches, self.batch_size, 2))
        position_logit = torch.zeros((num_batches, self.batch_size, 2))

        for idx, batch in tqdm(enumerate(dataloader), total=num_batches):
            logit, cache = self.model.run_with_cache(batch["corrupted_prompts"])
            final_logit[idx, :, 0], final_logit[idx, :, 1] = to_logit_token(logit[:, -1, :], batch["target"])

            residual_stream = cache[component, layer]
            residual_logit = einops.einsum(self.model.W_U, residual_stream[:, position, :], "d d_v, b d -> b d_v")
            residual_logit = self.model.ln_final(residual_logit)

            position_logit[idx, :, 0], position_logit[idx, :, 1] = to_logit_token(residual_logit, batch["target"])

        final_logit = einops.rearrange(final_logit, "b s d -> (b s) d")
        position_logit = einops.rearrange(position_logit, "b s d -> (b s) d")
        return final_logit, position_logit

    def get_correlation_all_len(self, position, **kwargs):
        lenghts = self.dataset.get_lengths()
        final_logit = {}
        position_logit = {}
        for l in lenghts:
            final, pos = self.get_logit_single_len(l, position=position, **kwargs)
            if final is not None and pos is not None:
                final_logit[l], position_logit[l] = final, pos
        final_logit = torch.cat(list(final_logit.values()), dim=0)
        position_logit = torch.cat(list(position_logit.values()), dim=0)
        return final_logit, position_logit

    def project_ratio_heads_single_len(self, length):
        self.set_len(length)
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        num_batches = len(dataloader)

        project_ratio_mem = torch.zeros((self.model.cfg.n_layers, self.model.cfg.n_heads, num_batches, self.batch_size))
        project_ratio_cp = torch.zeros((self.model.cfg.n_layers, self.model.cfg.n_heads, num_batches, self.batch_size))

        for idx, batch in tqdm(enumerate(dataloader), total=num_batches, desc="Project ratio"):
            logit, cache = self.model.run_with_cache(batch["corrupted_prompts"])
            for layer in range(self.model.cfg.n_layers):
                for head in range(self.model.cfg.n_heads):
                    residual_stream = cache["resid_post", layer]
                    W_OV = (self.model.blocks[layer].attn.W_V @ self.model.blocks[layer].attn.W_O)[head]
                    output_direction = (residual_stream[:, -1, :] @ W_OV)

                    project_ratio = torch.einsum("b d, b d -> b", output_direction, residual_stream[:, -1, :])


from concurrent.futures import ThreadPoolExecutor

class DecomposeResidDir(BaseExperiment):
    def __init__(self, model: WrapHookedTransformer, dataset: TlensDataset, batch_size, filter_outliers=False):
        super().__init__(dataset, model, batch_size, filter_outliers)

    def get_basis(self, token):
        matrix = self.model.W_U.T
        significant = (matrix[token,:] > matrix[token,:].mean() + 2*matrix[token,:].std()) # | (matrix[token,:] < matrix[token,:].mean() - 2*matrix[token,:].std())
        indices = torch.where(significant)[0]
        
        std_basis= torch.eye(matrix.size(1))[indices.cpu()]
        # return matrix[token, indices].unsqueeze(0) * std_basis
        return std_basis    
    
    def get_common_basis(self, base1, base2):
        comparison = base1.unsqueeze(1) == base2.unsqueeze(0)
        return base1[torch.where(comparison.all(dim=-1))[0],:]
    
    def project(self, vector, basis):
        # projection = einops.einsum( vector.cuda(), basis.cuda(),"d, n d ->  n")
        # projection = basis.cpu() @ vector.cpu()
        # print("vector shape", vector.shape, "basis shape", basis.shape)
        projection_matrix = basis.T @ basis
        projection = projection_matrix.cuda() @ vector.cuda()
        
        # print("projection", projection, "projection norm", projection.norm())
        return basis.shape[0]
    
    def analyze_subspace(self, l):
        self.set_len(l)
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        num_batches = len(dataloader)
        logit_diffs = []
        norm = []
        for i, batch in tqdm(enumerate(dataloader), total=num_batches):
            logit, cache = self.model.run_with_cache(batch["corrupted_prompts"])
            logit_mem, logit_cp = to_logit_token(logit[:,-1,:], batch["target"], normalize="none")
            residual_stream = cache["resid_post", 11][:,-1,:]
            for idx, tokens in enumerate(batch["target"]):
                mem_token = tokens[0]
                cp_token = tokens[1]
                mem_base = self.get_basis(mem_token)
                cp_base = self.get_basis(cp_token)
                common_base = self.get_common_basis(mem_base, cp_base)
                mem_projection = self.project(residual_stream[idx,:], common_base)
                logit_diff = (logit_cp[idx] - logit_mem[idx]).abs()
                # print("logit_diff", logit_diff, "basis_norm", mem_projection)
                logit_diffs.append(logit_diff.item())
                # logit_diffs.append(logit_diff.item())
                norm.append(mem_projection)
                
        return logit_diffs, norm
    
    
class AttentionPattern(BaseExperiment):
    def __init__(self, model: WrapHookedTransformer, dataset: TlensDataset, batch_size, filter_outliers=False):
        super().__init__(dataset, model, batch_size, filter_outliers)
    def get_attention_pattern_single_len(self, length):
        self.set_len(length, slice_to_fit_batch=False)
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        num_batches = len(dataloader)
        assert num_batches > 0, f"Lenght {length} has no examples"

        # attention_pattern = torch.zeros((self.model.cfg.n_layers, self.model.cfg.n_heads, len, len, self.num_batches, self.batch_size))
        attention_pattern = [[] for _ in range(self.model.cfg.n_layers)]
        for idx, batch in tqdm(enumerate(dataloader), total=num_batches, desc=f"Attention pattern at len {length}"):
            logit, cache = self.model.run_with_cache(batch["corrupted_prompts"])
            for layer in range(self.model.cfg.n_layers):
                pattern = cache["pattern", layer] # (batch_size, n_heads, len, len)
                attention_pattern[layer].append(pattern.cpu()) # list of [[(batch_size, n_heads, len, len),..], ...] for each layer
                
        # from list of list to list of tensor cat along the batch dimension
        attention_pattern = [torch.cat(layer, dim=0) for layer in attention_pattern] # list of [(num_batches, n_heads, len, len), ...] for each layer
        
        # from list of tensor to tensor of shape (n_layers, num_batches, n_heads, len, len)
        attention_pattern = torch.stack(attention_pattern, dim=0)
        
        # rearrange to (n_batches, n_layers, n_heads, len, len)
        attention_pattern = einops.rearrange(attention_pattern, "l b h p q -> b l h p q")
        
        return attention_pattern