from typing import List, Tuple, Union
from src.base_experiment import BaseExperiment, to_logit_token
from src.dataset import TlensDataset
from src.model import WrapHookedTransformer
import einops
import torch
from tqdm import tqdm
from functools import partial
from copy import deepcopy
import numpy as np




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

    # def aggregate_result(self, object_positions, logit_target, length):
    #     subject_1_1 = 5
    #     subject_1_2 = 6 if length > 15 else 5
    #     subject_1_3 = 7 if length > 17 else subject_1_2
    #     subject_2_1 = object_positions + 2
    #     subject_2_2 = object_positions + 3 if length > 15 else subject_2_1
    #     subject_2_3 = object_positions + 4 if length > 17 else subject_2_2
    #     subject_2_2 = subject_2_2 if subject_2_2 < length else subject_2_1
    #     subject_2_3 = subject_2_3 if subject_2_3 < length else subject_2_2
    #     last_position = length - 1
    #     object_positions_pre = object_positions - 1
    #     object_positions_next = object_positions + 1
    #     result_aggregate = torch.zeros((self.model.cfg.n_layers, 13))
    #     result_aggregate[:, 0] = logit_target[:, :subject_1_1].mean(dim=1)
    #     result_aggregate[:, 1] = logit_target[:, subject_1_1]
    #     result_aggregate[:, 2] = logit_target[:, subject_1_2]
    #     result_aggregate[:, 3] = logit_target[:, subject_1_3]
    #     result_aggregate[:, 4] = logit_target[:, subject_1_3 + 1:object_positions_pre].mean(dim=1)
    #     result_aggregate[:, 5] = logit_target[:, object_positions_pre]
    #     result_aggregate[:, 6] = logit_target[:, object_positions]
    #     result_aggregate[:, 7] = logit_target[:, object_positions_next]
    #     result_aggregate[:, 8] = logit_target[:, subject_2_1]
    #     result_aggregate[:, 9] = logit_target[:, subject_2_2]
    #     result_aggregate[:, 10] = logit_target[:, subject_2_3]

    #     result_aggregate[:, 11] = logit_target[:, subject_2_3 + 1:last_position ].mean(dim=1)
    #     # result_aggregate[:, 12] = logit_target[:, last_position - 1]
    #     result_aggregate[:, 12] = logit_target[:, last_position]
    #     return result_aggregate

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

