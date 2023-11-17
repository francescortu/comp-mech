
from src.base_experiment import BaseExperiment, to_logit_token
from src.model import WrapHookedTransformer
from src.dataset import TlensDataset
import einops
import torch
from tqdm import tqdm


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
    