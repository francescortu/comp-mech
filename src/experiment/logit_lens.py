import re
import torch
from torch.utils.data import DataLoader
import einops
from tqdm import tqdm
from src.dataset import TlensDataset
from src.model import WrapHookedTransformer
from src.base_experiment import BaseExperiment, to_logit_token
from typing import Optional, Tuple, Literal, Union
from src.utils import get_aggregator


class LogitStorage:
    """
    store logits and return shape (layers, position, examples) or (layers, position, heads, examples)
    """

    def __init__(self, n_layers: int, length: int, experiment:Literal["copyVSfact", "contextVSfact"] ,n_heads: int = 1):
        self.n_layers = n_layers
        self.length = length
        self.n_heads = n_heads
        self.size = n_layers * length * n_heads
        self.experiment:Literal["copyVSfact", "contextVSfact"] = experiment

        # Using a dictionary to store logit values
        self.logits = {
            "mem_logit": [[] for _ in range(self.size)],
            "cp_logit": [[] for _ in range(self.size)],
            "mem_winners": [[] for _ in range(self.size)],
            "cp_winners": [[] for _ in range(self.size)],
        }


    def _get_index(self, layer: int, position: int, head: int = 0):
        return (layer * self.length + position) * self.n_heads + head

    def _reshape_tensor(self, tensor: torch.Tensor):
        return einops.rearrange(
            tensor, "layer position examples -> examples layer position"
        )

    def _reshape_tensor_back(self, tensor: torch.Tensor):
        return einops.rearrange(
            tensor, "examples layer position -> layer position examples"
        )

    def _aggregate_result(
        self, object_positions: Union[torch.Tensor, int], pattern: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        pattern = self._reshape_tensor(pattern)
        aggregate_result = get_aggregator(self.experiment)
        intermediate_aggregate = aggregate_result(
            pattern, object_positions, self.length, **kwargs
        )
        return self._reshape_tensor_back(intermediate_aggregate)



    def store(
        self,
        layer: int,
        position: int,
        logit: Tuple[
            torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]
        ],
        head: int = 0,
        mem_winners: Optional[torch.Tensor] = None,
        cp_winners: Optional[torch.Tensor] = None,
    ):
        mem_logit, cp_logit, _, _ = logit
        mem_logit, cp_logit = mem_logit.to("cpu"), cp_logit.to("cpu")
        index = self._get_index(layer, position, head)
        self.logits["mem_logit"][index].append(mem_logit)
        self.logits["cp_logit"][index].append(cp_logit)
        if mem_winners is not None:
            self.logits["mem_winners"][index].append(mem_winners)
        if cp_winners is not None:
            self.logits["cp_winners"][index].append(cp_winners)

    def _reshape_logits(self, logits_list, shape):
        return torch.stack([torch.cat(logits, dim=0) for logits in logits_list]).view(
            shape
        )

    def get_logit(self):
        shape = (self.n_layers, self.length, -1)
        if self.logits["mem_winners"][0] == []:
            return tuple(
                self._reshape_logits(self.logits[key], shape) for key in self.logits if key != "mem_winners" and key != "cp_winners"
            )
        return tuple(
            self._reshape_logits(self.logits[key], shape) for key in self.logits
        )

    def get_aggregate_logit(self, object_position: torch.Tensor, **kwargs):
        return_tuple = self.get_logit()
        aggregate_tensor = []
        for elem in return_tuple:
            aggregate_tensor.append(self._aggregate_result(object_position, elem, **kwargs))

        return tuple(aggregate_tensor)


class IndexLogitStorage(LogitStorage):
    def __init__(self, n_layers: int, length: int, experiment:Literal["copyVSfact", "contextVSfact"],  n_heads: int = 1):
        super().__init__(n_layers, length, experiment, n_heads)
        self.logits.update(
            {
                "mem_logit_idx": [[] for _ in range(self.size)],
                "cp_logit_idx": [[] for _ in range(self.size)],
            }
        )

    def store(
        self,
        layer: int,
        position: int,
        logit: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        head: int = 0,
        mem_winners: Optional[torch.Tensor] = None,
        cp_winners: Optional[torch.Tensor] = None,
    ):
        mem_logit, cp_logit, mem_logit_idx, cp_logit_idx = logit
        if mem_logit_idx is not None:
            mem_logit_idx = mem_logit_idx.to("cpu")
        if cp_logit_idx is not None:
            cp_logit_idx = cp_logit_idx.to("cpu")
        mem_logit, cp_logit = mem_logit.to("cpu"), cp_logit.to("cpu")
        index = self._get_index(layer, position, head)
        self.logits["mem_logit"][index].append(mem_logit)
        self.logits["cp_logit"][index].append(cp_logit)
        # Handle None values for mem_logit_idx and cp_logit_idx
        self.logits["mem_logit_idx"][index].append(
            mem_logit_idx if mem_logit_idx is not None else torch.tensor([])
        )
        self.logits["cp_logit_idx"][index].append(
            cp_logit_idx if cp_logit_idx is not None else torch.tensor([])
        )
        if mem_winners is not None:
            self.logits["mem_winners"][index].append(mem_winners)
        if cp_winners is not None:
            self.logits["cp_winners"][index].append(cp_winners)


class HeadLogitStorage(IndexLogitStorage, LogitStorage):
    def __init__(self, n_layers: int, length: int, experiment:Literal["copyVSfact", "contextVSfact"], n_heads: int):
        super().__init__(n_layers, length, experiment, n_heads)

    @classmethod
    def from_logit_storage(cls, logit_storage: LogitStorage, n_heads: int):
        return cls(logit_storage.n_layers, logit_storage.length, logit_storage.experiment, n_heads)

    @classmethod
    def from_index_logit_storage(
        cls, index_logit_storage: IndexLogitStorage, n_heads: int
    ):
        return cls(index_logit_storage.n_layers, index_logit_storage.length,index_logit_storage.experiment, n_heads)

    def _reshape_tensor(self, tensor: torch.Tensor):
        return einops.rearrange(
            tensor, "layer position head examples -> examples layer head position"
        )

    def _reshape_tensor_back(self, tensor: torch.Tensor):
        return einops.rearrange(
            tensor, "examples layer head position -> layer head position examples"
        )

    def get_logit(self):
        shape = (self.n_layers, self.length, self.n_heads, -1)
        return tuple(
            self._reshape_logits(self.logits[key], shape) for key in self.logits
        )


class LogitLens(BaseExperiment):
    def __init__(
        self, dataset: TlensDataset, model: WrapHookedTransformer, batch_size: int, experiment: Literal["copyVSfact", "contextVSfact"],
    ):
        super().__init__(dataset, model, batch_size, experiment)
        self.valid_blocks = ["mlp_out", "resid_pre", "resid_post", "attn_out"]
        self.valid_heads = ["head"]
        self.mean_logit_per_layer = None

    def project_per_position(self, component_cached: torch.Tensor, length: int):
        # assert that the activation name is a f-string with a single placeholder for the layer
        assert (
            component_cached.shape[1] == length
        ), f"component_cached.shape[1] = {component_cached.shape[1]}, self.model.cfg.n_heads = {self.model.cfg.n_heads}"
        assert (
            component_cached.shape[2] == self.model.cfg.d_model
        ), f"component_cached.shape[2] = {component_cached.shape[2]}, self.model.cfg.d_model = {self.model.cfg.d_model}"

        for position in range(length):
            component_cached = self.model.ln_final(component_cached)
            logit = einops.einsum(
                self.model.W_U, component_cached[:, position, :], "d d_v, b d -> b d_v"
            )
            yield logit

    def project_length(
        self,
        length: int,
        component: str,
        return_index: bool = False,
        normalize_logit: Literal["none", "softmax", "log_softmax"] = "none",
    ):
        self.set_len(length, slice_to_fit_batch=False)
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        num_batches = len(dataloader)
        if num_batches == 0:
            return None

        if return_index:
            storer = IndexLogitStorage(self.model.cfg.n_layers, length, self.experiment)
            if component in self.valid_heads:
                storer = HeadLogitStorage.from_index_logit_storage(
                    storer, self.model.cfg.n_heads
                )
        else:
            storer = LogitStorage(self.model.cfg.n_layers, length, self.experiment)
            if component in self.valid_heads:
                storer = HeadLogitStorage.from_logit_storage(
                    storer, self.model.cfg.n_heads
                )
        subject_positions = []
        object_positions = []
        for batch in dataloader:
            _, cache = self.model.run_with_cache(batch["prompt"], prepend_bos=False)
            subject_positions.append(batch["subj_pos"])
            object_positions.append(batch["obj_pos"])
            for layer in range(self.model.cfg.n_layers):
                if component in self.valid_blocks:
                    cached_component = cache[component, layer]
                    for position, logit in enumerate(
                        self.project_per_position(cached_component, length)
                    ):
                        logit_token = to_logit_token(
                            logit,
                            batch["target"],
                            normalize=normalize_logit,
                            return_index=return_index,
                            return_rank=True,
                        )
                        mem_rank = logit_token[4]
                        cp_rank = logit_token[5]
                        # logit_token = list(logit_token)
                        # logit_mean = logit.mean(-1).cpu()
                        # logit_token[0] = (logit_token[0].cpu() - logit.mean(-1).cpu()) / logit.mean(-1).cpu() #! MEAN
                        # logit_token[1] = (logit_token[1].cpu() - logit.mean(-1).cpu()) / logit.mean(-1).cpu() #! MEAN
                        storer.store( #! MEAN
                            layer=layer,
                            position=position,
                            #logit=tuple(logit_token),  # type: ignore
                            logit=logit_token[0:4],  # type: ignore
                            mem_winners=mem_rank,
                            cp_winners=cp_rank,
                        )
                        #storer.store(layer=layer, position=position, logit=logit_token)  # type: ignore
                elif component in self.valid_heads:
                    cached_component = cache[f"blocks.{layer}.attn.hook_z"]
                    for head in range(self.model.cfg.n_heads):
                        output_head = einops.einsum(
                            cached_component[:, :, head, :],
                            self.model.blocks[layer].attn.W_O[head, :, :],  # type: ignore
                            "batch pos d_head, d_head d_model -> batch pos d_model",
                        )  # type: ignore
                        for position, logit in enumerate(
                            self.project_per_position(output_head, length)
                        ):
                            logit_token = to_logit_token(
                                logit,
                                batch["target"],
                                normalize=normalize_logit,
                                return_index=return_index,
                            )
                            storer.store(
                                layer=layer,
                                position=position,
                                head=head,
                                logit=logit_token,  # type: ignore
                            )
                else:
                    raise ValueError(
                        f"component must be one of {self.valid_blocks + self.valid_heads}"
                    )
        subject_positions = torch.cat(subject_positions, dim=0)
        if self.experiment == "contextVSfact":
            object_positions = torch.cat(object_positions, dim=0)
            return storer.get_aggregate_logit(object_position=object_positions, subj_positions=subject_positions, batch_dim=0)
        object_positions = self.dataset.obj_pos[0]
        return storer.get_aggregate_logit(object_position=object_positions)

    def project(
        self,
        component: str,
        return_index: bool = False,
        normalize_logit: Literal["none", "softmax", "log_softmax"] = "none",
    ):
        lengths = self.dataset.get_lengths()
        # remove 11 from lengths
        if 11 in lengths:
            lengths.remove(11)
        result = {}
        for l in tqdm(lengths, desc="Logit lens:"):
            result[l] = self.project_length(
                l, component, return_index=return_index, normalize_logit=normalize_logit
            )

        # select a random key to get the shape of the result
        tuple_shape = len(result[lengths[0]])
        # result is a dict {lenght: }
        aggregated_result = [
            torch.cat([result[l][idx_tuple] for l in lengths], dim=-1)
            for idx_tuple in range(tuple_shape)
        ]

        return tuple(aggregated_result)

    def compute_mean_over_layers(self, result: Tuple[torch.Tensor, ...]):
        # for each layer, compute the mean over the lengths
        mean_result = tuple(  # type: ignore
            [result[idx_tuple].mean(dim=-1) for idx_tuple in range(len(result))]
        )

        # compute the percentage increase for each position/head over the mean for the same layer across all positions/heads
        raise NotImplementedError("TODO")

    def run(
        self,
        component: str,
        return_index: bool = False,
        normalize_logit: Literal["none", "softmax", "log_softmax"] = "none",
    ):
        result = self.project(
            component, return_index=return_index, normalize_logit=normalize_logit
        )

        import pandas as pd

        data = []
        for layer in range(self.model.cfg.n_layers):
            #compute the avarage over the layers
            mem_avg_layer = result[0][layer].mean()
            cp_avg_layer = result[1][layer].mean()
            diff_avg_layer = (result[0][layer] - result[1][layer]).mean()
            for position in range(result[0][layer].shape[0]):
                if component in self.valid_heads:
                    for head in range(self.model.cfg.n_heads):
                        data.append(
                            {
                                "component": f"H{head}",
                                "layer": layer,
                                "position": position,
                                "mem": result[0][layer][head][position].mean().item(),
                                "cp": result[1][layer][head][position].mean().item(),
                                "mem_std": result[0][layer][head][position]
                                .std()
                                .item(),
                                "cp_std": result[1][layer][head][position].std().item(),
                                "mem_idx": None
                                if not return_index
                                else result[2][layer][head][position].argmax().item(),
                                "cp_idx": None
                                if not return_index
                                else result[3][layer][head][position].argmin().item(),
                            }
                        )
                else:
                    mem_perc = 100 * (result[0][layer][position].mean().item() - mem_avg_layer) / mem_avg_layer
                    cp_perc = 100 * (result[1][layer][position].mean().item() - cp_avg_layer) / cp_avg_layer
                    diff_perc = (result[0][layer][position] - result[1][layer][position]).mean()
                    diff_perc = 100 * (diff_perc - diff_avg_layer) / diff_avg_layer
                    
                    
                    data.append(
                        {
                            "component": f"{component}",
                            "layer": layer,
                            "position": position,
                            "mem": result[0][layer][position].mean().item(),
                            "cp": result[1][layer][position].mean().item() ,
                            "diff": (result[0][layer][position] - result[1][layer][position]).mean().item(),
                            "mem_perc": mem_perc.item(),
                            "cp_perc": cp_perc.item(),
                            "diff_perc": diff_perc.item(),
                            "mem_std": result[0][layer][position].std().item(),
                            "cp_std": result[1][layer][position].std().item(),
                            "mem_idx": None
                            if not return_index
                            else result[2][layer][position].argmax().item(),
                            "cp_idx": None
                            if not return_index
                            else result[3][layer][position].argmin().item(),
                        }
                    )

        return pd.DataFrame(data)
