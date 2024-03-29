from turtle import up
from typing import Callable, List, Optional, Union
import torch
from torch.utils.data import DataLoader
import einops
from tqdm import tqdm
from Src.dataset import BaseDataset
from Src.model import BaseModel
from Src.base_experiment import BaseExperiment
from typing import Tuple, Literal
from Src.utils import aggregate_result
import pandas as pd
import ipdb


class AttributeStorage:
    """
    Class to store the attributes of the logit attribution
    """

    def __init__(
        self, experiment: Literal["copyVSfact", "contextVSfact", "copyVSfact_factual"]
    ):
        self.mem_attribute = []
        self.cp_attribute = []
        self.diff_attribute = []
        self.labels:List[str] = []
        self.experiment: Literal[
            "copyVSfact", "contextVSfact", "copyVSfact_factual"
        ] = experiment

    def append(
        self,
        mem_attribute: torch.Tensor,
        cp_attribute: torch.Tensor,
        diff_attribute: torch.Tensor,
        labels: List[str],
        object_position: torch.Tensor,
        first_subject_positions: torch.Tensor,
        second_subject_positions: torch.Tensor,
        subject_lengths: torch.Tensor,
    ):
        mem_attribute, cp_attribute, diff_attribute = self.aggregate(
            mem_attribute,
            cp_attribute,
            diff_attribute,
            object_position,
            first_subject_positions,
            second_subject_positions,
            subject_lengths,
        )
        self.mem_attribute.append(mem_attribute.cpu())
        self.cp_attribute.append(cp_attribute.cpu())
        self.diff_attribute.append(diff_attribute.cpu())
        if len(self.labels) == 0:
            self.labels = labels

    def aggregate(
        self,
        mem_attribute,
        cp_attribute,
        diff_attribute,
        object_position,
        first_subject_positions,
        second_subject_positions,
        subject_lengths,
    ):
        length = mem_attribute.shape[-1]
        aggregated_mem = aggregate_result(
            experiment=self.experiment,
            pattern=mem_attribute,
            object_positions=object_position,
            first_subject_positions=first_subject_positions,
            second_subject_positions=second_subject_positions,
            subject_lengths=subject_lengths,
            length=length,   
        )
        aggregated_cp = aggregate_result(
            experiment=self.experiment,
            pattern=cp_attribute,
            object_positions=object_position,
            first_subject_positions=first_subject_positions,
            second_subject_positions=second_subject_positions,
            subject_lengths=subject_lengths,
            length=length,
        )
        aggregated_diff = aggregate_result(
            experiment=self.experiment,
            pattern=diff_attribute,
            object_positions=object_position,
            first_subject_positions=first_subject_positions,
            second_subject_positions=second_subject_positions,
            subject_lengths=subject_lengths,
            length=length,
        )
        return aggregated_mem, aggregated_cp, aggregated_diff

    def stack(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
        """
        from a list of tensors of shape (batch, component, position) to a tensor of shape (component, batch, position)
        Concatenate the tensors along the batch dimension
        """
        stacked_mem = torch.cat(self.mem_attribute, dim=1)
        stacked_cp = torch.cat(self.cp_attribute, dim=1)
        stacked_diff = torch.cat(self.diff_attribute, dim=1)
        return stacked_mem, stacked_cp, stacked_diff


class LogitAttribution(BaseExperiment):
    def __init__(
        self,
        dataset: BaseDataset,
        model: BaseModel,
        batch_size: int,
        experiment: Literal["copyVSfact", "contextVSfact"],
    ):
        super().__init__(dataset, model, batch_size, experiment)

    def slice_target(
        self, target: torch.Tensor, length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Slice the target into memory and control prompt
        return a tuple of (target_mem, target_cp) of shape (batch_size, seq_len), (batch_size, seq_len)
        """
        target_mem = target[:, 0]
        target_cp = target[:, 1]
        # repeat for batch_size times
        target_mem = einops.repeat(
            target_mem, "seq_len ->  seq_len length", length=length
        )
        target_cp = einops.repeat(
            target_cp, "seq_len ->  seq_len length", length=length
        )
        return target_mem, target_cp

    def attribute_single_len(
        self,
        length: int,
        storage: AttributeStorage,
        component: str,
        up_to_layer: Union[int, str] = "all",
        apply_ln: bool = False,
        normalize_logit: Literal["none", "softmax", "log_softmax"] = "none",
        hooks: Optional[List[Tuple[str, Callable]]] = None,
    ):
        if up_to_layer == "all":
            up_to_layer = self.model.cfg.n_layers
        elif isinstance(up_to_layer, str):
            raise ValueError("up_to_layer must be an integer or 'all'")
        elif up_to_layer > self.model.cfg.n_layers:
            raise ValueError(
                f"up_to_layer must be smaller than the number of layers: {self.model.cfg.n_layers}"
            )

        self.set_len(length, slice_to_fit_batch=False)
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

        # Create a storage object to store the logits
        for batch in dataloader:
            # print cuda memory
            # if hooks is not None:
            #     logits, cache = self.model.run_with_hooks(
            #         batch["prompt"], hooks=hooks, return_cache=True
            #     )
            # else:
            logits, cache = self.model.run_with_cache(
                batch["prompt"], prepend_bos=False
            )

            if normalize_logit != "none":
                raise NotImplementedError
            stack_of_resid, resid_labels = cache.decompose_resid(
                apply_ln=apply_ln, return_labels=True, mode="attn", layer=up_to_layer
            )
            stack_of_component, labels = cache.get_full_resid_decomposition(
                expand_neurons=False,
                apply_ln=apply_ln,
                return_labels=True,
                layer=up_to_layer,
            )  # return a tensor of shape (component_size, batch_size, seq_len, hidden_size)
            target_mem, target_cp = self.slice_target(batch["target"], length=length)

            labels = labels + resid_labels
            stack_of_component = torch.cat([stack_of_component, stack_of_resid], dim=0)

            mem_attribute = cache.logit_attrs(stack_of_component, tokens=target_mem)
            cp_attribute = cache.logit_attrs(stack_of_component, tokens=target_cp)
            diff_attribute = cache.logit_attrs(
                stack_of_component, tokens=target_mem, incorrect_tokens=target_cp
            )  # (component, batch, position)
            
            
            storage.append(
                mem_attribute.cpu(),
                cp_attribute.cpu(),
                diff_attribute.cpu(),
                labels,
                batch["obj_pos"].cpu(),
                batch["1_subj_pos"].cpu(),
                batch["2_subj_pos"].cpu(),
                batch["subj_len"].cpu(),
            )
            
        # clear the cuda cache
        torch.cuda.empty_cache()

    def attribute(self, apply_ln: bool = False, **kwargs) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], List[str]]:
        """
        run the logit attribution for all the lengths in the dataset and return a tuple of (mem, cp, diff) of shape (component, batch, position)
        """
        storage = AttributeStorage(self.experiment)
        lengths = self.dataset.get_lengths()
        for length in tqdm(lengths, desc="Attributing"):
            self.attribute_single_len(
                length, storage, "logit", apply_ln=apply_ln, **kwargs
            )

        return storage.stack(), storage.labels

    def run(
        self,
        #apply_ln: bool = False,
        normalize_logit: Literal["none", "softmax", "log_softmax"] = "none",
        **kwargs,
    ):
        """
        Run the logit attribution and return a dataframe
        """
        (mem, cp, diff), labels = self.attribute(
            apply_ln=apply_ln, normalize_logit=normalize_logit, **kwargs
        )

        data = []
        for i, label in enumerate(labels):  # type: ignore
            for position in range(mem.shape[-1]):
                data.append(
                    {
                        "label": label,
                        "position": position,
                        "mem_mean": mem[i, :, position].mean().item(),
                        "cp_mean": cp[i, :, position].mean().item(),
                        "diff_mean": diff[i, :, position].mean().item(),
                        "mem_std": mem[i, :, position].std().item(),
                        "cp_std": cp[i, :, position].std().item(),
                        "diff_std": diff[i, :, position].std().item(),
                    }
                )
        # index of head
        head_indexes = [i for i, label in enumerate(labels) if "H" in label]
        # sum all the head at position 12
        mem_all = mem[head_indexes, :, 12].mean()
        cp_all = cp[head_indexes, :, 12].mean()
        mem_all_std = mem[head_indexes, :, 12].std()
        cp_all_std = cp[head_indexes, :, 12].std()
        data.append(
            {
                "label": "all_heads",
                "position": 12,
                "mem_mean": mem_all.item(),
                "cp_mean": cp_all.item(),
                "diff_mean": 0,
                "mem_std": mem_all_std.item(),
                "cp_std": cp_all_std.item(),
                "diff_std": 0,
            }
        )
        # doing the same but just for layer 10 and 11
        head_indexes = [
            i for i, label in enumerate(labels) if ("H" in label and "L10" in label)
        ]
        mem_all = mem[head_indexes, :, 12].mean()
        cp_all = cp[head_indexes, :, 12].mean()
        mem_all_std = mem[head_indexes, :, 12].std()
        cp_all_std = cp[head_indexes, :, 12].std()
        data.append(
            {
                "label": "all_heads_L10",
                "position": 12,
                "mem_mean": mem_all.item(),
                "cp_mean": cp_all.item(),
                "diff_mean": 0,
                "mem_std": mem_all_std.item(),
                "cp_std": cp_all_std.item(),
                "diff_std": 0,
            }
        )
        # doing the same but just for layer 11
        head_indexes = [
            i for i, label in enumerate(labels) if ("H" in label and "L11" in label)
        ]
        mem_all = mem[head_indexes, :, 12].mean()
        cp_all = cp[head_indexes, :, 12].mean()
        mem_all_std = mem[head_indexes, :, 12].std()
        cp_all_std = cp[head_indexes, :, 12].std()
        data.append(
            {
                "label": "all_heads_L11",
                "position": 12,
                "mem_mean": mem_all.item(),
                "cp_mean": cp_all.item(),
                "diff_mean": 0,
                "mem_std": mem_all_std.item(),
                "cp_std": cp_all_std.item(),
                "diff_std": 0,
            }
        )

        # doint the same for layer 7
        head_indexes = [
            i for i, label in enumerate(labels) if ("H" in label and "L7" in label)
        ]
        mem_all = mem[head_indexes, :, 12].mean()
        cp_all = cp[head_indexes, :, 12].mean()
        mem_all_std = mem[head_indexes, :, 12].std()
        cp_all_std = cp[head_indexes, :, 12].std()
        data.append(
            {
                "label": "all_heads_L7",
                "position": 12,
                "mem_mean": mem_all.item(),
                "cp_mean": cp_all.item(),
                "diff_mean": 0,
                "mem_std": mem_all_std.item(),
                "cp_std": cp_all_std.item(),
                "diff_std": 0,
            }
        )
        # doint the same for layer 9
        head_indexes = [
            i for i, label in enumerate(labels) if ("H" in label and "L9" in label)
        ]
        mem_all = mem[head_indexes, :, 12].mean()
        cp_all = cp[head_indexes, :, 12].mean()
        mem_all_std = mem[head_indexes, :, 12].std()
        cp_all_std = cp[head_indexes, :, 12].std()
        data.append(
            {
                "label": "all_heads_L9",
                "position": 12,
                "mem_mean": mem_all.item(),
                "cp_mean": cp_all.item(),
                "diff_mean": 0,
                "mem_std": mem_all_std.item(),
                "cp_std": cp_all_std.item(),
                "diff_std": 0,
            }
        )

        return pd.DataFrame(data)
