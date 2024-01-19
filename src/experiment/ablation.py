from regex import F
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dataset import TlensDataset
from src.model import WrapHookedTransformer
from src.base_experiment import BaseExperiment, to_logit_token
from typing import Optional,  Tuple,  Dict, Any, Literal
import pandas as pd
from src.experiment import LogitStorage, HeadLogitStorage
from functools import partial
from copy import deepcopy


class Ablate(BaseExperiment):
    def __init__(
        self, dataset: TlensDataset, model: WrapHookedTransformer, batch_size: int, experiment: Literal["copyVSfact", "contextVSfact"] = "copyVSfact",
    ):
        super().__init__(dataset, model, batch_size, experiment)
        self.position_component = ["mlp_out", "attn_out"]
        # self.position_component = ["attn_out", "resid_pre", "mlp_out", "resid_pre"]
        self.head_component = ["head"]

    def _get_freezed_attn(self, cache) -> Dict[str, Tuple[torch.Tensor, Any]]:
        """
        Return the hooks to freeze the attention (i.e. the attention activations)
        """

        def freeze_hook(attn_activation, hook, clean_attn_activation):
            attn_activation = clean_attn_activation
            return attn_activation

        hooks = {}

        for layer in range(self.model.cfg.n_layers):
            hooks[f"L{layer}"] = (
                f"blocks.{layer}.hook_attn_out",
                partial(
                    freeze_hook,
                    clean_attn_activation=cache[f"blocks.{layer}.hook_attn_out"],
                ),
            )

        return hooks

    def _get_freezed_attn_pattern(self, cache) -> Dict[str, Tuple[torch.Tensor, Any]]:
        """
        Return the hooks to freeze the attention pattern (i.e. the attention weights)
        """

        def freeze_hook(attn_activation, hook, clean_attn_activation):
            attn_activation = clean_attn_activation
            return attn_activation

        hooks = {}

        for layer in range(self.model.cfg.n_layers):
            hooks[f"L{layer}"] = (
                f"blocks.{layer}.attn.hook_pattern",
                partial(
                    freeze_hook,
                    clean_attn_activation=cache[f"blocks.{layer}.attn.hook_pattern"],
                ),
            )

        return hooks

    def _get_current_hooks(
        self,
        layer: int,
        component: str,
        freezed_attn: Dict[str, Tuple[torch.Tensor, Any]],
        head: Optional[int] = None,
        position: Optional[int] = None,
    ):
        """
        Apply the ablation pattern to the model
        """
        hooks = deepcopy(freezed_attn)
        if component in self.position_component:
            hook_name = f"blocks.{layer}.hook_{component}"

            def pos_ablation_hook(activation, hook, position):
                activation[:, position, :] = 0
                return activation

            new_hook = (hook_name, partial(pos_ablation_hook, position=position))

        if component in self.head_component:
            hook_name = f"blocks.{layer}.attn.hook_pattern"

            def head_ablation_hook(activation, hook, head):
                activation[:, head, -1, :] = 0
                return activation

            new_hook = (hook_name, partial(head_ablation_hook, head=head))

        hooks[f"L{layer}"] = new_hook  # type: ignore

        list_hooks = list(hooks.values())
        return list_hooks

    def _run_with_hooks(self, batch, hooks):
        """
        launch the model with the given hooks
        """
        self.model.reset_hooks()
        with torch.no_grad():
            logit = self.model.run_with_hooks(
                batch["prompt"],
                fwd_hooks=hooks,
            )
        return logit[:, -1, :]  # type: ignore

    def _process_model_run(
        self, layer, position, head, component, batch, freezed_attn, storage, normalize_logit
    ):
        """
        run the model with the given hooks and store the logit in the storage
        """
        hooks = self._get_current_hooks(
            layer, component, freezed_attn, head=head, position=position
        )
        logit = self._run_with_hooks(batch, hooks)
        logit_token = to_logit_token(logit, batch["target"], normalize=normalize_logit)
        if component in self.position_component:
            storage.store(layer=layer, position=position, logit=logit_token)
        if component in self.head_component:
            storage.store(layer=layer, position=0, head=head, logit=logit_token)

    def ablate_single_len(
        self,
        length: int,
        component: str,
        normalize_logit: Literal["none", "softmax", "log_softmax"] = "none",
        total_effect: bool = False,
    ):
        """
        Ablate the dataset with a specific length and component
        """

        self.set_len(length, slice_to_fit_batch=False)
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        num_batches = len(dataloader)

        if num_batches == 0:
            return None

        if component in self.position_component:
            storage = LogitStorage(n_layers=self.model.cfg.n_layers, length=length, experiment=self.experiment)
        elif component in self.head_component:
            storage = HeadLogitStorage(
                n_layers=self.model.cfg.n_layers,
                length=1,
                n_heads=self.model.cfg.n_heads,
                experiment=self.experiment,
            )
        else:
            raise ValueError(f"component {component} not supported")
        subject_positions = []
        for batch in tqdm(dataloader, total=num_batches):
            subject_positions.append(batch["subj_pos"])
            _, cache = self.model.run_with_cache(batch["prompt"])
            if component == "mlp_out" or component == "attn_out" and total_effect is False:
                freezed_attn = self._get_freezed_attn_pattern(cache)
            elif component == "head" and total_effect is False:
                freezed_attn = self._get_freezed_attn(cache)
            elif total_effect is True:
                freezed_attn = {}
            else:
                raise ValueError(f"component {component} not supported")

            for layer in range(self.model.cfg.n_layers):
                if component in self.position_component:
                    for position in range(length):
                        self._process_model_run(
                            layer,
                            position,
                            None,
                            component,
                            batch,
                            freezed_attn,
                            storage,
                            normalize_logit,
                        )
                if component in self.head_component:
                    for head in range(self.model.cfg.n_heads):
                        self._process_model_run(
                            layer, None, head, component, batch, freezed_attn, storage, normalize_logit
                        )
                        
                if f"L{layer}" in freezed_attn:
                    freezed_attn.pop(
                        f"L{layer}"
                    )  # to speed up the process (no need to freeze the previous layer)

        if component in self.position_component:
            if self.experiment == "contextVSfact":
                subject_positions = torch.cat(subject_positions, dim=0)
                return storage.get_aggregate_logit(object_position=self.dataset.obj_pos[0], subj_positions=subject_positions, batch_dim=0)
            return storage.get_aggregate_logit(object_position=self.dataset.obj_pos[0])

        if component in self.head_component:
            return storage.get_logit()

    def ablate(
        self,
        component: str,
        normalize_logit: Literal["none", "softmax", "log_softmax"] = "none",
        **kwargs
    ):
        """
        Ablate the model by ablating each position in the sequence
        """
        lengths = self.dataset.get_lengths()
        if 11 in lengths:
            lengths.remove(11)
        result = {}
        for length in tqdm(lengths, desc=f"Ablating {component}", total=len(lengths)):
            result[length] = self.ablate_single_len(length, component, normalize_logit, **kwargs)

        tuple_shape = len(result[lengths[0]])
        aggregated_result = [
            torch.cat([result[length][idx_tuple] for length in lengths], dim=-1)
            for idx_tuple in range(tuple_shape)
        ]

        return tuple(aggregated_result)

    def run(
        self,
        component: str,
        normalize_logit: Literal["none", "softmax", "log_softmax"] = "none",
        total_effect: bool = False,
    ):
        """
        Run ablation for a specific component
        """

        result = self.ablate(component, normalize_logit, total_effect=total_effect)

        base_logit_mem, base_logit_cp = self.get_basic_logit(
            normalize_logit=normalize_logit
        )

        import pandas as pd

        if component in self.position_component:
            data = []
            for layer in range(self.model.cfg.n_layers):
                for position in range(result[0][layer].shape[0]):
                    data.append(
                        {
                            "component": component,
                            "layer": layer,
                            "position": position,
                            "head": "not applicable",
                            "mem": result[0][layer][position].mean().item(),
                            "cp": result[1][layer][position].mean().item(),
                            "diff": (
                                result[0][layer][position] - result[1][layer][position]
                            )
                            .mean()
                            .item(),
                            "mem_std": result[0][layer][position].std().item(),
                            "cp_std": result[1][layer][position].std().item(),
                            "diff_std": (
                                result[0][layer][position] - result[1][layer][position]
                            )
                            .std()
                            .item(),
                        }
                    )

        elif component in self.head_component:
            data = []
            for layer in range(self.model.cfg.n_layers):
                for head in range(self.model.cfg.n_heads):
                    data.append(
                        {
                            "component": component,
                            "layer": layer,
                            "position": "not applicable",
                            "head": head,
                            "mem": result[0][layer, 0, head, :].mean().item(),
                            "cp": result[1][layer, 0, head, :].mean().item(),
                            "diff": (
                                result[0][layer, 0, head, :]
                                - result[1][layer, 0, head, :]
                            )
                            .mean()
                            .item(),
                            "mem_std": result[0][layer, 0, head, :].std().item(),
                            "cp_std": result[1][layer, 0, head, :].std().item(),
                            "diff_std": (
                                result[0][layer, 0, head, :]
                                - result[1][layer, 0, head, :]
                            )
                            .std()
                            .item(),
                        }
                    )
        else:
            raise ValueError(f"component {component} not supported")

        data.append(
            {
                "component": "base",
                "layer": "not applicable",
                "position": "not applicable",
                "head": "not applicable",
                "mem": base_logit_mem.mean().item(),
                "cp": base_logit_cp.mean().item(),
                "diff": (base_logit_mem - base_logit_cp).mean().item(),
                "mem_std": base_logit_mem.std().item(),
                "cp_std": base_logit_cp.std().item(),
                "diff_std": (base_logit_mem - base_logit_cp).std().item(),
            }
        )

        return pd.DataFrame(data)

    def run_all(
        self, normalize_logit: Literal["none", "softmax", "log_softmax"] = "none", **kwargs
    ):
        """
        Run ablation for all components
        """
        dataframe_list = []
        for component in self.position_component + self.head_component:
            print(f"Running ablation for {component}")
            dataframe_list.append(self.run(component, normalize_logit, **kwargs))
        return pd.concat(dataframe_list)
