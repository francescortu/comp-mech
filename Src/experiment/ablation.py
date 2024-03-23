from re import sub
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from Src.dataset import BaseDataset
from Src.model import WrapHookedTransformer
from Src.base_experiment import BaseExperiment, to_logit_token
from typing import Optional, Tuple, Dict, Any, Literal, Union, List
import pandas as pd
from Src.experiment import LogitStorage, HeadLogitStorage
from functools import partial
from copy import deepcopy

WINDOW = 1

class Ablate(BaseExperiment):
    def __init__(
        self,
        dataset: BaseDataset,
        model: WrapHookedTransformer,
        batch_size: int,
        experiment: Literal["copyVSfact", "contextVSfact"] = "copyVSfact",
    ):
        super().__init__(dataset, model, batch_size, experiment)
        self.position_component = ["mlp_out", "attn_out", "attn_out_pattern"]
        # self.position_component = ["attn_out", "resid_pre", "mlp_out", "resid_pre"]
        self.head_component = ["head", "head_object_pos"]
        self.first_mech_winners = 0
        self.second_mech_winners = 0

    def _get_freezed_attn(self, cache) -> Dict[str, Tuple[torch.Tensor, Any]]:
        """
        Return the hooks to freeze the attention (i.e. the attention activations)
        """

        def freeze_hook(attn_activation, hook, clean_attn_activation):
            attn_activation = clean_attn_activation
            return attn_activation

        hooks = {}

        for layer in range(self.model.cfg.n_layers - 1 ):
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

        for layer in range(self.model.cfg.n_layers - 1):
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
        object_position: Optional[torch.Tensor] = None,
    ):
        """
        Apply the ablation pattern to the model
        """
        hooks = deepcopy(freezed_attn)
        if component in self.position_component:
            if component == "attn_out_pattern":
                hooks = []
                def head_ablation_hook(activation, hook, head):
                    activation[:, head, -1, position] = 0
                    return activation
                
                for head in range(self.model.cfg.n_heads):
                    hooks.append(
                        (f"blocks.{layer}.attn.hook_pattern", partial(head_ablation_hook, head=head))
                    )
                    hooks.append(
                        (f"blocks.{layer + 1}.attn.hook_pattern", partial(head_ablation_hook, head=head))
                    )
                return hooks
            hook_name = f"blocks.{layer}.hook_{component}"

            def pos_ablation_hook(activation, hook, position):
                activation[:, position, :] = 0
                return activation

            new_hook = (hook_name, partial(pos_ablation_hook, position=position))

        if component in self.head_component:
            hook_name = f"blocks.{layer}.attn.hook_pattern"
            if component == "head_object_pos":
                object_position= object_position[0]
                def head_ablation_hook(activation, hook, head):
                    activation[:, head, -1, object_position] = 0
                    return activation
            
            else:
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
        # hooks = []
        with torch.no_grad():
            logit = self.model.run_with_hooks(
                batch["prompt"],
                prepend_bos=False,
                fwd_hooks=hooks,
            )
        return logit[:, -1, :]  # type: ignore

    def _process_model_run(
        self,
        layer,
        position,
        head,
        component,
        batch,
        freezed_attn,
        storage,
        normalize_logit,
        object_position=None,
    ):
        """
        run the model with the given hooks and store the logit in the storage
        """
        hooks = self._get_current_hooks(
            layer, component, freezed_attn, head=head, position=position, object_position=object_position
        )
        logit = self._run_with_hooks(batch, hooks)  #!more performance
        logit_token = to_logit_token(
            logit, batch["target"], 
            normalize=normalize_logit, 
            return_winners=True
        )

        if component in self.position_component:
            storage.store(
                layer=layer,
                position=position,
                logit=(logit_token[0], logit_token[1], logit_token[2], logit_token[3]),
                mem_winners=logit_token[4],
                cp_winners=logit_token[5],
            )
        if component in self.head_component:
            storage.store(
                layer=layer,
                position=0,
                head=head,
                logit=(logit_token[0], logit_token[1], logit_token[2], logit_token[3]),
                mem_winners=logit_token[4],
                cp_winners=logit_token[5],
            )

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
            storage = LogitStorage(
                n_layers=self.model.cfg.n_layers-1,
                length=length,
                experiment=self.experiment,
            )
        elif component in self.head_component:
            storage = HeadLogitStorage(
                n_layers=self.model.cfg.n_layers -1,
                length=1,
                n_heads=self.model.cfg.n_heads,
                experiment=self.experiment,
            )
        else:
            raise ValueError(f"component {component} not supported")
        subject_positions = []
        object_position = []
        for batch in tqdm(dataloader, total=num_batches):
            subject_positions.append(batch["subj_pos"])
            object_position.append(batch["obj_pos"])
            _, cache = self.model.run_with_cache(batch["prompt"], prepend_bos=False)
            if (
                component == "mlp_out"
                or component == "attn_out"
                and total_effect is False
            ):
                freezed_attn = self._get_freezed_attn_pattern(cache)
            elif component == "head" and total_effect is False:
                freezed_attn = self._get_freezed_attn(cache)
            elif total_effect is True:
                freezed_attn = {}
            else:
                raise ValueError(f"component {component} not supported")

            for layer in range(self.model.cfg.n_layers - 1):
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
                            layer,
                            None,
                            head,
                            component,
                            batch,
                            freezed_attn,
                            storage,
                            normalize_logit,
                            batch["obj_pos"]
                        )

                if f"L{layer}" in freezed_attn:
                    freezed_attn.pop(
                        f"L{layer}"
                    )  # to speed up the process (no need to freeze the previous layer)

        if component in self.position_component:
            if self.experiment == "contextVSfact":
                subject_positions = torch.cat(subject_positions, dim=0)
                object_position = torch.cat(object_position, dim=0)
                return storage.get_aggregate_logit(
                    object_position=object_position,
                    subj_positions=subject_positions,
                    batch_dim=0,
                )
            return storage.get_aggregate_logit(object_position=self.dataset.obj_pos[0])

        if component in self.head_component:
            return storage.get_logit()

    def ablate_single_len_component_attn_pythia(
        self,
        length: int,
        component:str,
        normalize_logit: Literal["none", "softmax", "log_softmax"] = "none",
        total_effect: bool = False,
    ):
        self.set_len(length, slice_to_fit_batch=False)
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        num_batches = len(dataloader)
        
        if num_batches == 0:
            return None
        
        if component in self.position_component:
            storage = LogitStorage(
                n_layers=self.model.cfg.n_layers-WINDOW +1,
                length=length,
                experiment=self.experiment,
            )

        subject_positions = []
        object_position = []
        for batch in tqdm(dataloader, total=num_batches):
            subject_positions.append(batch["subj_pos"])
            object_position.append(batch["obj_pos"])
            _, cache = self.model.run_with_cache(batch["prompt"], prepend_bos=False)
            
            for layer in range(0, self.model.cfg.n_layers+1-WINDOW, 1):
                for position in range(length):
                    if position != self.dataset.obj_pos[0]:
                        
                        place_holder_tensor = torch.zeros_like(batch["input_ids"][:,0]).cpu()
                        storage.store(
                            layer=layer ,
                            position=position,
                            logit=(place_holder_tensor, place_holder_tensor, place_holder_tensor, place_holder_tensor),
                            mem_winners=place_holder_tensor,
                            cp_winners=place_holder_tensor,
                        )
                    else:
                        def head_ablation_hook(activation, hook, head):
                            activation[:, head, -1, position ] = 0
                            return activation
                        hooks = []
                        for head in range(self.model.cfg.n_heads):
                            for i in range(WINDOW):
                                hooks.append(
                                    (f"blocks.{layer +i}.attn.hook_pattern", partial(head_ablation_hook, head=head))
                                )

                        logit = self._run_with_hooks(batch, hooks)
                        logit_token = to_logit_token(
                            logit, batch["target"], normalize=normalize_logit, return_winners=True
                        )
                        storage.store(
                            layer=layer,
                            position=position,
                            logit=(logit_token[0], logit_token[1], logit_token[2], logit_token[3]),
                            mem_winners=logit_token[4],
                            cp_winners=logit_token[5],
                        )
        return storage.get_aggregate_logit(object_position=self.dataset.obj_pos[0])
                            
    def ablate_factual_head(
        self,
        length: int,
        component:str,
        normalize_logit: Literal["none", "softmax", "log_softmax"] = "none",
        total_effect: bool = False,
    ):
        self.set_len(length, slice_to_fit_batch=False)
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        num_batches = len(dataloader)
        
        if num_batches == 0:
            return None
        
        if component in self.position_component:
            storage = LogitStorage(
                n_layers=self.model.cfg.n_layers,
                length=length,
                experiment=self.experiment,
            )

        subject_positions = []
        object_position = []
        for batch in tqdm(dataloader, total=num_batches):
            subject_positions.append(batch["subj_pos"])
            object_position.append(batch["obj_pos"])
            _, cache = self.model.run_with_cache(batch["prompt"], prepend_bos=False)
            
            for layer in range(0, self.model.cfg.n_layers, 1):
                for position in range(length):
                    if position == self.dataset.obj_pos[0] and layer in (0,1,2,3,4,5): #,2,3,4,5,6,7):
                        def head_ablation_hook(activation, hook, head, multiplicator):
                            #activation[:, head, -1, position ] = multiplicator * activation[:, head, -1, position]
                            # activation[:, head, -1, :] = multiplicator* activation[:, head, -1, :]
                            activation[:, head, -1, position] = 0 
                            # activation[:, head, -1, position+1:] = 1.5 * activation[:, head, -1, position+1:]
                            return activation
                        hooks = []
                        if layer == 0:
                            hooks.append(
                                (f"blocks.{17}.attn.hook_pattern", partial(head_ablation_hook, head=28, multiplicator=1.5))
                            )

                            hooks.append(
                                (f"blocks.{20}.attn.hook_pattern", partial(head_ablation_hook, head=18, multiplicator=1.5))
                            )
                            hooks.append(
                                (f"blocks.{21}.attn.hook_pattern", partial(head_ablation_hook, head=8, multiplicator=1.5))
                            )
                        if layer == 1:
                            hooks.append(
                                (f"blocks.{17}.attn.hook_pattern", partial(head_ablation_hook, head=28, multiplicator=2))
                            )

                        if layer == 2:
                            hooks.append(
                                (f"blocks.{20}.attn.hook_pattern", partial(head_ablation_hook, head=18,multiplicator=3))
                            )

                        if layer == 3:
                            hooks.append(
                                (f"blocks.{21}.attn.hook_pattern", partial(head_ablation_hook, head=8, multiplicator=4))
                            )
                        if layer == 4:
                            hooks.append(
                                (f"blocks.{17}.attn.hook_pattern", partial(head_ablation_hook, head=28,multiplicator=5))
                            )

                            hooks.append(
                                (f"blocks.{20}.attn.hook_pattern", partial(head_ablation_hook, head=18,multiplicator=5))
                            )
                            hooks.append(
                                (f"blocks.{21}.attn.hook_pattern", partial(head_ablation_hook, head=8, multiplicator=5))
                            )
                        if layer == 5:
                            hooks.append(
                                (f"blocks.{17}.attn.hook_pattern", partial(head_ablation_hook, head=28,multiplicator=10))
                            )

                            hooks.append(
                                (f"blocks.{20}.attn.hook_pattern", partial(head_ablation_hook, head=18,multiplicator=10))
                            )
                            hooks.append(
                                (f"blocks.{21}.attn.hook_pattern", partial(head_ablation_hook, head=8, multiplicator=10))
                            )
                        if layer == 6:
                            hooks.append(
                                (f"blocks.{17}.attn.hook_pattern", partial(head_ablation_hook, head=28,multiplicator=30))
                            )

                            hooks.append(
                                (f"blocks.{20}.attn.hook_pattern", partial(head_ablation_hook, head=18,multiplicator=30))
                            )
                            hooks.append(
                                (f"blocks.{21}.attn.hook_pattern", partial(head_ablation_hook, head=8, multiplicator=30))
                            )
                        # if layer == 0:
                        #     hooks.append(
                        #         (f"blocks.{10}.attn.hook_pattern", partial(head_ablation_hook, head=7, multiplicator=2))
                        #     )

                        #     hooks.append(
                        #         (f"blocks.{11}.attn.hook_pattern", partial(head_ablation_hook, head=10, multiplicator=2))
                        #     )
                            
                        # if layer == 1:
                        #     # hooks.append(
                        #     #     (f"blocks.{10}.attn.hook_pattern", partial(head_ablation_hook, head=7, multiplicator=2))
                        #     # )

                        #     hooks.append(
                        #         (f"blocks.{10}.attn.hook_pattern", partial(head_ablation_hook, head=7,multiplicator=2))
                        #     )
                        # if layer == 2:

                        #     hooks.append(
                        #         (f"blocks.{11}.attn.hook_pattern", partial(head_ablation_hook, head=10, multiplicator=2))
                        #     )
                        # if layer == 2:
                        #     hooks.append(
                        #         (f"blocks.{10}.attn.hook_pattern", partial(head_ablation_hook, head=7,multiplicator=2))
                        #     )

                        #     hooks.append(
                        #         (f"blocks.{11}.attn.hook_pattern", partial(head_ablation_hook, head=10,multiplicator=2))
                        #     )
                        if layer == 3:
                            hooks.append(
                                (f"blocks.{10}.attn.hook_pattern", partial(head_ablation_hook, head=7,multiplicator=10))
                            )

                            # hooks.append(
                            #     (f"blocks.{11}.attn.hook_pattern", partial(head_ablation_hook, head=10,multiplicator=10))
                            # )
                        if layer == 4:
                            # hooks.append(
                            #     (f"dat.{10}.attn.hook_pattern", partial(head_ablation_hook, head=7,multiplicator=3.5))
                            # )

                            hooks.append(
                                (f"blocks.{11}.attn.hook_pattern", partial(head_ablation_hook, head=10,multiplicator=10))
                            )
                        if layer == 5:
                            hooks.append(
                                (f"blocks.{10}.attn.hook_pattern", partial(head_ablation_hook, head=7,multiplicator=10))
                            )

                            hooks.append(
                                (f"blocks.{11}.attn.hook_pattern", partial(head_ablation_hook, head=10,multiplicator=10))
                            )
                        # if layer == 6:
                        #     hooks.append(
                        #         (f"blocks.{10}.attn.hook_pattern", partial(head_ablation_hook, head=7,multiplicator=10))
                        #     )

                        #     hooks.append(
                        #         (f"blocks.{11}.attn.hook_pattern", partial(head_ablation_hook, head=10,multiplicator=10))
                        #     )
                        # if layer == 7:
                        #     hooks.append(
                        #         (f"blocks.{10}.attn.hook_pattern", partial(head_ablation_hook, head=7,multiplicator=30))
                        #     )

                        #     hooks.append(
                        #         (f"blocks.{11}.attn.hook_pattern", partial(head_ablation_hook, head=10,multiplicator=30))
                        #     )

                        logit = self._run_with_hooks(batch, hooks)
                        logit_token = to_logit_token(
                            logit, batch["target"], normalize=normalize_logit, return_winners=True
                        )
                        storage.store(
                            layer=layer,
                            position=position,
                            logit=(logit_token[0], logit_token[1], logit_token[2], logit_token[3]),
                            mem_winners=logit_token[4],
                            cp_winners=logit_token[5],
                        )
                    else:
                        
                        place_holder_tensor = torch.zeros_like(batch["input_ids"][:,0]).cpu()
                        storage.store(
                            layer=layer ,
                            position=position,
                            logit=(place_holder_tensor, place_holder_tensor, place_holder_tensor, place_holder_tensor),
                            mem_winners=place_holder_tensor,
                            cp_winners=place_holder_tensor,
                        )
        return storage.get_aggregate_logit(object_position=self.dataset.obj_pos[0])               

    def ablate(
        self,
        component: str,
        normalize_logit: Literal["none", "softmax", "log_softmax"] = "none",
        **kwargs,
    ):
        """
        Ablate the model by ablating each position in the sequence
        """
        lengths = self.dataset.get_lengths()
        if 11 in lengths:
            lengths.remove(11)
        result = {}
        for length in tqdm(lengths, desc=f"Ablating {component}", total=len(lengths)):
            result[length] = self.ablate_factual_head(
                length, component, normalize_logit, **kwargs
            )

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
        load_from_pt: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, torch.Tensor]]:
        """
        Run ablation for a specific component
        """
        print(load_from_pt)
        if load_from_pt is None:
            result = self.ablate(component, normalize_logit, total_effect=total_effect)

            base_logit_mem, base_logit_cp = self.get_basic_logit(
                normalize_logit=normalize_logit
            )
        else:
            result_dict = torch.load(load_from_pt)
            result = (result_dict["mem"], result_dict["cp"])
            base_logit_mem = result_dict["base_mem"]
            base_logit_cp = result_dict["base_cp"]

        import pandas as pd

        if component in self.position_component:
            data = []
            for layer in range(0,self.model.cfg.n_layers-WINDOW+1):
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
                            "mem_sum": result[2][layer][position].sum().item(),
                            "cp_sum": result[3][layer][position].sum().item(),
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
            for layer in range(self.model.cfg.n_layers-WINDOW+1):

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
                            "mem_sum": result[2][layer, 0, head, :].sum().item(),
                            "cp_sum": result[3][layer, 0, head, :].sum().item(),
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


        return pd.DataFrame(data), {
            "mem": result[0],
            "cp": result[1],
            "base_mem": base_logit_mem,
            "base_cp": base_logit_cp,
        }

    def run_all(
        self,
        normalize_logit: Literal["none", "softmax", "log_softmax"] = "none",
        load_from_pt: Optional[str] = None,
        **kwargs,
    ) -> Tuple[pd.DataFrame, Dict[str, Union[List[torch.Tensor], torch.Tensor]]]:
        """
        Run ablation for all components
        """
        dataframe_list = []
        tuple_results_cat = {"mem": [], "cp": [], "base_mem": [], "base_cp": []}
        for component in self.position_component + self.head_component:
            print(f"Running ablation for {component}")
            dataset, tuple_results = self.run(
                component, normalize_logit, load_from_pt=load_from_pt, **kwargs
            )
            dataframe_list.append(dataset)
            tuple_results_cat["mem"].append(tuple_results["mem"])
            tuple_results_cat["cp"].append(tuple_results["cp"])
            tuple_results_cat["base_mem"].append(tuple_results["base_mem"])
            tuple_results_cat["base_cp"].append(tuple_results["base_cp"])

        return pd.concat(dataframe_list), tuple_results_cat
