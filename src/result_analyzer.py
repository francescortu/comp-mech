import torch
import pandas as pd
from scipy import stats
import json


def _get_indices(flat_indices, size):
    rows = flat_indices // size
    cols = flat_indices % size
    return list(zip(rows.tolist(), cols.tolist()))


def compute_mean(delta):
    mean = delta.mean()
    std = delta.std()
    positive = (delta > 0).sum().item()
    negative = (delta < 0).sum().item()
    _, p_value = stats.ttest_1samp(delta, 0)
    return {
        "mean": mean,
        "std": std,
        "positive": positive,
        "negative": negative,
        "p-value": p_value,
    }


class ResultAnalyzer:
    def __init__(self, result_file_name):
        """
        Initialize the ResultAnalyzer with data from the provided file.

        Args:
        - result_file_name (str): Name of the result file.
        """
        self.result_file_name = result_file_name.split(".pt")[0]
        self.data = torch.load(
            f"../results/locate_mechanism/{result_file_name}",
            map_location=torch.device("cpu"),
        )
        self.data["mem"]["premise"] = [
            item for sublist in self.data["mem"]["premise"] for item in sublist
        ]
        self.data["cp"]["premise"] = [
            item for sublist in self.data["cp"]["premise"] for item in sublist
        ]

    def _process_data(self, data_key, sub_key, id_format, save_name):
        """
        Helper method to process data and compute metrics.

        Args:
        - data_key (str): Key to access primary data (e.g., "pos" or "neg").
        - sub_key (str): Key to access secondary data (e.g., "attn_head_out").
        - id_format (str): Format string for the ID (e.g., "L{layer}H{head}" or "L{layer}P{position}").

        Returns:
        - pd.DataFrame: DataFrame containing computed metrics.
        """

        # clean_logit = self.data[data_key][f"clean_logit_{logit_key}"]

        rows = []
        n_layers = self.data[data_key][sub_key]["mem_delta"].shape[1]
        n_components = self.data[data_key][sub_key]["mem_delta"].shape[2]
        for layer in range(n_layers):
            for idx in range(n_components):
                result = compute_mean(
                    self.data[data_key][sub_key][f"{data_key}_delta"][:, layer, idx]
                )
                rows.append(
                    {
                        "id": id_format.format(layer=layer, idx=idx),
                        "mean": result["mean"].item(),
                        "std": result["std"].item(),
                        "positive": result["positive"],
                        "negative": result["negative"],
                        # "p-value": result["p-value"],
                        "p-value": result["p-value"],
                        "kl-mean": self.data[data_key][sub_key]["kl-mean"][
                            0, layer, idx
                        ].item(),
                        "kl-std": self.data[data_key][sub_key]["kl-std"][
                            0, layer, idx
                        ].item(),
                    }
                )

        # Save the data
        pd.DataFrame(rows).to_csv(f"../results/locate_mechanism/{save_name}.csv")

        return pd.DataFrame(rows)

    def process_data(
        self, data_key: str, sub_key: str, id_format: str = None, save_name: str = None
    ):
        if save_name is None:
            save_name = f"{self.result_file_name}_{data_key}_{sub_key}"
        if sub_key in ["attn_head_out"]:
            id_format = "L{layer}H{idx}"
        elif sub_key in ["mlp_out", "attn_out_by_pos"]:
            id_format = "L{layer}P{idx}"
        return self._process_data(data_key, sub_key, id_format, save_name)

    def _process_top_component_per_prompt(self, data_key):
        logit_key = "mem" if data_key == "mem" else "cp"
        rows = []

        for prompt_idx, prompt in enumerate(self.data[data_key]["premise"]):
            # Get the top 3 attention heads
            result_attention_head = (
                self.data[data_key][f"clean_probs_{logit_key}"][prompt_idx]
                - self.data[data_key]["attn_head_out"][f"ablated_probs_{logit_key}"][
                    prompt_idx
                ]
            )

            # Get the top 3 component indices (layer, head) and their corresponding values for result_attention_head
            values, flat_indices = result_attention_head.view(-1).topk(3)
            result_attention_head_top3_idx = _get_indices(
                flat_indices, result_attention_head.size(1)
            )
            result_attention_head_top3_val = values.tolist()

            # Get the top 3 components mlp output
            result_component_mlp = (
                self.data[data_key][f"clean_probs_{logit_key}"][prompt_idx]
                - self.data[data_key]["mlp_out"][f"ablated_probs_{logit_key}"][
                    prompt_idx
                ]
            )

            # Get the top 3 component indices (layer, pos) and their corresponding values for result_component_mlp
            values_mlp, flat_indices_mlp = result_component_mlp.view(-1).topk(3)
            result_component_mlp_top3_idx = _get_indices(
                flat_indices_mlp, result_component_mlp.size(1)
            )
            result_component_mlp_top3_val = values_mlp.tolist()

            # Get the top 3 components attention output
            result_component_attn = (
                self.data[data_key][f"clean_probs_{logit_key}"][prompt_idx]
                - self.data[data_key]["attn_out_by_pos"][f"ablated_probs_{logit_key}"][
                    prompt_idx
                ]
            )

            # Get the top 3 component indices (layer, pos) and their corresponding values for result_component_attn
            values_attn, flat_indices_attn = result_component_attn.view(-1).topk(3)
            result_component_attn_top3_idx = _get_indices(
                flat_indices_attn, result_component_attn.size(1)
            )
            result_component_attn_top3_val = values_attn.tolist()

            rows.append(
                {
                    "prompt": prompt,
                    "top1_attention_head_idx": f"L{result_attention_head_top3_idx[0][0]}H{result_attention_head_top3_idx[0][1]}",
                    "top1_attention_head_val": result_attention_head_top3_val[0],
                    "top2_attention_head_idx": f"L{result_attention_head_top3_idx[1][0]}H{result_attention_head_top3_idx[1][1]}",
                    "top2_attention_head_val": result_attention_head_top3_val[1],
                    "top3_attention_head_idx": f"L{result_attention_head_top3_idx[2][0]}H{result_attention_head_top3_idx[2][1]}",
                    "top3_attention_head_val": result_attention_head_top3_val[2],
                    "top1_component_mlp_idx": f"L{result_component_mlp_top3_idx[0][0]}P{result_component_mlp_top3_idx[0][1]}",
                    "top1_component_mlp_val": result_component_mlp_top3_val[0],
                    "top2_component_mlp_idx": f"L{result_component_mlp_top3_idx[1][0]}P{result_component_mlp_top3_idx[1][1]}",
                    "top2_component_mlp_val": result_component_mlp_top3_val[1],
                    "top3_component_mlp_idx": f"L{result_component_mlp_top3_idx[2][0]}P{result_component_mlp_top3_idx[2][1]}",
                    "top3_component_mlp_val": result_component_mlp_top3_val[2],
                    "top3_component_mlp_val": result_component_mlp_top3_val,
                    "top1_component_attn_idx": f"L{result_component_attn_top3_idx[0][0]}P{result_component_attn_top3_idx[0][1]}",
                    "top1_component_attn_val": result_component_attn_top3_val[0],
                    "top2_component_attn_idx": f"L{result_component_attn_top3_idx[1][0]}P{result_component_attn_top3_idx[1][1]}",
                    "top2_component_attn_val": result_component_attn_top3_val[1],
                    "top3_component_attn_idx": f"L{result_component_attn_top3_idx[2][0]}P{result_component_attn_top3_idx[2][1]}",
                    "top3_component_attn_val": result_component_attn_top3_val[2],
                }
            )

        df = pd.DataFrame(rows)
        df.to_csv(
            f"../results/locate_mechanism/{self.result_file_name}_{logit_key}_top_component_per_prompt.csv"
        )
        return df

    def process_pos_top_component_per_prompt(self):
        return self._process_top_component_per_prompt("mem")

    def process_neg_top_component_per_prompt(self):
        return self._process_top_component_per_prompt("cp")

    def _compute_correlation(self, model, subkey, data_key="mem"):
        # load the dataset
        dataset = json.load(open(f"../data/dataset_{model.cfg.model_name}.json"))
        if data_key == "mem":
            dataset = dataset["memorizing_win"]
        else:
            dataset = dataset["copying_win"]

        df = pd.DataFrame()

        target = []
        for prompt in self.data[data_key]["premise"]:
            for dataset_el in dataset:
                if prompt == dataset_el["premise"]:
                    target.append(
                        model.to_tokens(
                            dataset_el["target"], prepend_bos=False
                        ).squeeze(-1)
                    )
                    break

        assert len(target) == len(self.data[data_key]["premise"])
        rows = []
        for layer in range(self.data[data_key][subkey]["mean"].shape[1]):
            for idx in range(self.data[data_key][subkey]["mean"].shape[2]):
                column_name = f"L{layer}H/P{idx}"

                delta_value = []
                for prompt_idx, prompt in enumerate(self.data[data_key]["premise"]):
                    delta_value.append(
                        self.data[data_key][subkey]["full_delta"][
                            prompt_idx, layer, idx
                        ].item()
                    )
                df[column_name] = delta_value

        probs_target = []
        for prompt_idx, prompt in enumerate(self.data[data_key]["premise"]):
            probs = torch.softmax(model(prompt), dim=-1)[:, -1, :]
            probs_target.append(probs[:, target[prompt_idx]].item())

        df["target_probs"] = probs_target

        df.to_csv(
            f"../results/locate_mechanism/{self.result_file_name}_{data_key}_{subkey}_correlation.csv"
        )

    def compute_correlation_pos(self, model, subkey):
        return self._compute_correlation(model, subkey, data_key="mem")
