import torch
import pandas as pd
from scipy import stats

def _get_indices(flat_indices, size):
    rows = flat_indices // size
    cols = flat_indices % size
    return list(zip(rows.tolist(), cols.tolist()))


def compute_metric_difference(logit, corrupted_logit):
    """
    Compute the metric difference between logits.

    Args:
    - logit (torch.Tensor): Original logit values.
    - corrupted_logit (torch.Tensor): Corrupted logit values.

    Returns:
    - dict: Computed metrics.
    """
    delta = logit - corrupted_logit
    ttest = stats.ttest_ind(delta.cpu().detach().numpy(), 0, axis=0)
    return {
        "mean": delta.mean(dim=0),
        "std": delta.std(dim=0),
        "t-test": ttest[0],
        "p-value": ttest[1]
    }


class ResultAnalyzer:
    def __init__(self, result_file_name):
        """
        Initialize the ResultAnalyzer with data from the provided file.

        Args:
        - result_file_name (str): Name of the result file.
        """
        self.data = torch.load(
            f"../results/locate_mechanism/{result_file_name}",
            map_location=torch.device("cpu")
        )

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
        logit_key = "mem" if data_key == "pos" else "cp"
        corrupted_logit = self.data[data_key][f"corrupted_logit_{logit_key}"]
        
        rows = []
        for layer in range(self.data[data_key][sub_key]["mean"].shape[1]):
            for idx in range(self.data[data_key][sub_key]["mean"].shape[2]):
                result = compute_metric_difference(
                    self.data[data_key][sub_key][f"patched_logits_{logit_key}"][:, layer, idx],
                    corrupted_logit
                )
                rows.append(
                    {
                        "id": id_format.format(layer=layer, idx=idx),
                        "mean": result["mean"].item(),
                        "std": result["std"].item(),
                        "t-test": result["t-test"],
                        "p-value": result["p-value"]
                    }
                )

        # Save the data
        pd.DataFrame(rows).to_csv(f"../results/locate_mechanism/{save_name}.csv")
        
        return pd.DataFrame(rows)

    def process_pos_attn_head_out(self, save_name = "pos_attn_head_out"):
        """Process positive attention head output data."""
        return self._process_data("pos", "attn_head_out", "L{layer}H{idx}", save_name)

    def process_neg_attn_head_out(self, save_name = "neg_attn_head_out"):
        """Process negative attention head output data."""
        return self._process_data("neg", "attn_head_out", "L{layer}H{idx}", save_name)

    def process_pos_component_out_by_pos(self, key, save_name = "pos_component_out_by_pos"):
        """Process positive component output data by position."""
        return self._process_data("pos", key, "L{layer}P{idx}", save_name)

    def process_neg_component_out_by_pos(self, key, save_name = "neg_component_out_by_pos"):
        """Process negative component output data by position."""
        return self._process_data("neg", key, "L{layer}P{idx}", save_name)

    def _process_top_component_per_prompt(self, data_key):
        logit_key = "mem" if data_key == "pos" else "cp"
        rows = []

        for prompt_idx, prompt in enumerate(self.data[data_key]["premise"]):

            # Get the top 3 attention heads
            result_attention_head = self.data[data_key]["attn_head_out"][f"patched_logits_{logit_key}"][prompt_idx] - self.data[data_key][f"corrupted_logit_{logit_key}"][prompt_idx]
            
            # Get the top 3 component indices (layer, head) and their corresponding values for result_attention_head
            values, flat_indices = result_attention_head.view(-1).topk(3)
            result_attention_head_top3_idx = _get_indices(flat_indices, result_attention_head.size(1))
            result_attention_head_top3_val = values.tolist()

            # Get the top 3 components mlp output
            result_component_mlp = self.data[data_key]["mlp_out"][f"patched_logits_{logit_key}"][prompt_idx] - self.data[data_key][f"corrupted_logit_{logit_key}"][prompt_idx]

            # Get the top 3 component indices (layer, pos) and their corresponding values for result_component_mlp
            values_mlp, flat_indices_mlp = result_component_mlp.view(-1).topk(3)
            result_component_mlp_top3_idx = _get_indices(flat_indices_mlp, result_component_mlp.size(1))
            result_component_mlp_top3_val = values_mlp.tolist()

            # Get the top 3 components attention output
            result_component_attn = self.data[data_key]["attn_out_by_pos"][f"patched_logits_{logit_key}"][prompt_idx] - self.data[data_key][f"corrupted_logit_{logit_key}"][prompt_idx]

            # Get the top 3 component indices (layer, pos) and their corresponding values for result_component_attn
            values_attn, flat_indices_attn = result_component_attn.view(-1).topk(3)
            result_component_attn_top3_idx = _get_indices(flat_indices_attn, result_component_attn.size(1))
            result_component_attn_top3_val = values_attn.tolist()

            rows.append(
                {
                    "prompt": prompt,
                    "top3_attention_head_idx": ",".join([f"L{layer}H{head}" for layer, head in result_attention_head_top3_idx]),
                    "top3_attention_head_val": result_attention_head_top3_val,
                    "top3_component_mlp_idx": ",".join([f"L{layer}P{pos}" for layer, pos in result_component_mlp_top3_idx]),
                    "top3_component_mlp_val": result_component_mlp_top3_val,
                    "top3_component_attn_idx": ",".join([f"L{layer}P{pos}" for layer, pos in result_component_attn_top3_idx]),
                    "top3_component_attn_val": result_component_attn_top3_val
                }
            )

        df = pd.DataFrame(rows)
        df.to_csv(f"../results/locate_mechanism/{data_key}_top_component_per_prompt.csv")
        return df
        
    def process_pos_top_component_per_prompt(self):
        return self._process_top_component_per_prompt("pos")
    def process_neg_top_component_per_prompt(self):
        return self._process_top_component_per_prompt("neg")