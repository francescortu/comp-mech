import torch
import pandas as pd
from scipy import stats



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

    def _process_data(self, data_key, sub_key, id_format):
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

        return pd.DataFrame(rows)

    def process_pos_attn_head_out(self):
        """Process positive attention head output data."""
        return self._process_data("pos", "attn_head_out", "L{layer}H{idx}")

    def process_neg_attn_head_out(self):
        """Process negative attention head output data."""
        return self._process_data("neg", "attn_head_out", "L{layer}H{idx}")

    def process_pos_component_out_by_pos(self, key):
        """Process positive component output data by position."""
        return self._process_data("pos", key, "L{layer}P{idx}")

    def process_neg_component_out_by_pos(self, key):
        """Process negative component output data by position."""
        return self._process_data("neg", key, "L{layer}P{idx}")
