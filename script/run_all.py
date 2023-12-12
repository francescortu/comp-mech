# Standard library imports
from dataclasses import dataclass, field
from math import exp, log
import os
import subprocess
import sys
import threading
import time
from typing import Optional, Literal

# Third-party library imports
from rich.console import Console
from rich.live import Live
from rich.progress import track
import argparse
import logging

# Local application/library specific imports
sys.path.append(os.path.abspath(os.path.join("..")))
sys.path.append(os.path.abspath(os.path.join("../src")))
sys.path.append(os.path.abspath(os.path.join("../data")))
from src.dataset import TlensDataset  # noqa: E402
from src.experiment import LogitAttribution, LogitLens, OV, Ablate  # noqa: E402
from src.model import WrapHookedTransformer  # noqa: E402
from src.utils import display_config, display_experiments, update_live, update_status  # noqa: E402

console = Console()
# set logging level to suppress warnings
logging.basicConfig(level=logging.ERROR)





@dataclass
class Config:
    model_name: str = "gpt2"
    batch_size: int = 10
    dataset_path: str = f"../data/full_data_sampled_{model_name}.json"
    dataset_slice: Optional[int] = None
    produce_plots: bool = True
    normalize_logit: Literal["none", "softmax", "log_softmax"] = "none"
    std_dev: int = 0  # 0 False, 1 True

    @classmethod
    def from_args(cls, args):
        return cls(
            model_name=args.model_name,
            batch_size=args.batch,
            dataset_path=f"../data/full_data_sampled_{args.model_name}.json",
            dataset_slice=args.slice,
            produce_plots=args.produce_plots,
            std_dev=0 if not args.std_dev else 1,
        )


@dataclass
class logit_attribution_config:
    std_dev: int = 0  # 0 False, 1 True


@dataclass
class logit_lens_config:
    component: str = "resid_post"
    return_index: bool = False
    normalize: str = "none"


### check folder and create if not exists
def save_dataframe(folder_path, file_name, dataframe):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    dataframe.to_csv(f"{folder_path}/{file_name}.csv", index=False)


def logit_attribution(model, dataset, config, args):
    dataset_slice_name = (
        "full" if config.dataset_slice is None else config.dataset_slice
    )
    if args.only_plot:
        subprocess.run(
            [
                "Rscript",
                "../src_figure/logit_attribution.R",
                f"../results/logit_attribution/{config.model_name}_{dataset_slice_name}",
                f"{config.std_dev}",
            ]
        )
        return

    print("Running logit attribution")
    attributor = LogitAttribution(dataset, model, config.batch_size // 5)
    dataframe = attributor.run(normalize_logit=config.normalize_logit)
    save_dataframe(
        f"../results/logit_attribution/{config.model_name}_{dataset_slice_name}",
        "logit_attribution_data",
        dataframe,
    )

    if config.produce_plots:
        # run the R script
        subprocess.run(
            [
                "Rscript",
                "../src_figure/logit_attribution.R",
                f"../results/logit_attribution/{config.model_name}_{dataset_slice_name}",
                f"{config.std_dev}",
            ]
        )


def logit_lens(model, dataset, config, args):
    data_slice_name = "full" if config.dataset_slice is None else config.dataset_slice
    if args.only_plot:
        subprocess.run(
            [
                "Rscript",
                "../src_figure/logit_lens.R",
                f"../results/logit_lens/{config.model_name}_{data_slice_name}",
            ]
        )
        return

    logit_lens_cnfg = logit_lens_config()
    print("Running logit lens")
    logit_lens = LogitLens(dataset, model, config.batch_size)
    dataframe = logit_lens.run(
        logit_lens_cnfg.component,
        logit_lens_cnfg.return_index,
        normalize_logit=config.normalize_logit,
    )
    save_dataframe(
        f"../results/logit_lens/{config.model_name}_{data_slice_name}",
        "logit_lens_data",
        dataframe,
    )

    if config.produce_plots:
        # run the R script
        subprocess.run(
            [
                "Rscript",
                "../src_figure/logit_lens.R",
                f"../results/logit_lens/{config.model_name}_{data_slice_name}",
            ]
        )


def ov_difference(model, dataset, config, args):
    data_slice_name = "full" if config.dataset_slice is None else config.dataset_slice
    if args.only_plot:
        subprocess.run(
            [
                "Rscript",
                "../src_figure/ov_difference.R",
                f"../results/ov_difference/{config.model_name}_{data_slice_name}",
            ]
        )
        return

    print("Running ov difference")
    ov = OV(dataset, model, config.batch_size)
    dataframe = ov.run(normalize_logit=config.normalize_logit)
    save_dataframe(
        f"../results/ov_difference/{config.model_name}_{data_slice_name}",
        "ov_difference_data",
        dataframe,
    )

    if config.produce_plots:
        # run the R script
        subprocess.run(
            [
                "Rscript",
                "../src_figure/ov_difference.R",
                f"../results/ov_difference/{config.model_name}_{data_slice_name}",
            ]
        )


def ablate(model, dataset, config, args):
    data_slice_name = "full" if config.dataset_slice is None else config.dataset_slice
    if args.only_plot:
        subprocess.run(
            [
                "Rscript",
                "../src_figure/ablation.R",
                f"../results/ablation/{config.model_name}_{data_slice_name}",
                f"{config.std_dev}",
            ]
        )
        return
    ablator = Ablate(dataset, model, config.batch_size)
    dataframe = ablator.run_all(normalize_logit=config.normalize_logit)
    save_dataframe(
        f"../results/ablation/{config.model_name}_{data_slice_name}",
        "ablation_data",
        dataframe,
    )

    if config.produce_plots:
        # run the R script
        subprocess.run(
            [
                "Rscript",
                "../src_figure/ablation.R",
                f"../results/ablation/{config.model_name}_{data_slice_name}",
                f"{config.std_dev}",
            ]
        )
import sys
import io

class CustomOutputStream(io.StringIO):
    def __init__(self, live, index, status, experiments):
        super().__init__()
        self.live = live
        self.index = index
        self.status = status
        self.experiments = experiments

    def write(self, text):
        super().write(text)
        self.status[self.index] = text
        self.live.update(display_experiments(self.experiments, self.status))


def main(args):
    config = Config().from_args(args)
    console.print(display_config(config))
    model = WrapHookedTransformer.from_pretrained(config.model_name)
    dataset = TlensDataset(config.dataset_path, model, slice=config.dataset_slice)

    experiments = []
    if args.logit_attribution:
        experiments.append(logit_attribution)
    if args.logit_lens:
        experiments.append(logit_lens)
    if args.ov_diff:
        experiments.append(ov_difference)
    if args.ablate:
        experiments.append(ablate)
    if args.all:
        experiments = [logit_attribution, logit_lens, ov_difference, ablate]

    status = ["Pending" for _ in experiments]


    for i, experiment in enumerate(experiments):
        status[i] = "Running"
        table = display_experiments(experiments, status)
        console.print(table)
        experiment(model, dataset, config, args)
        status[i] = "Done"
    

if __name__ == "__main__":
    config_defaults = Config()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=config_defaults.model_name)
    parser.add_argument("--slice", type=int, default=config_defaults.dataset_slice)
    parser.add_argument(
        "--no-plot", dest="produce_plots", action="store_false", default=True
    )
    parser.add_argument("--batch", type=int, default=config_defaults.batch_size)
    parser.add_argument("--only-plot", action="store_true")
    parser.add_argument("--std-dev", action="store_true")

    parser.add_argument("--logit-attribution", action="store_true")
    parser.add_argument("--logit_lens", action="store_true")
    parser.add_argument("--ov-diff", action="store_true")
    parser.add_argument("--ablate", action="store_true")
    parser.add_argument("--all", action="store_true")
    
    args = parser.parse_args()
    main(args)
