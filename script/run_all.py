# Standard library imports
import sys
import os
sys.path.append(os.path.abspath(os.path.join("..")))
sys.path.append(os.path.abspath(os.path.join("../src")))
sys.path.append(os.path.abspath(os.path.join("../data")))
from dataclasses import dataclass
from src.config import hf_access_token, hf_model_cache_dir # noqa: E402
os.environ["HF_HOME"] = hf_model_cache_dir
from re import A
import io
import subprocess
from typing import Optional, Literal, Union


# Third-party library imports
from rich.console import Console
import argparse
import logging
import torch
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer

# Local application/library specific imports

from src.dataset import TlensDataset  # noqa: E402
from src.experiment import LogitAttribution, LogitLens, OV, Ablate, HeadPattern  # noqa: E402
from src.model import WrapHookedTransformer  # noqa: E402
from src.utils import display_config, display_experiments, check_dataset_and_sample  # noqa: E402
console = Console()
# set logging level to suppress warnings
logging.basicConfig(level=logging.ERROR)




def get_hf_model_name(model_name):
    if "Llama" in model_name:
        return "meta-llama/" + model_name
    elif "opt" in model_name:
        return "facebook/" + model_name
    else:
        print("No HF model name found for model name: ", model_name)
    return model_name


@dataclass
class Config:
    model_name: str = "gpt2"
    hf_model_name: str = "gpt2"
    batch_size: int = 10
    dataset_path: str = f"../data/full_data_sampled_{model_name}.json"
    dataset_slice: Optional[int] = None
    produce_plots: bool = True
    normalize_logit: Literal["none", "softmax", "log_softmax"] = "none"
    std_dev: int = 1  # 0 False, 1 True
    total_effect: bool = False
    up_to_layer: Union[int, str] = "all"

    @classmethod
    def from_args(cls, args):
        return cls(
            model_name=args.model_name,
            batch_size=args.batch,
            dataset_path=f"../data/full_data_sampled_{args.model_name}.json",
            dataset_slice=args.slice,
            produce_plots=args.produce_plots,
            std_dev=1 if not args.std_dev else 0,
            total_effect=args.total_effect if args.total_effect else False,
            hf_model_name= get_hf_model_name(args.model_name)
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
    dataset_slice_name = (
        dataset_slice_name if config.up_to_layer == "all" else f"{dataset_slice_name}_layer_{config.up_to_layer}"
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
    dataframe = attributor.run(normalize_logit=config.normalize_logit, up_to_layer=config.up_to_layer)
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
    data_slice_name = f"{data_slice_name}_total_effect" if config.total_effect else data_slice_name
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
    dataframe = ablator.run_all(normalize_logit=config.normalize_logit, total_effect=args.total_effect)
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
        
def pattern(model, dataset, config, args):
    data_slice_name = "full" if config.dataset_slice is None else config.dataset_slice
    if args.only_plot:
        subprocess.run(
            [
                "Rscript",
                "../src_figure/head_pattern.R",
                f"../results/head_pattern/{config.model_name}_{data_slice_name}",
            ]
        )
        return
    print("Running head pattern")
    pattern = HeadPattern(dataset, model, config.batch_size)
    dataframe = pattern.run()
    save_dataframe(
        f"../results/head_pattern/{config.model_name}_{data_slice_name}",
        "head_pattern_data",
        dataframe,
    )

    if config.produce_plots:
        # run the R script
        subprocess.run(
            [
                "Rscript",
                "../src_figure/head_pattern.R",
                f"../results/head_pattern/{config.model_name}_{data_slice_name}",
            ]
        )

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

def load_model(config) -> Union[WrapHookedTransformer, HookedTransformer]:
    if config.model_name == "Llama-2-7b-hf":
        tokenizer = LlamaTokenizer.from_pretrained(config.hf_model_name, use_auth_token = hf_access_token,)
        model = LlamaForCausalLM.from_pretrained(config.hf_model_name, use_auth_token = hf_access_token, low_cpu_mem_usage=True)
        model = WrapHookedTransformer.from_pretrained(config.hf_model_name, tokenizer=tokenizer, fold_ln=False, hf_model=model, device="cuda")
        #model = model.to("cuda")
        return model # type: ignore
    model = WrapHookedTransformer.from_pretrained(config.model_name)
    return model

def main(args):
    config = Config().from_args(args)
    console.print(display_config(config))
    check_dataset_and_sample(config.dataset_path, config.model_name, config.hf_model_name)
    model = load_model(config)
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
    if args.pattern:
        experiments.append(pattern)
    if args.all:
        experiments = [logit_attribution, logit_lens, ov_difference, ablate, pattern]

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
    parser.add_argument("--total-effect", action="store_true")
    parser.add_argument("--pattern", action="store_true")
    parser.add_argument("--all", action="store_true")
    
    args = parser.parse_args()
    main(args)
