import sys
import os
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory (..) to sys.path
sys.path.append(os.path.join(script_dir, ".."))
# Optionally, add the 'src' directory directly
sys.path.append(os.path.join(script_dir, "..", "src"))
from src.model import WrapHookedTransformer
from src.dataset import TlensDataset
from src.experiment import LogitAttribution, LogitLens
from dataclasses import dataclass, field
from typing import Optional
import subprocess
import argparse 



@dataclass
class Config:
    model_name: str = "gpt2"
    batch_size: int = 10
    dataset_path: str = f"../data/full_data_sampled_{model_name}.json"
    dataset_slice: Optional[int] = None
    produce_plots: bool = False
    
    @classmethod
    def from_args(cls, args):
        return cls(
            model_name=args.model_name,
            batch_size=args.batch_size,
            dataset_path=f"../data/full_data_sampled_{args.model_name}.json",
            dataset_slice=args.dataset_slice,
            produce_plots=args.produce_plots
        )
    
@ dataclass
class logit_attribution_config:
    std_dev: int = 0 #0 False, 1 True

@dataclass
class logit_lens_config():
    component: str = "resid_post"
    return_index: bool = False
    normalize: str = "none"
    

def logit_attribution(model, dataset,config, args):
    print("Running logit attribution")
    attributor = LogitAttribution(dataset, model, config.batch_size)
    dataframe = attributor.run()
    dataset_slice_name = "full" if config.dataset_slice is None else config.dataset_slice
    # check if the folder exists "../results/logit_attribution/model_name_dataset_slice"
    if not os.path.exists(f"../results/logit_attribution/{config.model_name}_{dataset_slice_name}"):
        os.makedirs(f"../results/logit_attribution/{config.model_name}_{dataset_slice_name}")
    
    dataframe.to_csv(f"../results/logit_attribution/{config.model_name}_{dataset_slice_name}/logit_attribution_data.csv", index=False)
    if config.produce_plots:
        # run the R script
        subprocess.run(["Rscript", "../src_figure/logit_attribution.R", f"../results/logit_attribution/{config.model_name}_{dataset_slice_name}", f"{logit_attribution_config.std_dev}"])

def logit_lens(model, dataset, config, args):
    logit_lens_cnfg = logit_lens_config()
    print("Running logit lens")
    logit_lens = LogitLens(dataset, model, config.batch_size)
    dataframe = logit_lens.run(logit_lens_cnfg.component, logit_lens_cnfg.return_index, logit_lens_cnfg.normalize)
    data_slice_name = "full" if config.dataset_slice is None else config.dataset_slice
    # check if the folder exists "../results/logit_lens/model_name_dataset_slice"ù
    if not os.path.exists(f"../results/logit_lens/{config.model_name}_{data_slice_name}"):
        os.makedirs(f"../results/logit_lens/{config.model_name}_{data_slice_name}")
        
    dataframe.to_csv(f"../results/logit_lens/{config.model_name}_{data_slice_name}/logit_lens_data.csv", index=False)
    if config.produce_plots:
        # run the R script
        subprocess.run(["Rscript", "../src_figure/logit_lens.R", f"../results/logit_lens/{config.model_name}_{data_slice_name}"])


def main(args):
    config = Config().from_args(args)
    print("Config", config)
    model = WrapHookedTransformer.from_pretrained(config.model_name)
    dataset = TlensDataset(config.dataset_path, model, slice=config.dataset_slice)
    
    print("Running experiment on", config.model_name, "with", config.dataset_slice, "samples")
    #create the list of experiments to run
    experiments = []
    if args.logit_attribution:
        experiments.append(logit_attribution)
    if args.logit_lens:
        experiments.append(logit_lens)
        
    for experiment in experiments:
        print("Running experiment", experiment.__name__)
        experiment(model, dataset, config, args)
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--dataset_slice", type=int, default=None)
    parser.add_argument("--produce_plots", action="store_true")
    parser.add_argument("--logit_attribution", action="store_true")
    parser.add_argument("--logit_lens", action="store_true")
    parser.add_argument("--batch_size", type=int, default=10)
    args = parser.parse_args()
    main(args)
