import sys
import os


# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory (..) to sys.path
sys.path.append(os.path.join(script_dir, ".."))

# Optionally, add the 'src' directory directly
sys.path.append(os.path.join(script_dir, "..", "src"))

from src.score_models import EvaluateMechanism  # noqa: E402
from src.dataset import HFDataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
import torch  # noqa: E402
import os  # noqa: E402
from argparse import ArgumentParser  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from typing import List  # noqa: E402
from src.utils import check_dataset_and_sample  # noqa: E402

NUM_SAMPLES = 10
FAMILY_NAME = "gpt2"


@dataclass
class Options:
    models_name: List[str] = field(
        default_factory=lambda: ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
    )
    premise: List[str] = field(
        default_factory=lambda: ["Redefine", "Assume", "Suppose", "Context"]
    )
    similarity: List[bool] = field(default_factory=lambda: [True, False])
    interval: List[int] = field(default_factory=lambda: [4, 3, 2, 1])


@dataclass
class LaunchConfig:
    model_name: str
    hf_model_name: str
    similarity: bool
    interval: int
    family_name: str
    premise: str = "Redefine"
    num_samples: int = 1
    batch_size: int = 50


def launch_evaluation(config: LaunchConfig):
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Loading model", config.model_name)
    print("Launch config", config)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
    )
    if len(config.model_name.split("/")) > 1:
        save_name = config.model_name.split("/")[1]
    else:
        save_name = config.model_name
    dataset_path = f"../data/full_data_sampled_{save_name}.json"
    check_dataset_and_sample(dataset_path, config.model_name, config.hf_model_name)
    dataset = HFDataset(
        model=config.model_name,
        tokenizer=tokenizer,
        path=dataset_path,
        slice=10000,
        premise=config.premise,
        similarity=(config.similarity, config.interval, "logit"),
    )
    print("Dataset loaded")
    evaluator = EvaluateMechanism(
        model_name=config.model_name,
        dataset=dataset,
        device=DEVICE,
        batch_size=config.batch_size,
        similarity=(config.similarity, config.interval, "logit"),
        premise=config.premise,
        family_name=config.family_name,
        num_samples=config.num_samples,
    )
    evaluator.evaluate_all()


def evaluate_size(options: Options):
    for model_name in options.models_name:
        launch_config = LaunchConfig(
            model_name=model_name,
            hf_model_name=model_name,
            similarity=False,
            interval=0,
            family_name=FAMILY_NAME,
            num_samples=NUM_SAMPLES,
        )
        launch_evaluation(launch_config)


def evaluate_premise(options: Options):
    for model_name in options.models_name:
        for premise in options.premise:
            launch_config = LaunchConfig(
                model_name=model_name,
                hf_model_name=model_name,
                similarity=False,
                interval=0,
                family_name=FAMILY_NAME,
                premise=premise,
                num_samples=NUM_SAMPLES,
            )
            launch_evaluation(launch_config)


def evaluate_similarity_default_premise(options: Options):
    for model_name in options.models_name:
        for interval in options.interval:
            launch_config = LaunchConfig(
                model_name=model_name,
                hf_model_name=model_name,
                similarity=True,
                interval=interval,
                family_name=FAMILY_NAME,
                num_samples=NUM_SAMPLES,
            )
            launch_evaluation(launch_config)


def evaluate_similarity_all_premise(options: Options):
    for model_name in options.models_name:
        for premise in options.premise:
            for interval in options.interval:
                launch_config = LaunchConfig(
                    model_name=model_name,
                    hf_model_name=model_name,
                    similarity=True,
                    interval=interval,
                    family_name=FAMILY_NAME,
                    premise=premise,
                    num_samples=NUM_SAMPLES,
                )
                launch_evaluation(launch_config)


def main(args):
    experiments = []
    if args.size:
        experiments.append(evaluate_size)
    if args.premise and args.similarity:
        experiments.append(evaluate_similarity_all_premise)
    if args.premise and not args.similarity:
        experiments.append(evaluate_premise)
    if args.similarity and not args.premise:
        experiments.append(evaluate_similarity_default_premise)
    if args.all:
        experiments = [evaluate_size, evaluate_premise, evaluate_similarity_all_premise]

    options = Options()
    for experiment in experiments:
        print("Running experiment", experiment.__name__)
        experiment(options)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--size", action="store_true")
    parser.add_argument("--premise", action="store_true")
    parser.add_argument("--similarity", action="store_true")
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    NUM_SAMPLES = args.num_samples
    main(args)
