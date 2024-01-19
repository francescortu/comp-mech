import json
import openai
from src.config import openai_api_key
from typing import List, Dict, Optional
from tqdm import tqdm

import random


class ContextDatasetGPT:
    def __init__(self, model_name: str, n_samples: Optional[int] = 10):
        self.model_name = model_name
        self.original_dataset = self.load_original_dataset(model_name)
        if n_samples is not None:
            random.seed(42)
            self.n_samples = n_samples
            self.original_dataset = random.sample(self.original_dataset, n_samples)
        # sample n_samples from the original dataset randomly

    def load_original_dataset(self, model_name: str):
        with open(f"../data/full_data_sampled_{model_name}.json", "r") as f:
            return json.load(f)

    def process_dataset(self):
        new_dataset = []
        self.new_dataset, index = self.load_checkpoints()
        for i,d in tqdm(
            enumerate(self.original_dataset[index:]),
            total=len(self.original_dataset[index:]),
            desc="Adding Context using OpenAI API",
        ):
            new_dataset.append(self.process_sample(d))
            if i % 1000 == 0:
                self.save_dataset(new_dataset, checkpoint=True)
        return new_dataset

    def load_checkpoints(self):
        #check if there is a checkpoint
        try:
            with open(f"../data/checkpoints/context_dataset_{self.model_name}_checkpoint.json", "r") as f:
                print("CHECKPOINT FOUND: loading checkpoint...")
                return json.load(f), len(json.load(f))
        except:
            return self.original_dataset, 0

    def process_sample(self, sample: Dict) -> Dict:
        new_sample = {}
        new_sample["target_true"] = sample["target_true"]
        new_sample["target_new"] = sample["target_new"]
        new_sample["base_prompt"] = sample["base_prompt"]
        new_sample["prompt"] = self.add_random_context(sample)
        # compute subject position
        return new_sample

    def add_random_context(self, sample: Dict) -> str:
        context = self.generate_random_context(context_target=sample["target_new"])
        formatter = " "
        new_template = f"{formatter}{context} {sample['base_prompt']}"
        return new_template

    def generate_random_context(self, context_target: str) -> str:
        PROMPT = f"Generate a random sentence about {context_target}. The sentence should contain the word {context_target}:"
        openai.api_key = openai_api_key

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo-1106",  #! TODO: set temperature to 0    use a specifi checkpoint of the model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": PROMPT},
            ],
            temperature=0,
        )
        string_response = response.choices[0].message.content
        # parsed_string = string_response.split('"')[1].split('"')[0]
        return string_response  # type: ignore

    def save_dataset(self, dataset: List[Dict], checkpoint: Optional[bool] = False):
        if checkpoint:
            with open(f"../data/checkpoints/context_dataset_{self.model_name}_checkpoint.json", "w") as f:
                json.dump(dataset, f, indent=4)
                return
        with open(f"../data/context_dataset_{self.model_name}.json", "w") as f:
            json.dump(dataset, f, indent=4)


    def run(self):
        dataset = self.process_dataset()
        self.save_dataset(dataset)
