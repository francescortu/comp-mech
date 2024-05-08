import sys
import os  # noqa: F401
import json  # noqa:  F811

import torch  # noqa: F401
from torch.utils.data import DataLoader  # noqa: F401
from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402
from tqdm import tqdm  # noqa: F401
from typing import Literal, Optional, Tuple  # noqa: F401
from dataclasses import dataclass  # noqa: F401


sys.path.append("..")
sys.path.append("../src")
sys.path.append("../data")
from Src.dataset import BaseDataset  # noqa: E402
from Src.model import BaseModel  # noqa: E402

FILENAME = "../results/{}_evaluate_mechanism_data_sampling.csv"

class EvaluateMechanism:
    def __init__(
        self,
        model: BaseModel,
        dataset: BaseDataset,
        device="cpu",
        batch_size=100,
        similarity: Tuple[
            bool, int, Literal["data-sampling", "modify-self-similarity", "self-similarity"]
        ] = (True, 4, "self-similarity"),
        premise="Redefine",
        family_name: Optional[str] = None,
        num_samples=1,
    ):
        self.tokenizer = model.get_tokenizer()
        self.model = model
        self.model = self.model.to(device)
        self.model_name = model.cfg.model_name
        self.dataset = dataset
        # self.lenghts = self.dataset.get_lengths()
        self.device = device
        self.batch_size = batch_size
        self.similarity = similarity
        self.premise = premise
        self.family_name = family_name
        self.n_samples = num_samples
        print("Model device", self.model.device)

    def update(
        self,
        dataset: BaseDataset,
        similarity: Tuple[bool, int, Literal["self-similarity", "modify-self-similarity","data-sampling"]],
        premise: str,
    ):
        self.dataset = dataset
        self.similarity = similarity
        self.premise = premise

    def check_prediction(self, logit, target):
        probs = torch.softmax(logit, dim=-1)[:, -1, :]
        # count the number of times the model predicts the target[:, 0] or target[:, 1]
        num_samples = target.shape[0]
        target_true = 0
        target_false = 0
        other = 0
        target_true_indices = []
        target_false_indices = []
        other_indices = []
        for i in range(num_samples):
            if torch.argmax(probs[i]) == target[i, 0]:
                # print("DEBUG:", self.tokenizer.decode(target[i, 0]), self.tokenizer.decode(torch.argmax(probs[i])))
                target_true += 1
                target_true_indices.append(i)
            elif torch.argmax(probs[i]) == target[i, 1]:
                target_false += 1
                target_false_indices.append(i)
            else:
                other += 1
                other_indices.append(i)
        return target_true_indices, target_false_indices, other_indices

    def evaluate(self, length, dataloader):

        target_true, target_false, other = 0, 0, 0
        n_batch = len(dataloader)
        all_true_indices = []
        all_false_indices = []
        all_other_indices = []

        idx = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            logits = self.model(input_ids)["logits"]
            count = self.check_prediction(logits, batch["target"])
            target_true += len(count[0])
            target_false += len(count[1])
            other += len(count[2])

            all_true_indices.extend(
                [
                    self.dataset.original_index[i + idx * self.batch_size]
                    for i in count[0]
                ]
            )
            all_false_indices.extend(
                [
                    self.dataset.original_index[i + idx * self.batch_size]
                    for i in count[1]
                ]
            )
            all_other_indices.extend(
                [
                    self.dataset.original_index[i + idx * self.batch_size]
                    for i in count[2]
                ]
            )
            idx += 1
        return (
            target_true,
            target_false,
            other,
            all_true_indices,
            all_false_indices,
            all_other_indices,
        )

    def evaluate_all(self):
        target_true, target_false, other = [], [], []
        print("SAMPLES", self.n_samples)
        for sample in tqdm(range(self.n_samples), total=self.n_samples, desc="Samples"):
            self.dataset.reset(new_similarity_level = self.similarity[1])
            target_true_tmp, target_false_tmp, other_tmp = 0, 0, 0
            all_true_indices = []
            all_false_indices = []
            all_other_indices = []
            for length in tqdm(self.dataset.get_lengths()):
                self.dataset.set_len(length)
                if len(self.dataset) == 0:
                    continue
                dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
                result = self.evaluate(length, dataloader)
                target_true_tmp += result[0]
                target_false_tmp += result[1]
                other_tmp += result[2]

                # assert duplicates
                all_index = result[3] + result[4] + result[5]
                assert len(all_index) == len(
                    set(all_index)
                ), "Duplicates in the indices"

                all_true_indices.extend(result[3])
                all_false_indices.extend(result[4])
                all_other_indices.extend(result[5])

            # add the results of the sample to the total
            target_true.append(target_true_tmp)
            target_false.append(target_false_tmp)
            other.append(other_tmp)

        print("Target true length", len(target_true))
        # average the results over the number of samples
        Target_true = target_true
        Target_false = target_false
        Other = other
        print("target true", Target_true)

        target_true = torch.mean(torch.tensor(Target_true).float())
        target_false = torch.mean(torch.tensor(Target_false).float())
        other = torch.mean(torch.tensor(Other).float())

        target_true_std = torch.std(torch.tensor(Target_true).float())
        target_false_std = torch.std(torch.tensor(Target_false).float())
        other_std = torch.std(torch.tensor(Other).float())

        print(
            f"Total: Target True: {target_true}, Target False: {target_false}, Other: {other}"
        )
        print(
            f"Total: Target True std: {target_true_std}, Target False std: {target_false_std}, Other std: {other_std}"
        )

        # index = torch.cat(index, dim=1)

        if len(self.model_name.split("/")) > 1:
            save_name = self.model_name.split("/")[1]
        else:
            save_name = self.model_name
        if self.similarity[0]:
            save_name += "similarity"
        # save results

        filename = FILENAME.format(self.family_name)
        # if file not exists, create it and write the header
        if not os.path.isfile(filename):
            with open(filename, "w") as file:
                file.write(
                    "model_name,orthogonalize,premise,interval,similarity_type,target_true,target_false,other,target_true_std,target_false_std,other_std\n"
                )

        with open(filename, "a+") as file:
            file.seek(0)
            # if there is aleardy a line with the same model_name and orthogonalize, delete it
            lines = file.readlines()
            # Check if a line with the same model_name and orthogonalize exists
            line_exists = any(
                line.split(",")[0] == self.model_name
                and line.split(",")[1] == str(self.similarity[0])
                and line.split(",")[2] == self.premise
                and line.split(",")[3] == self.similarity[1]
                and line.split(",")[4] == self.similarity[2]
                for line in lines
            )

            # If the line exists, remove it
            if line_exists:
                lines = [
                    line
                    for line in lines
                    if not (
                        line.split(",")[0] == self.model_name
                        and line.split(",")[1]
                        == str(
                            self.similarity[0]
                            and line.split(",")[2] == self.premise
                            and line.split(",")[3] == self.similarity[1]
                            and line.split(",")[4] == self.similarity[2]
                        )
                    )
                ]

                # Rewrite the file without the removed line
                file.seek(0)  # Move the file pointer to the start of the file
                file.truncate()  # Truncate the file (i.e., remove all content)
                file.writelines(lines)  # Write the updated lines back to the file
            file.write(
                f"{self.model_name},{self.similarity[0]},{self.premise},{self.similarity[1]},{self.similarity[2]} ,{target_true},{target_false},{other},{target_true_std},{target_false_std},{other_std}\n"
            )

        # save indices
        if not os.path.isdir(f"../results/{self.family_name}_evaluate_mechs_indices"):
            # if the directory does not exist, create it
            os.makedirs(f"../results/{self.family_name}_evaluate_mechs_indices")

        with open(
            f"../results/{self.family_name}_evaluate_mechs_indices/{save_name}_evaluate_mechanism_indices.json",
            "w",
        ) as file:
            json.dump(
                {
                    "target_true": all_true_indices,  # type: ignore
                    "target_false": all_false_indices,  # type: ignore
                    "other": all_other_indices,  # type: ignore
                },
                file,
            )

        return target_true, target_false, other
