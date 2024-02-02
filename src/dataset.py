from abc import abstractmethod
from genericpath import isfile
from math import log

from gensim.models import Word2Vec
import gensim.downloader as api
from matplotlib.pyplot import cla
import torch
from torch._tensor import Tensor
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import random
from typing import Literal, Optional, Tuple, Union, List, Dict, Any
from src.model import WrapHookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
import logging
import os

import time
from multiprocessing import Pool, process, set_start_method
from typing import Tuple
from functools import partial
import random
from collections import defaultdict


class BaseDataset(Dataset):
    def __init__(
        self,
        path: str,
        slice: Optional[int] = None,
        start: Optional[int] = None,
        experiment: Literal["copyVSfact", "contextVSfact"] = "copyVSfact",
        premise: str = "Redefine:",
        similarity: Tuple[
            bool, int, Literal["word2vec", "logit", "self-similarity"]
        ] = (
            False,
            0,
            "logit",
        ),
        family_name: str = "gpt2",
    ):
        self.full_data = json.load(open(path))
        if slice is not None:
            self.full_data = self.full_data[:slice]
        if start is not None:
            self.full_data = self.full_data[start:]
        self.premise = premise
        self.similarity = similarity
        self.experiment = experiment
        if self.similarity[0] is True:
            # dict path is the path to the similarity score dict
            self.dict_path = f"../data/similarity_score_{family_name}_w2v.pt"

        # if the file exist, load it
        if self.similarity[0] and self.similarity[2] != "self-similarity":
            print(f"Loading similarity score dict {self.dict_path}")
            if os.path.isfile(self.dict_path):
                self.similarity_score_dict = torch.load(self.dict_path)
            else:
                self.similarity_score_dict = {}
        if self.experiment == "contextVSfact":
            if self.similarity[0] is True:
                if self.similarity[2] == "self-similarity":
                    pass
                else:
                    raise ValueError(
                        "only self-similarity is supported for contextVSfact experiment"
                    )

        if similarity[0] is True:
            similarity_path = (
                path.split(".json")[0] + f"_similarity_{similarity[2]}.json"
                if slice is None
                else path.split(".json")[0]
                + f"_similarity_{similarity[2]}_{slice}.json"
            )
            self.similarity_path = similarity_path
            if similarity[2] in ["word2vec", "logit"]:
                print("Search similarity path:", similarity_path)
                if os.path.isfile(similarity_path):
                    print("Similarity file found, loading it")
                    self.similarity_data = json.load(open(similarity_path))
                else:
                    print("Similarity file not found, generating it")
                    self.similarity_path = similarity_path
                    self.similarity_data = self.generate_similarity_dataset(
                        similarity[2]
                    )
                    json.dump(self.full_data, open(similarity_path, "w"), indent=2)
            elif similarity[2] == "self-similarity":
                self.similarity_data = self.generate_similarity_dataset(similarity[2])
            else:
                raise ValueError(
                    f"similarity type must be in ['word2vec', 'logit', 'self-similarity'] while it is {similarity[2]}"
                )

            self.full_data = self.similarity_data
        self.lengths = self._get_lenghts_and_tokenize()

        self.prompts = []
        self.tokenized_prompts = []
        self.targets = []
        self.obj_pos = []

    def reset(
        self,
        similarity: Tuple[bool, int, Literal["word2vec", "logit", "self-similarity"]],
    ):
        if similarity is not None:
            self.similarity = similarity
            self.full_data = self.similarity_data
        self.lengths = self._get_lenghts_and_tokenize()
        self.prompts = []
        self.tokenized_prompts = []
        self.targets = []
        self.obj_pos = []

    def update(
        self,
        premise: str,
        similarity: Tuple[bool, int, Literal["word2vec", "logit", "self-similarity"]],
    ):
        print(
            f"Updating dataset from {self.premise} to {premise} and {self.similarity} to {similarity}"
        )
        self.similarity = similarity
        self.premise = premise
        self.full_data = self.similarity_data
        self.prompts = []
        self.tokenized_prompts = []
        self.targets = []
        self.obj_pos = []
        self.lengths = []
        self.lengths = self._get_lenghts_and_tokenize()

    @classmethod
    def set_init_argument(
        cls,
        path: str,
        slice: Optional[int] = None,
        premise: str = "Redefine:",
        similarity: Tuple[
            bool, int, Literal["word2vec", "logit", "self-similarity"]
        ] = (
            False,
            0,
            "logit",
        ),
    ):
        cls.init_args = locals()
        return cls

    def __len__(self):
        if len(self.prompts) == 0:
            raise ValueError("Dataset is empty: please call set_len() first")
        return len(self.prompts)

    def __getitem__(self, idx):
        if len(self.prompts) == 0:
            raise ValueError("Dataset is empty: please call set_len() first")
        else:
            return {
                "prompt": self.prompts[idx],
                "input_ids": self.tokenized_prompts[idx],
                "target": self.targets[idx],
                "obj_pos": self.obj_pos[idx],
                "subj_pos": self.subj_pos[idx],
            }

    def get_lengths(self):
        return self.lengths

    def slice_to_fit_batch(self, batch_size: int):
        max_data_size = (len(self.data) // batch_size) * batch_size
        self.slice(max_data_size)

    def slice(self, slice: int):
        raise NotImplementedError

    def _get_lenghts_and_tokenize(self):
        """
        Tokenize the prompts and apply the similarity if needed along with the premise.
        Return the lenghts of the dataset
        """

        lenghts = []
        log_data = []
        for d in self.full_data:
            while True:
                if self.similarity[0] is True:
                    if self.similarity[2] in ["word2vec", "logit"]:
                        target_new = self.get_similar_token(d, self.similarity[1])
                    elif self.similarity[2] == "self-similarity":
                        target_new = d["target_new"]
                        # if the similarity_group is not the same as the similarity level, continue
                    else:
                        raise ValueError(
                            f"similarity type must be in ['word2vec', 'logit', 'self-similarity'] while it is {self.similarity[2]}"
                        )
                else:
                    target_new = d["target_new"]
                if self.experiment == "copyVSfact":
                    prompt = d["template"].format(self.premise, target_new)
                elif self.experiment == "contextVSfact":
                    if d["prompt"][0] == " ":
                        prompt = d["prompt"]
                    else:
                        prompt = " " + d["prompt"]
                else:
                    raise ValueError(
                        f"experiment must be either 'copyVSfact' or 'contextVSfact' while it is {self.experiment}"
                    )
                d["prompt"] = prompt
                d["tokenized_prompt"] = self._tokenize_prompt(prompt, True)  # ( L)
                target_new_token = self._tokenize_target(
                    target_new, False
                ).cuda()  # (1)
                d["target_new_token"] = target_new_token
                target_true_token = self._tokenize_target(
                    d["target_true"], False
                ).cuda()  # (1)
                d["target_true_token"] = target_true_token
                d["targets"] = torch.cat(
                    [target_true_token, target_new_token], dim=0
                )  # (2)
                if self.experiment == "contextVSfact":
                    subject_token = self._tokenize_prompt(
                        " " + d["base_prompt"], False
                    )[0]
                    for i in range(len(d["tokenized_prompt"]), 0, -1):
                        if d["tokenized_prompt"][i - 1] == subject_token:
                            d["subj_position"] = i - 1
                            break

                try:
                    obj_pos_indices = (
                        d["tokenized_prompt"].cpu() == target_new_token.cpu()
                    ).nonzero(as_tuple=True)[0]
                    if obj_pos_indices.size(0) > 0:
                        d["obj_pos"] = obj_pos_indices[0].item()
                    else:
                        if (
                            self.similarity[0] is True
                            and self.experiment != "contextVSfact"
                        ):
                            continue  # Resample if similarity is true
                        else:
                            log_data.append(
                                {
                                    "prompt": d["prompt"],
                                    "target_new": target_new,
                                }
                            )
                            # remove the sample if the target is not found in the prompt
                            break
                            # raise ValueError(
                            #     "Target not found in prompt"
                            # )  # Throw exception otherwise
                    d["length"] = d["tokenized_prompt"].shape[0]
                    if d["length"] not in lenghts:
                        lenghts.append(d["length"])
                    break

                except RuntimeError:
                    if self.similarity[0] is True:
                        continue  # Resample if similarity is true
                    else:
                        raise  # Throw exception otherwise

        if len(log_data) > 0:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            log_dir = "../logs"
            os.makedirs(log_dir, exist_ok=True)
            with open(f"{log_dir}/log_{timestamp}.json", "w") as f:
                json.dump(log_data, f, indent=2)
            for logdata in log_data:
                for d in self.full_data[:]:
                    if d["prompt"] == logdata["prompt"]:
                        self.full_data.remove(d)

        if self.similarity[2] == "self-similarity":
            # remove the element in a different similarity group
            self.full_data = [
                d for d in self.full_data if d["similarity_group"] == self.similarity[1]
            ]

            # check if there are lenghts with zero elements
            for lenght in lenghts[:]:
                if len([d for d in self.full_data if d["length"] == lenght]) == 0:
                    lenghts.remove(lenght)

        self._clear_cache()  # free up memory, we don't need the model anymore
        print(len(self.full_data))
        return lenghts

    def generate_similarity_dataset(
        self, method: Literal["word2vec", "logit", "self-similarity"]
    ) -> List[dict]:
        if method == "word2vec":
            return self.generate_similarity_dataset_word2vec()
        elif method == "logit":
            return self.generate_similarity_dataset_logit()
        elif method == "self-similarity":
            return self.generate_self_similarity_dataset()
        else:
            raise ValueError("method must be either 'word2vec' or 'logit'")

    def generate_self_similarity_dataset(self) -> List[dict]:
        word2vec = api.load("word2vec-google-news-300")
        similarity_score_list = []
        for d in tqdm(
            self.full_data,
            desc="Generating self similarity tokens (word2vec)",
            total=len(self.full_data),
        ):
            base_target = d["target_true"]
            other_target = d["target_new"]
            # remove first space if present
            if other_target[0] == " ":
                other_target = other_target[1:]
            if base_target[0] == " ":
                base_target = base_target[1:]

            # compute similarity
            try:
                similarity_score = word2vec.similarity(base_target, other_target)  # type: ignore
                similarity_score_list.append(similarity_score)
            except:
                similarity_score = -100
            # save the similarity score
            d["similarity_score"] = similarity_score

        # compute three thresholds based on the similarity score
        similarity_score_list = torch.tensor(similarity_score_list)
        # save the distribution of the similarity score
        path = self.similarity_path.split(".json")[0] + ".pt"

        similarity_score_list = similarity_score_list.sort(descending=False).values
        # divide the similarity score in group of 1000 values each
        num_of_samples = len(similarity_score_list)
        num_of_group = num_of_samples // 1000
        print("DEBUG: Num_samples", num_of_samples)
        group_intervals = [
            similarity_score_list[(i + 1) * 1000] for i in range(num_of_group)
        ]
        print("DEBUG: group interval", group_intervals)
        for d in self.full_data:
            similarity_score = d["similarity_score"]
            if similarity_score == -100:
                d["similarity_group"] = -100
                continue
            for i in range(num_of_group, -1, -1):
                if similarity_score >= group_intervals[i - 1]:
                    print("DEBUG: similarity score", similarity_score, "group", i, "interval", group_intervals[i - 1])
                    d["similarity_group"] = i
                    break
        return self.full_data

        # torch.save(similarity_score_list, path)
        # # generate 5 groups based on the similarity score
        # (
        #     quartile_1,
        #     quartile_2,
        #     quartile_3,
        #     quartile_4,
        #     quartile_5,
        #     quartile_6,
        # ) = torch.quantile(
        #     similarity_score_list, torch.tensor([0.2, 0.4, 0.6, 0.8, 0.9, 0.95])
        # )
        # # quartile_1, quartile_2, quartile_3 = torch.quantile(similarity_score_list, torch.tensor([0.25, 0.5, 0.75]))

        # ticks = [
        #     0.15,
        #     0.25,
        #     0.35,
        #     # 0.40,
        #     0.45,
        #     # 0.50,
        #     0.55,
        #     # 0.6,
        #     0.65,
        # ]

        # # divide the similarity in group tick
        # for d in self.full_data:
        #     similarity_score = d["similarity_score"]
        #     if similarity_score == -100:
        #         d["similarity_group"] = -100
        #         continue
        #     if similarity_score < ticks[0]:
        #         d["similarity_group"] = 0
        #     elif similarity_score < ticks[1]:
        #         d["similarity_group"] = 1
        #     elif similarity_score < ticks[2]:
        #         d["similarity_group"] = 2
        #     elif similarity_score < ticks[3]:
        #         d["similarity_group"] = 3
        #     elif similarity_score < ticks[4]:
        #         d["similarity_group"] = 4
        #     elif similarity_score < ticks[5]:
        #         d["similarity_group"] = 5
        #     else:
        #         d["similarity_group"] = 6

        # # # assign a group to each data point based on the similarity score
        # # for d in self.full_data:
        # #     similarity_score = d["similarity_score"]
        # #     if similarity_score == -100:
        # #         d["similarity_group"] = -100
        # #     elif similarity_score < quartile_1:
        # #         d["similarity_group"] = 6
        # #     elif similarity_score < quartile_2:
        # #         d["similarity_group"] = 5
        # #     elif similarity_score < quartile_3:
        # #         d["similarity_group"] = 4
        # #     elif similarity_score < quartile_4:
        # #         d["similarity_group"] = 3
        # #     elif similarity_score < quartile_5:
        # #         d["similarity_group"] = 2
        # #     elif similarity_score < quartile_6:
        # #         d["similarity_group"] = 1
        # #     else:
        # #         d["similarity_group"] = 0

        # # for each group, random sample 400 data points and set the other to -100
        # # First, collect data points by their groups
        # grouped_data_points = defaultdict(list)
        # for index, d in enumerate(self.full_data):
        #     grouped_data_points[d["similarity_group"]].append(index)

        # # Now, sample 400 data points from each group and mark the rest as -100
        # for group, indices in grouped_data_points.items():
        #     if (
        #         group == -100
        #     ):  # Skip if the group is already for error-handled data points
        #         continue

        #     # Shuffle the indices to ensure randomness
        #     random.shuffle(indices)

        #     # If the group has more than 400 data points, sample 400, else take all
        #     selected_indices = set(indices[:400])
        #     # Update the groups for non-selected data points
        #     for index in indices:
        #         if index not in selected_indices:
        #             self.full_data[index]["similarity_group"] = -100

        # return self.full_data

        # #assign a group to each data point based on the similarity score
        # for d in self.full_data:
        #     similarity_score = d["similarity_score"]
        #     if similarity_score == -100:
        #         d["similarity_group"] = -100
        #     elif similarity_score < quartile_1:
        #         d["similarity_group"] = 4
        #     elif similarity_score < quartile_2:
        #         d["similarity_group"] = 3
        #     elif similarity_score < quartile_3:
        #         d["similarity_group"] = 2
        #     else:
        #         d["similarity_group"] = 1
        # return self.full_data

    def generate_similarity_dataset_word2vec(self) -> List[dict]:
        word2vec = api.load("word2vec-google-news-300")
        for d in tqdm(
            self.full_data,
            desc="Generating similarity tokens (word2vec)",
            total=len(self.full_data),
        ):
            base_target = d["target_true"]
            all_token_with_similarity = self.compute_similarity_word2vec(
                base_target, word2vec, d["target_new"]
            )
            # save the distribution of the similarity score
            similarity_score = torch.tensor(
                [score for token, score in all_token_with_similarity]
            )
            tokens = [token for token, _ in all_token_with_similarity]

            # torch.save(similarity_score, f"../data/similarity_score/{base_target}.pt")
            #
            # divide the tokens into 4 groups based on the quantile
            # quartile_1, quartile_2, quartile_3, top_2 = torch.quantile(similarity_score, torch.tensor([0.25, 0.5, 0.75, 0.98]))

            # Creating masks for each group
            # masks = {
            #     4: similarity_score < quartile_1,
            #     3: (similarity_score >= quartile_1) & (similarity_score < quartile_2),
            #     2: (similarity_score >= quartile_2) & (similarity_score < quartile_3),
            #     1: (similarity_score >= quartile_3),
            #     0: similarity_score >= top_2,
            #     }

            # Grouping tokens based on masks
            # for i in range(5):
            #    d[f"similar_tokens_{i}"] = [tokens[j] for j in torch.where(masks[i])[0]]

            # torch.save(self.similarity_score_dict, self.dict_path)
            # return self.full_data

            # (
            #     quartile_1,
            #     quartile_2,
            #     quartile_3,
            #     quartile_4,
            #     quartile_5,
            #     quartile_6,
            #     quartile_7,
            # ) = torch.quantile(
            #     similarity_score, torch.tensor([0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.98])
            # )
            # # Creating masks for each group
            # masks = {
            #     6: (similarity_score > quartile_1) & (similarity_score < quartile_2),
            #     5: (similarity_score >= quartile_2) & (similarity_score < quartile_3),
            #     4: (similarity_score >= quartile_3) & (similarity_score < quartile_4),
            #     3: (similarity_score >= quartile_4) & (similarity_score < quartile_5),
            #     2: (similarity_score >= quartile_5) & (similarity_score < quartile_6),
            #     1: (similarity_score >= quartile_6) & (similarity_score < quartile_7),
            #     0: similarity_score >= quartile_7,
            # }

            # # Grouping tokens based on masks
            # for i in range(5):
            #     d[f"similar_tokens_{i}"] = [tokens[j] for j in torch.where(masks[i])[0]]

            ticks = [
                0.15,
                0.25,
                0.35,
                # 0.40,
                0.45,
                # 0.50,
                0.55,
                # 0.6,
                0.65,
            ]

            mask = {
                0: (similarity_score < ticks[0]),
                1: (similarity_score > ticks[0]) & (similarity_score < ticks[1]),
                2: (similarity_score > ticks[1]) & (similarity_score < ticks[2]),
                3: (similarity_score > ticks[2]) & (similarity_score < ticks[3]),
                4: (similarity_score > ticks[3]) & (similarity_score < ticks[4]),
                5: (similarity_score > ticks[4]) & (similarity_score < ticks[5]),
                6: similarity_score > ticks[5],
            }
            for i in range(len(ticks)):
                d[f"similar_tokens_{i}"] = [tokens[j] for j in torch.where(mask[i])[0]]

        torch.save(self.similarity_score_dict, self.dict_path)
        return self.full_data

    # def generate_similarity_dataset_word2vec(self) -> List[dict]:
    #     word2vec = api.load("word2vec-google-news-300")
    #     for d in tqdm(
    #         self.full_data,
    #         desc="Generating similarity tokens (word2vec)",
    #         total=len(self.full_data),
    #     ):
    #         base_target = d["target_true"]
    #         all_token_with_similarity = self.compute_similarity_word2vec(
    #             base_target, word2vec, d["target_new"]
    #         )
    #         # save the distribution of the similarity score
    #         similarity_score = torch.tensor(
    #             [score for token, score in all_token_with_similarity]
    #         )
    #         # sort the tokens by similarity
    #         sorted_tokens = sorted(
    #             all_token_with_similarity, key=lambda x: x[1], reverse=True
    #         )
    #         # write the tokens in the dataset
    #         d["similar_tokens"] = [ (token, round(score.item(), 3)) for token, score in sorted_tokens]

    #     return self.full_data

    @abstractmethod
    def compute_similarity_word2vec(
        self, base_target: str, word2vec, other_target: str
    ) -> List[Tuple[str, float]]:
        pass

    def generate_similarity_dataset_logit(self) -> List[dict]:
        """
        Generate the similarity token from the top predictions of the model
        """
        model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.bfloat16)
        model = model.cuda()
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        for d in tqdm(
            self.full_data,
            desc="Generating similarity tokens",
            total=len(self.full_data),
        ):
            (
                best_string_token_predictions,
                best_token_score,
            ) = self.get_best_string_token_predictions(
                d["base_prompt"], tokenizer, model, 1000
            )
            token_per_similarity_level = self.get_token_per_similarity_level(
                best_string_token_predictions,
            )
            d["similar_tokens_1"] = token_per_similarity_level[1]
            d["similar_tokens_2"] = token_per_similarity_level[2]
            d["similar_tokens_3"] = token_per_similarity_level[3]
            d["similar_tokens_4"] = token_per_similarity_level[4]

        return self.full_data

    def get_best_string_token_predictions(
        self,
        prompt: str,
        tokenizer,
        model,
        n: int = 1000,
    ) -> Tuple[List[str], Tensor]:
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(model.device)  # type: ignore
        logits = model(input_ids)["logits"][0, -1, :].cpu()  # type: ignore
        sorted_indices = logits.argsort(descending=True)
        sorted_indices = sorted_indices[:n]
        best_string_token_predictions = self._to_string_token(sorted_indices, tokenizer)
        best_token_score = logits[sorted_indices]
        return best_string_token_predictions, best_token_score

    def get_token_per_similarity_level(
        self,
        best_string_token_predictions: List[str],
    ) -> Dict[int, List[str]]:
        # # sort the tokens by similarity
        # sorted_tokens = sorted(
        #     similarity_scores.items(), key=lambda x: x[1], reverse=True
        # )
        # sorted_tokens = [token for token, score in sorted_tokens]

        # # get the lenght of the dataset and split it into 4 groups
        # length = len(sorted_tokens)

        # group = {}
        # group[1] = sorted_tokens[: length // 4]
        # group[2] = sorted_tokens[length // 4 : length // 2]
        # group[3] = sorted_tokens[length // 2 : 3 * length // 4]
        # group[4] = sorted_tokens[3 * length // 4 :]

        # return group
        n_tokens = len(best_string_token_predictions)
        group = {}
        group[1] = best_string_token_predictions[: n_tokens // 4]
        group[2] = best_string_token_predictions[n_tokens // 4 : n_tokens // 2]
        group[3] = best_string_token_predictions[n_tokens // 2 : 3 * n_tokens // 4]
        group[4] = best_string_token_predictions[3 * n_tokens // 4 :]
        return group

    def _to_string_token(self, token_id: List[int], tokenizer) -> List[str]:
        list_of_strings = [tokenizer.decode(j) for j in token_id]
        return list_of_strings

    # @abstractmethod
    # def get_best_string_token_predictions(self, prompt: str, n: int) -> List[str]:
    #     pass

    # @abstractmethod
    # def get_token_per_similarity_level(
    #     self, prompt: str, target: str, best_string_token_predictions: List[str]
    # ) -> Dict[int, List[str]]:
    #     pass

    def _clear_cache(self):
        """
        Remove model and tokenizer from memory to free up space
        """
        self.model = None
        # self.tokenizer = None
        torch.cuda.empty_cache()

    def cuda(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cpu":
            logging.warning("Warning: CUDA not available, using CPU")
        self.targets = [t.to(device) for t in self.targets]
        self.tokenized_prompts = [t.to(device) for t in self.tokenized_prompts]
        self.obj_pos = [t.to(device) for t in self.obj_pos]

    def get_len(self):
        return self.len

    def set_len(self, length: int):
        """
        Set the length of the dataset to the given length, and filter the data accordingly
        """
        self.len = length
        self.prompts = []
        self.tokenized_prompts = []
        self.targets = []
        self.obj_pos = []
        self.data = []
        self.subj_pos = []

        self.data = [d for d in self.full_data if d["length"] == length]
        # filter data by length
        self.prompts = [d["prompt"] for d in self.data]
        self.tokenized_prompts = [d["tokenized_prompt"] for d in self.data]
        self.targets = [d["targets"] for d in self.data]
        self.obj_pos = [d["obj_pos"] for d in self.data]
        self.original_index = [
            i for i, d in enumerate(self.full_data) if d["length"] == length
        ]
        if self.experiment == "contextVSfact":
            self.subj_pos = [d["subj_position"] for d in self.data]
        else:
            self.subj_pos = [-100 for d in self.data]
        # if self.similarity[0] is True:
        #     self.apply_similarity()

        assert self.prompts != [], f"Dataset is empty for length {length}"
        assert self.check_duplicate()

    # def apply_similarity(self):
    #     """
    #     Apply the similarity to the dataset
    #     """
    #     # similarity level
    #     similarity_level = self.similarity[1]
    #     similarity_type = self.similarity[2]

    #     for d in tqdm(self.full_data, desc="Applying similarity", total=len(self.full_data)):
    #         similar_token_str, similar_token = self.get_similar_token(
    #             d, similarity_level, similarity_type
    #         )
    #         d["prompt"] = d["prompt"].replace(d["target_new"], similar_token_str)
    #         d["tokenized_prompt"][d["obj_pos"]] = similar_token.item()
    #         d["targets"][1] = similar_token_str
    #         d["target_new"] = similar_token_str
    #         d["target_new_token"] = similar_token
    #         print(d)

    #     # for idx in range(self.__len__()):
    #     #     similar_token_str, similar_token = self.get_similar_token(
    #     #         idx, similarity_level, similarity_type
    #     #     )
    #     #     self.prompts[idx] = self.prompts[idx].replace(self.data[idx]["target_new"], similar_token_str)
    #     #     self.tokenized_prompts[idx][0, self.data[idx]["obj_pos"]] = similar_token[0, 0]
    #     #     self.targets[idx][0, 1] = similar_token[0, 0]
    #     #     self.data[idx]["target_new"] = similar_token_str
    #     #     self.data[idx]["target_new_token"] = similar_token

    def check_duplicate(self):
        seen = set()
        for i, d in enumerate(self.full_data):
            if d["prompt"] in seen:
                for j, d2 in enumerate(self.full_data):
                    if j == i:
                        continue
                    if d["prompt"] == d2["prompt"]:
                        if d["target_new"] == d2["target_new"]:
                            if d["target_true"] == d2["target_true"]:
                                return False
            seen.add(d["prompt"])
        return True

    def get_similar_token(self, data_point: dict, similarity_level: int) -> str:
        # token_to_be_similar_str = data_point["target_true"]
        # token_to_be_similar = data_point["target_true_token"]
        # base_prompt = data_point["base_prompt"]
        # return self._get_similar_token(token_to_be_similar_str, token_to_be_similar, similarity_level, similarity_type, base_prompt)
        num_possible_choices = len(data_point[f"similar_tokens_{similarity_level}"])
        index = random.randint(0, num_possible_choices - 1)
        string_token = data_point[f"similar_tokens_{similarity_level}"][index]
        # print(index)
        return string_token

    @abstractmethod
    def _tokenize_prompt(self, prompt: str, prepend_bos: bool) -> torch.Tensor:
        pass

    @abstractmethod
    def _tokenize_target(self, target: str, prepend_bos: bool) -> torch.Tensor:
        pass

    @abstractmethod
    def _get_similar_token(
        self,
        token_to_be_similar_str: str,
        token_to_be_similar: torch.Tensor,
        similarity_level: int,
        similarity_type: str,
        base_prompt: str,
    ) -> Tuple[str, torch.Tensor]:
        pass

    # @abstractmethod
    # def _to_string_token(self, token_id: List[int]) -> str:
    #     pass

    @abstractmethod
    def _get_similarity_score(self, base_prompt: str, aux_prompt: str) -> torch.Tensor:
        pass


class TlensDataset(BaseDataset):
    def __init__(
        self,
        path: str,
        experiment: Literal["copyVSfact", "contextVSfact"],
        model: Union[WrapHookedTransformer, str, HookedTransformer],
        slice: Optional[int] = None,
        start: Optional[int] = None,
        premise: str = "Redefine:",
        similarity: Tuple[
            bool, int, Literal["word2vec", "logit", "self-similarity"]
        ] = (
            False,
            0,
            "logit",
        ),
    ):
        if isinstance(model, str):
            self.model = WrapHookedTransformer.from_pretrained(model, device="cuda")
        else:
            self.model = model
        self.model.eval()
        super().__init__(path, slice, start, experiment, premise, similarity)

    def _tokenize_prompt(self, prompt: str, prepend_bos: bool) -> torch.Tensor:
        tokens = self.model.to_tokens(prompt, prepend_bos).squeeze(0)
        assert len(tokens.shape) == 1
        return tokens

    def _tokenize_target(self, target: str, prepend_bos: bool) -> Tensor:
        if self.model.predict_with_space is False:
            # remove the first space
            target = target[1:]
        tokens = torch.tensor([self.model.to_tokens(target, prepend_bos).squeeze(0)[0]])
        # print(self.model.to_str_tokens(target,prepend_bos))
        assert (
            tokens.shape[0] == 1
        ), "tokens is not a 1D tensor with one element (the target)"
        return tokens

    def _get_similar_token(
        self,
        token_to_be_similar_str: str,
        token_to_be_similar: torch.Tensor,
        similarity_level: int,
        similarity_type: str,
        base_prompt: str,
    ) -> Tuple[str, torch.Tensor]:
        if similarity_type == "input":
            with torch.no_grad():
                token_embedding = self.model.W_E[token_to_be_similar].squeeze(0)
                embeddings = self.model.W_E

            cosine_similarity = torch.nn.functional.cosine_similarity(
                embeddings, token_embedding, dim=-1
            )
            cosine_similarity, sorted_indices = cosine_similarity.sort(descending=True)

            sorted_indices = sorted_indices[1:]
            cosine_similarity = cosine_similarity[1:]

            group = {}
            group[1] = sorted_indices[
                cosine_similarity < torch.quantile(cosine_similarity, 0.25)
            ]
            group[2] = sorted_indices[
                (cosine_similarity >= torch.quantile(cosine_similarity, 0.25))
                & (cosine_similarity < torch.quantile(cosine_similarity, 0.5))
            ]
            group[3] = sorted_indices[
                (cosine_similarity >= torch.quantile(cosine_similarity, 0.5))
                & (cosine_similarity < torch.quantile(cosine_similarity, 0.75))
            ]
            group[4] = sorted_indices[
                cosine_similarity >= torch.quantile(cosine_similarity, 0.75)
            ]

            sampled_token_idx = torch.randint(
                0, len(group[similarity_level]), (1,)
            ).item()
            sampled_token = group[similarity_level][sampled_token_idx]
            sampled_token_str = self.model.to_string(sampled_token.item())
            assert (
                sampled_token_str != token_to_be_similar_str
            ), "sampled_token_str is the same as token_to_be_similar_str"
            assert sampled_token.shape[0] == 1, "sampled_token is not a 1D tensor"
            assert len(sampled_token.shape) == 2, "sampled_token is not a 2D tensor"
            assert isinstance(
                sampled_token_str, str
            ), "sampled_token_str is not a string"
            return sampled_token_str, sampled_token.unsqueeze(0).unsqueeze(0)
        elif similarity_type == "output":
            raise NotImplementedError
        else:
            raise ValueError("similarity_type must be either 'input' or 'output'")

    @abstractmethod
    def _get_vocab_size(self) -> int:
        return self.model.vocab_size

    @abstractmethod
    def _to_string_token(self, token_id: int) -> str:
        return self.model.to_string(torch.tensor(token_id))

    def get_best_string_token_predictions(self, prompt: str, n: int) -> List[str]:
        raise NotImplementedError

    def get_token_per_similarity_level(
        self, prompt: str, best_string_token_predictions: List[str]
    ) -> Dict[int, List[str]]:
        raise NotImplementedError

    def compute_similarity_word2vec(
        self, base_target: str, word2vec
    ) -> List[Tuple[str, float]]:
        raise NotImplementedError


class HFDataset(BaseDataset):
    def __init__(
        self,
        model: Union[AutoModelForCausalLM, str],
        tokenizer,
        path: str,
        experiment: Literal["copyVSfact", "contextVSfact"],
        slice: Optional[int] = None,
        premise: str = "Redefine:",
        similarity: Tuple[
            bool, int, Literal["word2vec", "logit", "self-similarity"]
        ] = (
            False,
            0,
            "logit",
        ),
        family_name: str = "gpt2",
    ):
        if isinstance(model, str):
            self.model = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype=torch.bfloat16,
            )
            self.model = self.model.cuda()

        else:
            self.model = model
        self.tokenizer = tokenizer
        super().__init__(
            path=path,
            slice=slice,
            experiment=experiment,
            premise=premise,
            similarity=similarity,
            family_name=family_name,
        )

    def reset(
        self,
        similarity: Tuple[
            bool, int, Literal["word2vec", "logit", "self-similarity"]
        ] = None,
    ):
        super().reset(similarity=similarity)

    def _tokenize_prompt(self, prompt: str, prepend_bos: bool) -> torch.Tensor:
        tokens = self.tokenizer.encode(
            prompt, return_tensors="pt", add_special_tokens=True
        ).squeeze(0)
        assert len(tokens.shape) == 1
        return tokens

    def _tokenize_target(self, target: str, prepend_bos: bool) -> torch.Tensor:
        tokens = self.tokenizer.encode(
            target, return_tensors="pt", add_special_tokens=False
        ).squeeze(0)
        assert len(tokens.shape) == 1
        if tokens.shape[0] > 1:
            tokens = tokens[0]
            tokens = tokens.unsqueeze(0)
        return tokens

    def compute_similarity_word2vec(
        self, base_target: str, word2vec, other_target: str
    ) -> List[Tuple[str, float]]:
        similarity = []
        # remove the first space
        base_target = base_target[1:]

        if self.similarity_score_dict.get(base_target) is not None:
            return self.similarity_score_dict[base_target]

        # print(self.tokenizer.encode(" C"))
        for token in range(self.tokenizer.vocab_size):
            str_token = self.tokenizer.decode(token)
            if str_token[0] == " ":
                space_token = str_token
                str_token = str_token[1:]
            else:
                space_token = " " + str_token
            # if str_token == other_target:
            #     print("found")
            try:
                similarity.append(
                    (space_token, word2vec.similarity(base_target, str_token))
                )
            except KeyError:
                if " " + str_token == other_target:
                    print("other_target", other_target, " is not in the w2v vocab")

        # save the similarity score
        self.similarity_score_dict[base_target] = similarity
        # similarity_len = len(similarity)
        # print(similarity_len)
        return similarity


#     def compute_similarity_word2vec(self, base_target, word2vec, other_target):
#         base_target = base_target[1:]

#         with Pool(processes=4) as pool:  # Adjust the number of processes as needed
#             results = pool.starmap(
#                 compute_similarity_for_token,
#                 [(base_target, word2vec, token, self.tokenizer) for token in range(self.tokenizer.vocab_size)]
#             )

#         # Filter out None results and return
#         return [result for result in results if result is not None]


# def compute_similarity_for_token(base_target, word2vec, token, tokenizer):
#     str_token = tokenizer.decode(token)
#     if str_token[0] == " ":
#         space_token = str_token
#         str_token = str_token[1:]
#     else:
#         space_token = " " + str_token

#     try:
#         return space_token, word2vec.similarity(base_target, str_token)
#     except KeyError:
#         return None


class SampleDataset:
    def __init__(self, path: str, model, save_path: str, tokenizer: Optional[object]):
        self.data = json.load(open(path))
        self.model = model
        self.save_path = save_path
        self.checkpoint_size = 500
        if type(model) == WrapHookedTransformer:
            self.model_type = "WrapHookedTransformer"
            self.tokenizer = model.tokenizer
        else:
            self.model_type = "AutoModelForCausalLM"
            try:
                self.tokenizer = tokenizer
            except AttributeError:
                raise ValueError("With HuggingFace models, you must pass a tokenizer")

    def sample(self, size: int = 10000):
        if type(self.model) == WrapHookedTransformer:
            self.sample_dataset_tlens(size)
        else:
            self.sample_dataset_hf(size)

    def sample_dataset_tlens(self, size: int):
        # random.seed(42)
        new_data, index = self.load_from_checkpoint()

        random.shuffle(self.data)
        self.data = self.data[index:]
        size = size - len(new_data)
        new_data = []
        random.shuffle(self.data)
        with tqdm(total=size) as pbar:
            for i, d in enumerate(self.data):
                if i % self.checkpoint_size == 0:
                    self.checkpoint(i, new_data)
                # empty_prompt = d["template"].format("Redefine", self.model.tokenizer.pad_token)
                empty_prompt = d["base_prompt"]
                if self.model.predict(empty_prompt)[1][0] == d["target_true"]:
                    new_data.append(d)
                    if len(new_data) == size:
                        break
                pbar.update(len(new_data) - pbar.n)
            self.data = new_data

    def sample_dataset_hf(self, size: int):
        # random.seed(42)
        new_data, index = self.load_from_checkpoint()

        random.shuffle(self.data)
        self.data = self.data[index:]
        print(len(self.data), index)

        with tqdm(total=size) as pbar:
            pbar.update(len(new_data))
            for i, d in enumerate(self.data):
                if i % self.checkpoint_size == 0:
                    self.checkpoint(i, new_data)
                empty_prompt = d["base_prompt"]
                # encode the prompt
                input_ids = self.tokenizer.encode(empty_prompt, return_tensors="pt")  # type: ignore
                input_ids = input_ids.to(self.model.device)  # type: ignore
                target_true = self.tokenizer.encode(
                    d["target_true"], return_tensors="pt", add_special_tokens=False
                )  # type: ignore
                # predict the next token
                logits = self.model(input_ids)["logits"][0, -1, :].cpu()
                # get the index of the predicted token
                index = logits.argmax()
                # check if the predicted token is the target

                if index in target_true:
                    new_data.append(d)
                    if len(new_data) == size:
                        break
                pbar.update(len(new_data) - pbar.n)
            self.data = new_data

    def save(self):
        json.dump(self.data, open(self.save_path, "w"), indent=2)

    def checkpoint(self, size: int, data: list):
        # create checkpoint folder

        if not os.path.isdir("../data/checkpoints"):
            os.mkdir("../data/checkpoints")
        # save the data
        # split save path
        save_path = self.save_path.split("/")[2]
        json.dump(data, open(f"../data/checkpoints/{save_path}", "w"), indent=2)

    def load_from_checkpoint(self):
        save_path = self.save_path.split("/")[2]
        # check if the checkpoint exists
        if not os.path.isfile(f"../data/checkpoints/{save_path}"):
            return [], 0
        print("Starting from checkpoint")
        data = json.load(open(f"../data/checkpoints/{save_path}"))
        # get index of the last data point
        index = len(data) - 1
        if index < 0:
            return [], 0
        return data, index


class DatasetGenerator:
    def __init__(self, path):
        self.data = json.load(open(path))

    def generate_dataset(self, model, lenghts=[17, 19, 23]):
        my_data = []
        for i, d in tqdm(
            enumerate(self.data), total=len(self.data), desc="Generating dataset"
        ):
            target_new = " " + d["requested_rewrite"]["target_true"]["str"]
            target_true = " " + d["requested_rewrite"]["target_new"]["str"]
            if i % 50 == 0:
                unique_strs = set(json.dumps(d) for d in my_data)
                my_data = [json.loads(s) for s in unique_strs]
                print(len(my_data))
                # if len(my_data) > 1000:
                #     break
            for p in d["attribute_prompts"]:
                template = "Redefine: " + p + "{}" + ". " + p
                # find position of {} in template
                if (
                    len(model.to_str_tokens(template.format(model.tokenizer.pad_token)))
                    not in lenghts
                ):
                    continue
                try:
                    obj_pos = (
                        model.to_str_tokens(
                            template.format(model.tokenizer.pad_token)
                        ).index(".")
                        - 1
                    )
                except:  # noqa: E722
                    continue
                if target_true in template:
                    continue
                prediction = model.predict(template.format(model.tokenizer.pad_token))[
                    1
                ][0]
                copy_prediction = model.predict(template.format(target_new))[1][0]
                if prediction == target_true and copy_prediction == target_new:
                    my_data.append(
                        {
                            "prompt": p,
                            "template": template,
                            "prediction": prediction,
                            "copy_prediction": copy_prediction,
                            "target_true": target_true,
                            "target_new": target_new,
                            "length": len(
                                model.to_str_tokens(
                                    template.format(model.tokenizer.pad_token)
                                )
                            ),
                            "lenght_copy": len(
                                model.to_str_tokens(template.format(target_new))
                            ),
                            "obj_pos": obj_pos,
                        }
                    )
            for p in d["neighborhood_prompts"]:
                template = "Redefine: " + p + "{}" + ". " + p
                # find position of {} in template
                if (
                    len(model.to_str_tokens(template.format(model.tokenizer.pad_token)))
                    not in lenghts
                ):
                    continue
                try:
                    obj_pos = (
                        model.to_str_tokens(
                            template.format(model.tokenizer.pad_token)
                        ).index(".")
                        - 1
                    )
                except:  # noqa: E722
                    continue
                if target_true in template:
                    continue
                prediction = model.predict(template.format(model.tokenizer.pad_token))[
                    1
                ][0]
                copy_prediction = model.predict(template.format(target_new))[1][0]
                if prediction == target_true and copy_prediction == target_new:
                    # check if is a duplicate

                    my_data.append(
                        {
                            "prompt": p,
                            "template": template,
                            "prediction": prediction,
                            "copy_prediction": copy_prediction,
                            "target_true": target_new,
                            "target_new": target_true,
                            "length": len(
                                model.to_str_tokens(
                                    template.format(model.tokenizer.pad_token)
                                )
                            ),
                            "lenght_copy": len(
                                model.to_str_tokens(template.format(target_new))
                            ),
                            "obj_pos": obj_pos,
                        }
                    )

        print(
            "Number of examples:", len(my_data), "Number of possible lengths:", lenghts
        )
        self.my_data = my_data

    def save(self, path):
        json.dump(self.my_data, open(path, "w"), indent=2)
