from src.dataset import TlensDataset
from src.model import WrapHookedTransformer
import torch
from torch.utils.data import DataLoader
from typing import Optional,  Literal

torch.set_grad_enabled(False)


def to_logit_token(
    logit, target, normalize="logsoftmax", return_index=False
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    assert (
        len(logit.shape) in [2, 3]
    ), "logit should be of shape (batch_size, d_vocab) or (batch_size, seq_len, d_vocab)"
    if len(logit.shape) == 3:
        logit = logit[:, -1, :]  # batch_size, d_vocab
    if normalize == "logsoftmax":
        logit = torch.log_softmax(logit, dim=-1)
    elif normalize == "softmax":
        logit = torch.softmax(logit, dim=-1)
    elif normalize == "none":
        pass
    logit_mem = torch.zeros(target.shape[0])
    logit_cp = torch.zeros(target.shape[0])
    index_mem = torch.zeros(target.shape[0])
    index_cp = torch.zeros(target.shape[0])

    if return_index:
        for i in range(target.shape[0]):
            sorted_indices = torch.argsort(logit[i], descending=True).tolist()
            index_mem[i] = sorted_indices.index(target[i, 0])
            index_cp[i] = sorted_indices.index(target[i, 1])

    for i in range(target.shape[0]):
        logit_mem[i] = logit[i, target[i, 0]]
        # save the position of target[i, 0] in the logit sorted
        # index_mem[i] = torch.argsort(logit[i], descending=True).tolist().index(target[i, 0])
        logit_cp[i] = logit[i, target[i, 1]]
        # index_cp[i] = torch.argsort(logit[i], descending=True).tolist().index(target[i, 1])

    if return_index:
        return logit_mem, logit_cp, index_mem, index_cp
    return logit_mem, logit_cp, None, None


class BaseExperiment:
    def __init__(
        self,
        dataset: TlensDataset,
        model: WrapHookedTransformer,
        batch_size: int,
        experiment: Literal["copyVSfact", "contextVSfact"] = "copyVSfact",
        filter_outliers: bool = False,
    ):
        self.dataset = dataset
        self.model = model
        self.model.eval()
        self.batch_size = batch_size
        self.filter_outliers = filter_outliers
        self.experiment: Literal["copyVSfact", "contextVSfact"] = experiment
        #requires grad to false
        torch.set_grad_enabled(False)
        # if self.model.cfg.model_name != self.dataset.model.cfg.model_name:
        #     raise ValueError("Model and dataset should have the same model_name, found {} and {}".format(
        #         self.model.cfg.model_name, self.dataset.model.cfg.model_name))

    def set_len(self, length: int, slice_to_fit_batch: bool = True) -> None:
        self.dataset.set_len(length)

    def get_basic_logit(
        self, normalize_logit: Literal["none", "softmax", "log_softmax"] = "none"
    ) -> tuple[torch.Tensor, torch.Tensor]: # type: ignore
        lengths = self.dataset.get_lengths()
        for length in lengths:
            self.set_len(length, slice_to_fit_batch=False)
            dataloader = DataLoader(
                self.dataset, batch_size=self.batch_size, shuffle=False
            )
            num_batches = len(dataloader)
            logit_mem_list, logit_cp_list = [], []
            if num_batches == 0:
                continue
            for batch in dataloader:
                logit, _ = self.model.run_with_cache(batch["prompt"])
                logit = logit[:, -1, :]  # type: ignore
                logit_mem, logit_cp, _, _ = to_logit_token(logit, batch["target"], normalize=normalize_logit)
                logit_mem_list.append(logit_mem)
                logit_cp_list.append(logit_cp)

            logit_mem = torch.cat(logit_mem_list, dim=0)
            logit_cp = torch.cat(logit_cp_list, dim=0)
            return logit_mem, logit_cp

    def compute_logit(self) -> tuple[torch.Tensor, torch.Tensor]:
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

        corrupted_logit = []
        target = []
        for batch in dataloader:
            corrupted_logit.append(self.model(batch["prompt"])[:, -1, :].cpu())
            target.append(batch["target"].cpu())
        # clean_logit = torch.cat(clean_logit, dim=0)
        corrupted_logit = torch.cat(corrupted_logit, dim=0)
        target = torch.cat(target, dim=0)
        return corrupted_logit, target

    def get_batch(self, len: int, **kwargs):
        self.set_len(len, **kwargs)
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        return next(iter(dataloader))
    
    def get_position(self, resid_pos: str):
        if self.experiment == "copyVSfact":
            return self.__get_position_copyVSfact__(resid_pos)
        elif self.experiment == "contextVSfact":
            return 0
            raise NotImplementedError

    def __get_position_copyVSfact__(self, resid_pos: str) -> int:
        positions = {
            "1_1_subject": 5,
            "1_2_subject": 6,
            "1_3_subject": 7,
            "definition": self.dataset.obj_pos[0],
            "2_1_subject": self.dataset.obj_pos[0] + 2,
            "2_2_subject": self.dataset.obj_pos[0] + 3,
            "2_3_subject": self.dataset.obj_pos[0] + 4,
            "last_pre": -2,
            "last": -1,
        }
        return positions.get(
            resid_pos,
            ValueError(
                "resid_pos not recognized: should be one of 1_1_subject, 1_2_subject, 1_3_subject, definition, 2_1_subject, 2_2_subject, 2_3_subject"
            ),
        )

    def aggregate_result(self, **args):
        if self.experiment == "copyVSfact":
            return self.__aggregate_result_copyVSfact__(**args)
        elif self.experiment == "contextVSfact":
            return self.__aggregate_result_contextVSfact__(**args)
        
    def __aggregate_result_contextVSfact__(self, object_positions: int, pattern: torch.Tensor, length: int, dim: int = -1) -> torch.Tensor:
        raise NotImplementedError

    def __aggregate_result_copyVSfact__(
        self, object_positions: int, pattern: torch.Tensor, length: int, dim: int = -1
    ) -> torch.Tensor:
        subject_1_1 = 5
        subject_1_2 = 6 if length > 15 else 5
        subject_1_3 = 7 if length > 17 else subject_1_2
        subject_2_1 = object_positions + 2
        subject_2_2 = object_positions + 3 if length > 15 else subject_2_1
        subject_2_3 = object_positions + 4 if length > 17 else subject_2_2
        subject_2_2 = subject_2_2 if subject_2_2 < length else subject_2_1
        subject_2_3 = subject_2_3 if subject_2_3 < length else subject_2_2
        last_position = length - 1
        object_positions_pre = object_positions - 1
        object_positions_next = object_positions + 1
        *leading_dims, pen_len, last_len = pattern.shape

        result_aggregate = torch.zeros((*leading_dims, 13, 13))
        intermediate_aggregate = torch.zeros((*leading_dims, pen_len, 13))
        # aggregate for pre-last dimension
        intermediate_aggregate[..., 0] = pattern[..., :subject_1_1].mean(dim=-1)
        intermediate_aggregate[..., 1] = pattern[..., subject_1_1]
        intermediate_aggregate[..., 2] = pattern[..., subject_1_2]
        intermediate_aggregate[..., 3] = pattern[..., subject_1_3]
        intermediate_aggregate[..., 4] = pattern[
            ..., subject_1_3 + 1 : object_positions_pre
        ].mean(dim=-1)
        intermediate_aggregate[..., 5] = pattern[..., object_positions_pre]
        intermediate_aggregate[..., 6] = pattern[..., object_positions]
        intermediate_aggregate[..., 7] = pattern[..., object_positions_next]
        intermediate_aggregate[..., 8] = pattern[..., subject_2_1]
        intermediate_aggregate[..., 9] = pattern[..., subject_2_2]
        intermediate_aggregate[..., 10] = pattern[..., subject_2_3]
        intermediate_aggregate[..., 11] = pattern[
            ..., subject_2_3 + 1 : last_position
        ].mean(dim=-1)
        intermediate_aggregate[..., 12] = pattern[..., last_position]
        if dim == -1:
            return intermediate_aggregate
        # aggregate for last dimension
        result_aggregate[..., 0, :] = intermediate_aggregate[..., :subject_1_1, :].mean(
            dim=-2
        )
        result_aggregate[..., 1, :] = intermediate_aggregate[..., subject_1_1, :]
        result_aggregate[..., 2, :] = intermediate_aggregate[..., subject_1_2, :]
        result_aggregate[..., 3, :] = intermediate_aggregate[..., subject_1_3, :]
        result_aggregate[..., 4, :] = intermediate_aggregate[
            ..., subject_1_3 + 1 : object_positions_pre, :
        ].mean(dim=-2)
        result_aggregate[..., 5, :] = intermediate_aggregate[
            ..., object_positions_pre, :
        ]
        result_aggregate[..., 6, :] = intermediate_aggregate[..., object_positions, :]
        result_aggregate[..., 7, :] = intermediate_aggregate[
            ..., object_positions_next, :
        ]
        result_aggregate[..., 8, :] = intermediate_aggregate[..., subject_2_1, :]
        result_aggregate[..., 9, :] = intermediate_aggregate[..., subject_2_2, :]
        result_aggregate[..., 10, :] = intermediate_aggregate[..., subject_2_3, :]
        result_aggregate[..., 11, :] = intermediate_aggregate[
            ..., subject_2_3 + 1 : last_position, :
        ].mean(dim=-2)
        result_aggregate[..., 12, :] = intermediate_aggregate[..., last_position, :]
        return result_aggregate

    @staticmethod
    def to_logit_token(
        logit: torch.Tensor, target: torch.Tensor, normalize: str = "logsoftmax"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert (
            len(logit.shape) in [2, 3]
        ), "logit should be of shape (batch_size, d_vocab) or (batch_size, seq_len, d_vocab)"
        assert normalize in [
            "logsoftmax",
            "softmax",
            "none",
        ], "normalize should be one of logsoftmax, softmax, none"
        if len(logit.shape) == 3:
            logit = logit[:, -1, :]  # batch_size, d_vocab
        if normalize == "logsoftmax":
            logit = torch.log_softmax(logit, dim=-1)
        elif normalize == "softmax":
            logit = torch.softmax(logit, dim=-1)
        elif normalize == "none":
            pass
        logit_mem = torch.zeros(target.shape[0])
        logit_cp = torch.zeros(target.shape[0])
        for i in range(target.shape[0]):
            logit_mem[i] = logit[i, target[i, 0]]
            logit_cp[i] = logit[i, target[i, 1]]
        return logit_mem, logit_cp
