import random
from Src.dataset import BaseDataset
from Src.model import BaseModel
import torch
from torch.utils.data import DataLoader
from typing import Optional, Literal
# from line_profiler import profile


torch.set_grad_enabled(False)


# @profile
def to_logit_token(
    logit,
    target,
    normalize="none",
    return_index=False,
    return_winners=False,
    return_rank=False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[int],
    Optional[int],
]:
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
    index_mem = torch.zeros(target.shape[0], dtype=torch.float32)
    index_cp = torch.zeros(target.shape[0], dtype=torch.float32)

    if return_index:
        sorted_indices = torch.argsort(logit, descending=True)
    # batch_indices = torch.arange(target.shape[0])
    # logit_mem = logit[batch_indices, target[:, 0]]
    # logit_cp = logit[batch_indices, target[:, 1]]
    logit_argmaxs = torch.argmax(logit, dim=-1)
    mem_winners = torch.zeros(target.shape[0], dtype=torch.float32)
    cp_winners = torch.zeros(target.shape[0], dtype=torch.float32)

    for i in range(target.shape[0]):
        logit_mem[i] = logit[i, target[i, 0]]
        # save the position of target[i, 0] in the logit sorted
        # index_mem[i] = torch.argsort(logit[i], descending=True).tolist().index(target[i, 0])
        logit_cp[i] = logit[i, target[i, 1]]
        if return_winners:
            if logit_argmaxs[i] == target[i, 0]:
                mem_winners[i] = 1

            if logit_argmaxs[i] == target[i, 1]:
                cp_winners[i] = 1
        if return_index:
            target_expanded_0 = target[:, 0].unsqueeze(1) == sorted_indices
            target_expanded_1 = target[:, 1].unsqueeze(1) == sorted_indices
            index_mem = target_expanded_0.nonzero()[
                :, 1
            ]  # Select column index for matches of target[:, 0]
            index_cp = target_expanded_1.nonzero()[
                :, 1
            ]  # Select column index for matches of target[:, 1]
        # index_cp[i] = torch.argsort(logit[i], descending=True).tolist().index(target[i, 1])

    if return_winners:
        if return_index:
            return logit_mem, logit_cp, index_mem, index_cp, mem_winners, cp_winners
        return logit_mem, logit_cp, None, None, mem_winners, cp_winners
    if return_index:
        return logit_mem, logit_cp, index_mem, index_cp
    return logit_mem, logit_cp, None, None


class BaseExperiment:
    def __init__(
        self,
        dataset: BaseDataset,
        model: BaseModel,
        batch_size: int,
        experiment: Literal["copyVSfact", "contextVSfact"] = "copyVSfact",
    ):
        self.dataset = dataset
        self.model = model
        self.model.eval()
        self.batch_size = batch_size
        self.experiment: Literal["copyVSfact", "contextVSfact"] = experiment
        # requires grad to false
        torch.set_grad_enabled(False)
        # if self.model.cfg.model_name != self.dataset.model.cfg.model_name:
        #     raise ValueError("Model and dataset should have the same model_name, found {} and {}".format(
        #         self.model.cfg.model_name, self.dataset.model.cfg.model_name))

    def set_len(self, length: int, slice_to_fit_batch: bool = True) -> None:
        self.dataset.set_len(length)

    def get_basic_logit(
        self, normalize_logit: Literal["none", "softmax", "log_softmax"] = "none"
    ) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore
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
                logit, _ = self.model.run_with_cache(batch["prompt"], prepend_bos=False)
                logit = logit[:, -1, :]  # type: ignore
                logit_mem, logit_cp, _, _ = to_logit_token(
                    logit, batch["target"], normalize=normalize_logit
                )
                logit_mem_list.append(logit_mem)
                logit_cp_list.append(logit_cp)

            logit_mem = torch.cat(logit_mem_list, dim=0)
            logit_cp = torch.cat(logit_cp_list, dim=0)
            return logit_mem, logit_cp

    def get_batch(self, len: Optional[int] = None, **kwargs):
        if len is None:
            # take a random length
            len = random.choice(self.dataset.get_lengths())
        self.set_len(len, **kwargs)
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        return next(iter(dataloader))
