from src.dataset import TlensDataset
from src.model import WrapHookedTransformer
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

torch.set_grad_enabled(False)


def to_logit_token(logit, target, normalize="logsoftmax") -> tuple[torch.Tensor, torch.Tensor]:
    assert len(logit.shape) in [2,
                                3], "logit should be of shape (batch_size, d_vocab) or (batch_size, seq_len, d_vocab)"
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


class BaseExperiment():
    def __init__(self, dataset: TlensDataset, model: WrapHookedTransformer, batch_size:int, filter_outliers:bool=False):
        self.dataset = dataset
        self.model = model
        self.model.eval()
        self.batch_size = batch_size
        self.filter_outliers = filter_outliers
        if self.model.cfg.model_name != self.dataset.model.cfg.model_name:
            raise ValueError("Model and dataset should have the same model_name, found {} and {}".format(
                self.model.cfg.model_name, self.dataset.model.cfg.model_name))

    def set_len(self, length:int, slice_to_fit_batch:bool=True) -> None:
        self.dataset.set_len(length)
        if self.filter_outliers:
            self._filter_outliers()
        elif slice_to_fit_batch:
            print("WARNING: slicing to fit batch")
            self.dataset.slice_to_fit_batch(self.batch_size)
        

    def _filter_outliers(self, save_filtered:bool=False)->None:
        print("save filtered:", save_filtered)
        print("Number of examples before outliers:", len(self.dataset))
        corrupted_logit, target = self.compute_logit()
        # clean_logit_mem, clean_logit_cp = to_logit_token(clean_logit, target)
        corrupted_logit_mem, corrupted_logit_cp = to_logit_token(corrupted_logit, target)

        outliers_under = torch.where(corrupted_logit_mem < (corrupted_logit_mem.mean() - corrupted_logit_mem.std()))[0]
        outliers_over = torch.where(corrupted_logit_cp > (corrupted_logit_cp.mean() + corrupted_logit_cp.std()))[0]
        outliers_indexes = torch.cat([outliers_under, outliers_over], dim=0).tolist()

        maxdatasize = ((len(self.dataset) - len(outliers_indexes)) // self.batch_size) * self.batch_size

        self.dataset.filter_from_idx(outliers_indexes, exclude=True, save_filtered=save_filtered)
        self.dataset.slice(maxdatasize)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        self.num_batches = len(self.dataloader)
        print("Number of examples after outliers:", len(self.dataloader) * self.batch_size)

    def compute_logit(self) -> tuple[ torch.Tensor, torch.Tensor]:
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        corrupted_logit = []
        target = []
        for batch in tqdm(dataloader):
            corrupted_logit.append(self.model(batch["corrupted_prompts"])[:, -1, :].cpu())
            target.append(batch["target"].cpu())
        # clean_logit = torch.cat(clean_logit, dim=0)
        corrupted_logit = torch.cat(corrupted_logit, dim=0)
        target = torch.cat(target, dim=0)
        return  corrupted_logit, target

    def get_batch(self, len:int, **kwargs):
        self.set_len(len, **kwargs)
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        return next(iter(dataloader))

    def get_position(self, resid_pos: str) -> int:
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
        return positions.get(resid_pos, ValueError(
            "resid_pos not recognized: should be one of 1_1_subject, 1_2_subject, 1_3_subject, definition, 2_1_subject, 2_2_subject, 2_3_subject"))

    def aggregate_result(self, object_positions:int, pattern:torch.Tensor, length:int, dim:int=-1) -> torch.Tensor:
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
        
        result_aggregate = torch.zeros((*leading_dims, 13,13))
        intermediate_aggregate = torch.zeros((*leading_dims, pen_len, 13))
        #aggregate for pre-last dimension
        intermediate_aggregate[..., 0] = pattern[..., :subject_1_1].mean(dim=-1)
        intermediate_aggregate[..., 1] = pattern[..., subject_1_1]
        intermediate_aggregate[..., 2] = pattern[..., subject_1_2]
        intermediate_aggregate[..., 3] = pattern[..., subject_1_3]
        intermediate_aggregate[..., 4] = pattern[..., subject_1_3 + 1:object_positions_pre].mean(dim=-1)
        intermediate_aggregate[..., 5] = pattern[..., object_positions_pre]
        intermediate_aggregate[..., 6] = pattern[..., object_positions]
        intermediate_aggregate[..., 7] = pattern[..., object_positions_next]
        intermediate_aggregate[..., 8] = pattern[..., subject_2_1]
        intermediate_aggregate[..., 9] = pattern[..., subject_2_2]
        intermediate_aggregate[..., 10] = pattern[..., subject_2_3]
        intermediate_aggregate[..., 11] = pattern[..., subject_2_3 + 1:last_position].mean(dim=-1)
        intermediate_aggregate[..., 12] = pattern[..., last_position]
        if dim == -1:
            return intermediate_aggregate
        #aggregate for last dimension
        result_aggregate[..., 0,:] = intermediate_aggregate[..., :subject_1_1,:].mean(dim=-2)
        result_aggregate[..., 1,:] = intermediate_aggregate[..., subject_1_1,:]
        result_aggregate[..., 2,:] = intermediate_aggregate[..., subject_1_2,:]
        result_aggregate[..., 3,:] = intermediate_aggregate[..., subject_1_3,:]
        result_aggregate[..., 4,:] = intermediate_aggregate[..., subject_1_3 + 1:object_positions_pre,:].mean(dim=-2)
        result_aggregate[..., 5,:] = intermediate_aggregate[..., object_positions_pre,:]
        result_aggregate[..., 6,:] = intermediate_aggregate[..., object_positions,:]
        result_aggregate[..., 7,:] = intermediate_aggregate[..., object_positions_next,:]
        result_aggregate[..., 8,:] = intermediate_aggregate[..., subject_2_1,:]
        result_aggregate[..., 9,:] = intermediate_aggregate[..., subject_2_2,:]
        result_aggregate[..., 10,:] = intermediate_aggregate[..., subject_2_3,:]
        result_aggregate[..., 11,:] = intermediate_aggregate[..., subject_2_3+ 1:last_position,:].mean(dim=-2)
        result_aggregate[..., 12,:] = intermediate_aggregate[..., last_position,:]
        return result_aggregate
    
    @staticmethod
    def to_logit_token(logit:torch.Tensor, target:torch.Tensor, normalize:str="logsoftmax") -> tuple[torch.Tensor, torch.Tensor]:
        assert len(logit.shape) in [2,
                                    3], "logit should be of shape (batch_size, d_vocab) or (batch_size, seq_len, d_vocab)"
        assert normalize in ["logsoftmax", "softmax", "none"], "normalize should be one of logsoftmax, softmax, none"
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