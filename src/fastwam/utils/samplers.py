from math import ceil
from typing import Iterator, Sized

import torch
from torch.utils.data import Sampler


class ResumableEpochSampler(Sampler[int]):
    def __init__(self, dataset: Sized, seed: int, batch_size: int, num_processes: int):
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.epoch = 0
        self.epoch_offset = 0
        self.resume_batch_offset = 0

        if self.batch_size < 1 or self.num_processes < 1:
            raise ValueError("batch_size and num_processes must be positive")

    @property
    def padded_length(self) -> int:
        """Length after deterministic global-batch padding.

        Padding before applying a resume offset is important. Otherwise
        Accelerate pads the shortened remainder and repeats a different sample
        than an uninterrupted epoch would have used on the final rank.
        """
        size = len(self.dataset)
        if size == 0:
            return 0
        global_batch = self.batch_size * self.num_processes
        return ceil(size / global_batch) * global_batch

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_epoch_offset(self, epoch_offset: int):
        self.epoch_offset = int(epoch_offset)

    def set_resume_batch_offset(self, batch_in_epoch: int):
        self.resume_batch_offset = int(batch_in_epoch)

    def clear_resume_batch_offset(self):
        self.resume_batch_offset = 0

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + self.epoch + self.epoch_offset)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()
        if indices and len(indices) < self.padded_length:
            padding = self.padded_length - len(indices)
            repeats = (padding + len(indices) - 1) // len(indices)
            indices.extend((indices * repeats)[:padding])
        if self.epoch == 0 and self.resume_batch_offset > 0:
            sample_offset = self.resume_batch_offset * self.batch_size * self.num_processes
            indices = indices[sample_offset:]
        return iter(indices)

    def __len__(self) -> int:
        length = self.padded_length
        if self.epoch == 0 and self.resume_batch_offset > 0:
            sample_offset = self.resume_batch_offset * self.batch_size * self.num_processes
            length = max(length - sample_offset, 0)
        return length
