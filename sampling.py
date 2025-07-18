import random
import itertools
import torch


class RandResBatchSampler(torch.utils.data.sampler.BatchSampler):
    def __init__(self, sampler, num_batches, drop_last, resolutions=None, res_shuffle=True):
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(num_batches, int) or isinstance(num_batches, bool) or \
                num_batches <= 0:
            raise ValueError(f"batch_size should be a positive integer value, but got batch_size={num_batches}")
        if not isinstance(drop_last, bool):
            raise ValueError(f"drop_last should be a boolean value, but got drop_last={drop_last}")
        self.sampler = sampler
        self.drop_last = drop_last
        self.num_batches = num_batches
        self.resolutions = (
            [(self.sampler.data_source.crop_size, 100)]
            if resolutions is None# else hasattr(self.sampler.data_source, "crop_size")
            else resolutions
        )
        self.res_idx = self.rand_idx() if res_shuffle else 0
        self.res_shuffle = res_shuffle
        self.batch_num = 0
        self.sampler_iter = iter(self.sampler)
    @property
    def batch_size(self):
        _, batch_len = self.get_res_and_batch_size()
        return batch_len
    def rand_idx(self):
        return random.randint(0, len(self.resolutions) - 1)
    def get_res_and_batch_size(self):
        img_res, batch_size = self.resolutions[self.res_idx]
        return img_res, batch_size
    def next_res_and_batch_size(self):
        img_res, batch_len = self.get_res_and_batch_size()
        self.res_idx = (
            self.rand_idx()
            if self.res_shuffle
            else (self.res_idx + 1) % len(self.resolutions)
        )
        return img_res, batch_len
    def __iter__(self):
        img_res, batch_size = self.next_res_and_batch_size()
        for _ in range(self.num_batches):
            batch = [(img_res, item) for item in itertools.islice(self.sampler_iter, batch_size)]
            if len(batch) < batch_size:
                self.sampler_iter = iter(self.sampler)
                batch = [(img_res, item) for item in itertools.islice(self.sampler_iter, batch_size)]
            yield batch
            img_res, batch_size = self.next_res_and_batch_size()
        # img_res, batch_size = self.next_res_and_batch_size()
        # batch = [(img_res, item) for item in itertools.islice(sampler_iter, batch_size)]
        # while batch:
        #     yield batch
        #     self.batch_num += 1
        #     if self.batch_num >= self.num_batches:
        #         self.batch_num = 0
        #         return
        #     img_res, batch_size = self.next_res_and_batch_size()
        #     batch = [(img_res, item) for item in itertools.islice(sampler_iter, batch_size)]
    def __len__(self):
        return self.num_batches