# _*_coding:utf-8_*_
# @auther:FelixFu
# @Date: 2021.4.14
# @github:https://github.com/felixfu520

import numpy as np

import torch
from torch.utils.data import DataLoader


class BaseDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.shuffle = shuffle
        self.dataset = dataset
        self.nbr_examples = len(dataset)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers,
            'pin_memory': True
        }
        super(BaseDataLoader, self).__init__(**self.init_kwargs)


class DataPrefetcher(object):
    def __init__(self, loader, device, stop_after=None):
        """
        self.train_loader = DataPrefetcher(train_loader, device=self.device)
        """
        self.loader = loader
        self.dataset = loader.dataset
        self.stream = torch.cuda.Stream()
        self.stop_after = stop_after
        self.next_input = None
        self.next_target = None
        self.next_image_path = None
        self.device = device

    def __len__(self):
        return len(self.loader)

    def preload(self):
        try:
            self.next_input, self.next_target, self.next_image_path = next(self.loaditer)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.next_image_path = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(device=self.device, non_blocking=True)
            self.next_target = self.next_target.cuda(device=self.device, non_blocking=True)
            self.next_image_path = self.next_image_path

    def __iter__(self):
        count = 0
        self.loaditer = iter(self.loader)
        self.preload()
        while self.next_input is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            input = self.next_input
            target = self.next_target
            image_path = self.next_image_path
            self.preload()
            count += 1
            yield input, target, image_path
            if type(self.stop_after) is int and (count > self.stop_after):
                break