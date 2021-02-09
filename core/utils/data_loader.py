import os
import pickle

import torch
from torch.utils.data import Dataset, DataLoader

from core.utils.torch_utils import tensor


class ToTensor(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, sample):
        return {'state': tensor(sample['state'], self.device),
                'action': tensor(sample['action'], self.device).long(),
                'reward': tensor(sample['reward'], self.device).double(),
                'next_state': tensor(sample['next_state'], self.device),
                'done': tensor(sample['done'].item(), self.device).byte()}


class GridTransitions(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.files = list(map(lambda fname: os.path.join(data_dir, fname),
                              os.listdir(self.data_dir)))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        # print(file_path)
        with open(file_path, "rb") as f:
            instance = pickle.load(f)
            sample = dict()
            sample['state'], sample['action'], sample['reward'], sample['next_state'], sample['done'] = instance
            if self.transform:
                sample = self.transform(sample)
            return sample


# if __name__ == '__main__':
#
#     project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
#     g = GridTransitions(os.path.join(project_root, 'data',
#                                                   'output', 'test', 'transitions'),
#                            transform=ToTensor())
#
#     data_loader = DataLoader(g, batch_size=4, shuffle=True, num_workers=1)
#     print("Dataset size: {}".format(len(g)))
#     for i_batch, b in enumerate(data_loader):
#         print(i_batch, b['state'], b['action'], b['next_state'])
#         break
