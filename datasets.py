import os
from torch.utils.data import Dataset
import torch
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, directory: str, mode: str='WT') -> None:
        super(CustomDataset).__init__()
        names = os.listdir(directory)
        names.sort()
        self.paths = [os.path.join(directory, name) for name in names]
        self.len = len(self.paths)
        self.mode = mode.lower()

    def __getitem__(self, index):
        a = np.load(self.paths[index])
        if self.mode == 'wt':
            return (torch.from_numpy(a['input']), torch.from_numpy(np.stack([a['mask'][0], ~a['mask'][0]])))
        elif self.mode == 'tc':
            return (torch.from_numpy(a['input']), torch.from_numpy(np.stack([a['mask'][1], ~a['mask'][1]])))
        elif self.mode == 'et':
            return (torch.from_numpy(a['input']), torch.from_numpy(np.stack([a['mask'][2], ~a['mask'][2]])))
        elif self.mode == 'threeclass':
            return (torch.from_numpy(a['input']), torch.from_numpy(a['mask']))
        elif self.mode =='fourclass':
            return (torch.from_numpy(a['input']), torch.from_numpy(a['mask']))
    
    def __len__(self):
        return self.len

class DatasetFromLoadedData(Dataset):
    def __init__(self, data: np.ndarray) -> None:
        super(DatasetFromLoadedData).__init__()
        self.len = data.shape[0]
        self.data = data
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return self.len