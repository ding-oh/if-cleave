import torch
from torch_geometric.data import Batch


class CleavageDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def custom_collate(batch):
    return Batch.from_data_list(batch)


def compute_feature_stats(data_list):
    total = total_sq = None
    count = 0
    for item in data_list:
        x = item.x.float()
        if total is None:
            total = torch.zeros(x.shape[1], dtype=x.dtype)
            total_sq = torch.zeros(x.shape[1], dtype=x.dtype)
        total += x.sum(dim=0)
        total_sq += (x * x).sum(dim=0)
        count += x.shape[0]
    mean = total / max(count, 1)
    var = total_sq / max(count, 1) - mean * mean
    std = torch.sqrt(torch.clamp(var, min=1e-12))
    return mean, std


def apply_standardization(data_list, mean, std):
    for item in data_list:
        item.x = (item.x.float() - mean) / std
