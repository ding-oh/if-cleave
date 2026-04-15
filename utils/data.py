import torch
from torch_geometric.data import Batch


class CleavageDataset(torch.utils.data.Dataset):
    """Dataset wrapper for cleavage data."""

    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def custom_collate(batch):
    """Custom collate function for batching PyG Data objects."""
    return Batch.from_data_list(batch)
