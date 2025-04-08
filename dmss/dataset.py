from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class PolypDataset(Dataset):
    def __init__(
        self,
    ):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


def get_data_loaders(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """Returns a dataloader for the given dataset."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def tran_val_loaders():
    pass