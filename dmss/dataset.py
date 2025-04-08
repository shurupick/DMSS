from typing import Tuple

from torch.utils.data import DataLoader, Dataset


class PolypDataset(Dataset):
    def __init__(
        self,
        image_dir: str = None,
        mask_dir: str = None,
        transform=None,
        mode: str = "train",  # 'train', 'valid', 'test'
    ):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


def get_data_loaders(
    data_dir=None,
    transform=None,  # transforms.Compose([transforms.ToTensor()]) or None
    batch_size: int = 16,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    Data = PolypDataset()

    """Returns a dataloader for the given dataset."""
    train_dataset = PolypDataset
    valid_dataset = PolypDataset
    test_dataset = PolypDataset

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, valid_loader, test_loader
