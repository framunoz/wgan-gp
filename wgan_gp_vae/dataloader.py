import os
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader, random_split
from torchvision.datasets.vision import VisionDataset

PATH_DATASETS = Path("dataset")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64
NUM_WORKERS = int(os.cpu_count() / 2)


class H5Loader(data.Dataset):
    def __init__(self, root, transform=None):
        super().__init__(root=root, transform=transform)
        import h5py

        self.h5_file = h5py.File(self.root, "r")
        self.data = self.h5_file.get("data")

    def __getitem__(self, index):
        sample = self.data[index]

        if self.transform:
            sample = self.transform(sample)

        return sample, 0

    def __len__(self):
        return len(self.data)

    def __del__(self):
        self.h5_file.close()


class NpyDataset(VisionDataset):
    def __init__(
        self,
        root=PATH_DATASETS / "quick_draw/face.npy",
        transform=None,
        shape=(28, 28),
    ) -> None:
        super().__init__(root=root, transform=transform)
        import numpy as np

        X = np.load(root)
        self.data = X.reshape(-1, *shape)

    def __getitem__(self, index):
        img = Image.fromarray(self.data[index], mode="L")

        if self.transform:
            img = self.transform(img)

        return img, 0

    def __len__(self):
        return len(self.data)


class NpyDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir=PATH_DATASETS / "quick_draw/face.npy",
        transform=None,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    ) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

    def setup(self, stage: str):
        train, val, test = random_split(
            NpyDataset(self.data_dir, self.transform), [0.8, 0.1, 0.1]
        )
        if stage == "fit" or stage is None:
            self.ds_train, self.ds_val = train, val

        if stage == "test" or stage is None:
            self.ds_test = test

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


def get_h5_dataset(path="shoes_images/shoes.hdf5", batch_size=128, shuffle=True):
    return data.DataLoader(
        dataset=H5Loader(path), batch_size=batch_size, shuffle=shuffle
    )
