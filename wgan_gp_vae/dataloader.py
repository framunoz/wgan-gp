from pathlib import Path

import torch
from PIL import Image
from torch.utils import data
from torchvision.datasets.vision import VisionDataset


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
        root=Path("dataset/quick_draw/face.npy"),
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


def get_h5_dataset(path="shoes_images/shoes.hdf5", batch_size=128, shuffle=True):
    return data.DataLoader(
        dataset=H5Loader(path), batch_size=batch_size, shuffle=shuffle
    )
