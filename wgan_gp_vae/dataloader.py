import io
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import requests
import torch
from PIL import Image
from torch.utils import data
from torchvision.datasets.vision import VisionDataset

BINARY_URL = "https://storage.googleapis.com/quickdraw_dataset/full/binary/"
NUMPY_URL = (
    "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
)
CACHE_DIR = Path(".", ".quickdrawcache")


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
        root: str | Path | None = Path("dataset/quick_draw/face.npy"),
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


class QuickDraw(VisionDataset):
    def __init__(
        self,
        category: str = "face",
        recognized: bool = None,
        transform: Optional[Callable] = None,
        download: bool = False,
        root: str = None,
    ) -> None:
        super().__init__(root=root or CACHE_DIR, transform=transform)

        self.category = category
        self.recognized = recognized

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        self.data = self._load_data().reshape(-1, 28, 28)

    @property
    def folder(self) -> Path:
        return Path(self.root, self.__class__.__name__, self.category)

    def _check_exists(self) -> bool:
        return self.folder.exists()

    def _check_all_files_exists(self) -> bool:
        all_data = self.folder / f"{self.category}.npy"
        recog_data = self.folder / f"{self.category}_recognized.npy"
        not_recog_data = self.folder / f"{self.category}_not_recognized.npy"
        return (
            all_data.exists()
            and recog_data.exists()
            and not_recog_data.exists()
        )

    def download(self):
        if self._check_all_files_exists():
            return
        import ndjson

        # Create path of directories
        self.folder.mkdir(parents=True, exist_ok=True)
        # Download npy file
        url = f"https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{self.category}.npy"
        try:
            print(f"Downloading {url}")
            response = requests.get(url)
            response.raise_for_status()
            data = np.load(io.BytesIO(response.content))
            np.save(self.folder / f"{self.category}.npy", data)
        except Exception as error:
            print(f"Failed to download (trying next):\n{error}")
            return
        finally:
            print()

        # Download ndjson file
        url = f"https://storage.googleapis.com/quickdraw_dataset/full/raw/{self.category}.ndjson"
        try:
            print(f"Downloading {url}")
            response = requests.get(url)
            items = response.json(cls=ndjson.Decoder)
            recognized = []
            for item in items:
                recognized.append(item["recognized"])
            recognized = np.array(recognized)
            data_recognized = data[recognized]
            data_not_recognized = data[~recognized]
            np.save(
                self.folder / f"{self.category}_recognized.npy", data_recognized
            )
            np.save(
                self.folder / f"{self.category}_not_recognized.npy",
                data_not_recognized,
            )
        except Exception as error:
            print(f"Failed to download (trying next):\n{error}")
        finally:
            print()

    def _load_data(self):
        if self.recognized is None:
            return np.load(self.folder / f"{self.category}.npy")

        if self.recognized:
            return np.load(self.folder / f"{self.category}_recognized.npy")

        return np.load(self.folder / f"{self.category}_not_recognized.npy")

    def __getitem__(self, index):
        img = Image.fromarray(self.data[index], mode="L")

        if self.transform:
            img = self.transform(img)

        return img, 0

    def __len__(self):
        return len(self.data)


def get_h5_dataset(
    path="shoes_images/shoes.hdf5", batch_size=128, shuffle=True
):
    return data.DataLoader(
        dataset=H5Loader(path), batch_size=batch_size, shuffle=shuffle
    )
