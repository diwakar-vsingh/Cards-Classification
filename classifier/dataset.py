import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import kaggle
import pandas as pd
import torch
import torchvision.transforms as T
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.io import read_image

from classifier.constant import MEAN, STD


def download_kaggle_dataset(dataset_name: str, dataset_path: Path) -> None:
    """Download a dataset from Kaggle and unzip it.

    Args:
        dataset_name (str): Name of the dataset on Kaggle. It's usually in the format
            of "username/dataset-name".
        dataset_path (Path): Path to the directory where you want to download the dataset.
    """
    # Authenticate your API token
    kaggle.api.authenticate()

    # Download the dataset
    kaggle.api.dataset_download_files(dataset_name, path=dataset_path, unzip=True)


def get_norm_stats(data_dir: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get the mean and standard deviation of the training set."""
    dataset = CardsDataset(data_dir)
    img_tensor = torch.stack([dataset[idx][0] for idx in range(len(dataset))], dim=0)
    mean = img_tensor.mean(dim=(0, 2, 3))
    std = img_tensor.std(dim=(0, 2, 3))

    return mean, std


class CardsDataset(VisionDataset):
    def __init__(
        self,
        root: Path,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = split
        self.df = self.read_metadata()
        self.vocab: Dict[int, str] = (
            self.df[["class index", "labels"]]
            .set_index("class index")
            .to_dict()["labels"]
        )
        self.image_paths: List[Path] = self.df.filepaths.tolist()
        self.labels: List[str] = self.df["class index"].tolist()

    @property
    def num_classes(self) -> int:
        """Return number of classes."""
        return len(self.vocab)

    def read_metadata(self) -> pd.DataFrame:
        df = pd.read_csv(self.root / "cards.csv")
        df = df[df["data set"] == self.split]
        df.filepaths = df.filepaths.map(lambda x: Path("data") / x)
        return df

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path: str = str(self.image_paths[idx])
        label: str = self.labels[idx]
        try:
            image: torch.Tensor = read_image(image_path).float() / 255.0
        except RuntimeError:
            return self.__getitem__(random.randint(0, len(self)))

        input_transform = self.transform or T.ToTensor()
        image_t = input_transform(image)

        target_transform = self.target_transform or torch.as_tensor
        label_t = target_transform(label)

        return (image_t, label_t)


class CardsDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        num_workers: int = 8,
        normalize: bool = True,
        batch_size: int = 8,
        shuffle: bool = True,
        pin_memory: bool = False,
        drop_last: bool = True,
        train_transforms: Optional[Callable] = None,
        val_transforms: Optional[Callable] = None,
        test_transforms: Optional[Callable] = None,
    ):
        """
        Args:
            data_dir: Path to the dataset directory containing the train, valid, and test folders
            num_workers: Number of workers to use for loading the data
            normalize: If True, normalizes the data using the mean and standard deviation of the training set
            batch_size: Batch size to use for training, validation, and testing
            shuffle: If True, shuffles the train data at every epoch
            pin_memory: If True, the data loader will copy Tensors into CUDA pinned memory before returning them
            drop_last: If True, drops the last incomplete batch
        """
        super().__init__()
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.normalize = normalize
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.train_transform = train_transforms
        self.valid_transform = val_transforms
        self.test_transform = test_transforms

    def prepare_data(self) -> None:
        # download and split the data
        # called only on 1 GPU in distributed
        if not self.data_dir.exists():
            download_kaggle_dataset(
                "gpiosenka/cards-image-datasetclassification", self.data_dir
            )

    def setup(self, stage: Optional[str] = None) -> None:
        # called on every process in DDP
        if stage == "fit" or stage is None:
            train_transforms = self.train_transform or self.augmentation()
            valid_transforms = self.valid_transform or self.transform()
            self.train_dataset = CardsDataset(
                self.data_dir, split="train", transform=train_transforms
            )
            self.valid_dataset = CardsDataset(
                self.data_dir, split="valid", transform=valid_transforms
            )

        if stage == "test" or stage is None:
            test_transform = self.test_transform or self.transform()
            self.test_dataset = CardsDataset(
                self.data_dir, split="test", transform=test_transform
            )

    def augmentation(self) -> Callable:
        transforms: List[torch.nn.Module] = [
            T.RandomAutocontrast(p=0.2),
            T.RandomAdjustSharpness(sharpness_factor=2.0, p=0.2),
            T.RandomInvert(p=0.1),
            T.ColorJitter(
                brightness=(1, 2), contrast=0.5, saturation=0.5, hue=(-0.25, 0.25)
            ),
            T.RandomRotation(
                degrees=(-10, 10),
                interpolation=T.InterpolationMode.BILINEAR,
                expand=True,
                fill=random.random(),
            ),
            T.RandomPerspective(distortion_scale=0.1, p=0.2, fill=random.random()),
            T.Resize(size=(224, 224), antialias=True),
        ]
        if self.normalize:
            transforms += [T.Normalize(MEAN, STD)]

        return T.Compose(transforms)

    def transform(self) -> Callable:
        transforms: List[torch.nn.Module] = [T.Resize(size=(224, 224), antialias=True)]
        if self.normalize:
            transforms += [T.Normalize(MEAN, STD)]
        return T.Compose(transforms)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def one_batch(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get a batch of data from the training set."""
        for batch in self.train_dataloader():
            return batch
        return None
