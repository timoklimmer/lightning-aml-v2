import os
import pandas as pd
from torchvision import transforms
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import torch


class CustomDataset(Dataset):
    def __init__(self, input_dataset, transform=None):
        self.input_dataset = pd.read_csv(input_dataset)
        self.transform = transform

    def __len__(self):
        return len(self.input_dataset)

    def __getitem__(self, idx):
        row = self.input_dataset.iloc[idx]
        label = row[0]
        image_data = (
            row[1:].values.astype("float32").reshape((28, 28, 1))
        )  # Reshape to (28, 28, 1) for grayscale
        if self.transform:
            image = self.transform(image_data)
        else:
            image = (
                torch.tensor(image_data, dtype=torch.float32) / 255.0
            )  # Convert numpy array to tensor and normalize
        label = torch.tensor(label, dtype=torch.long)
        return image, label


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, input_dataset, batch_size=32):
        super().__init__()
        self.input_dataset = input_dataset
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Define the transforms
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5,), (0.5,)
                ),  # Mean and std dev as tuples for each channel
            ]
        )
        full_dataset = CustomDataset(self.input_dataset, transform=transform)

        # Calculate split sizes for 80-10-10 distribution
        train_size = int(0.8 * len(full_dataset))
        val_size = (len(full_dataset) - train_size) // 2
        test_size = (
            len(full_dataset) - train_size - val_size
        )  # This ensures all data is used

        # Perform the splits
        self.train_dataset, remaining_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, len(full_dataset) - train_size]
        )
        self.val_dataset, self.test_dataset = torch.utils.data.random_split(
            remaining_dataset, [val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=os.cpu_count()
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=os.cpu_count()
        )
