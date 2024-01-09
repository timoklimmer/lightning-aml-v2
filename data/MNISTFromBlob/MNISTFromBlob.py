import os
import pandas as pd
from torchvision import transforms
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import torch


class MNISTFromBlobDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        # Assuming we want the first CSV file in the folder
        files = [file for file in os.listdir(data_folder) if file.endswith(".csv")]
        if not files:
            raise RuntimeError("No CSV files found in the data folder")

        file_path = os.path.join(data_folder, files[0])
        self.input_dataset_df = pd.read_csv(file_path)
        self.transform = transform

    def __len__(self):
        return len(self.input_dataset_df)

    def __getitem__(self, idx):
        row = self.input_dataset_df.iloc[idx]
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


class MNISTFromBlob(pl.LightningDataModule):
    def __init__(self, data_folder, batch_size=32):
        super().__init__()
        self.data_folder = data_folder
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
        full_dataset = MNISTFromBlobDataset(self.data_folder, transform=transform)

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
