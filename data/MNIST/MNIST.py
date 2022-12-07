import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST as tvd_MNIST

class MNIST(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./@data_cache"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.prepare_data_per_node = True

    def prepare_data(self):
        tvd_MNIST(self.data_dir, train=True, download=True)
        tvd_MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        if stage == "fit":
            mnist_full = tvd_MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        if stage == "test":
            self.mnist_test = tvd_MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.mnist_predict = tvd_MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32, num_workers=os.cpu_count())

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32, num_workers=os.cpu_count())

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32, num_workers=os.cpu_count())

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=32, num_workers=os.cpu_count())
