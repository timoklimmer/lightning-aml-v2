import pytorch_lightning as pl
import torch
from torch.nn import functional as F


class CustomModel(pl.LightningModule):
    def __init__(self):
        super(CustomModel, self).__init__()

        # assumed image size: (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 10)
        torch.nn.init.kaiming_normal_(
            self.layer_1.weight, mode="fan_out", nonlinearity="relu"
        )
        torch.nn.init.kaiming_normal_(
            self.layer_2.weight, mode="fan_out", nonlinearity="relu"
        )

        self.batch_norm1 = torch.nn.BatchNorm1d(128)

    def forward(self, x):
        batch_size, _, _, _ = x.size()
        x = x.view(batch_size, -1)
        x = self.batch_norm1(F.relu(self.layer_1(x)))
        x = self.layer_2(x)
        x = F.log_softmax(x, dim=1)

        return x

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log("test_loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer
