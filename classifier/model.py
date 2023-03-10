import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy


class Model(pl.LightningModule):
    def __init__(
        self,
        channels: int = 3,
        width: int = 224,
        height: int = 224,
        num_classes: int = 53,
        hidden_size: int = 64,
        learning_rate: float = 1e-5,
    ):
        super().__init__()
        self.channels = channels
        self.width = width
        self.height = height
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                self.channels * self.width * self.height, self.hidden_size
            ),  # 224 * 224 = 50176 -> 64
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),  # 64 -> 64
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_classes),  # 64 -> 53
            nn.Softmax(dim=1),
        )

    def forward(self, x) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=self.num_classes)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self) -> torch.optim:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
