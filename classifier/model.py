from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.functional import accuracy


class Model(pl.LightningModule):
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (3, 224, 224),
        num_classes: int = 53,
        hidden_size: int = 64,
        learning_rate: float = 1e-3,
    ):
        super().__init__()

        # Init parameters
        self.channels, self.width, self.height = input_shape
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        # Design model
        self.model = self.create_model()

        # Loss function
        self.loss = nn.CrossEntropyLoss()

        # Optimizer Parameters
        self.learning_rate = learning_rate

        # Save hyperparameters to self.hparams (auto-logged by WandbLogger)
        self.save_hyperparameters()

    def create_model(self) -> nn.Module:
        """Create a simple MLP model"""
        model = nn.Sequential(
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
        return model

    def forward(self, x) -> torch.Tensor:
        """Method used for inference"""
        return self.model(x)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Needs to return loss for a single batch"""
        _, loss, acc = self._get_preds_loss_acc(batch)

        # Log loss and metric
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Used for logging metrics"""
        preds, loss, acc = self._get_preds_loss_acc(batch)

        # Log loss and metric
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        # Let's return preds to use it in a custom callback
        return preds

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Used for logging metrics"""
        _, loss, acc = self._get_preds_loss_acc(batch)

        # Log loss and metric
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

    def _get_preds_loss_acc(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convenience method to get preds, loss and acc for a batch. Used in training_step,
        validation_step and test_step.
        """
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)

        loss = self.loss(logits, y)
        acc = accuracy(preds, y, task="multiclass", num_classes=self.num_classes)
        return preds, loss, acc

    def configure_optimizers(self) -> torch.optim:
        """Initialize Adam optimizer"""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
