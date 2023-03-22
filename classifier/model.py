from functools import partial
from typing import Tuple

import lightning as L
import torch
import torch.nn as nn
from torchmetrics.functional import accuracy
from torchvision.models import ResNet18_Weights, resnet18


class LitModel(L.LightningModule):
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (3, 224, 224),
        num_classes: int = 53,
        learning_rate: float = 1e-3,
        feature_extractor: bool = False,
    ):
        super().__init__()

        # Init parameters
        self.channels, self.width, self.height = input_shape

        # Save hyperparameters to self.hparams (auto-logged by WandbLogger)
        self.save_hyperparameters()

        # Design model
        self.model = self.create_model(feature_extractor)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Metric
        self.metric = partial(accuracy, task="multiclass", num_classes=num_classes)

    def create_model(self, feature_extractor: bool) -> nn.Module:
        """Create model"""
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        if feature_extractor:
            self.set_parameter_requires_grad(model)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, self.hparams.num_classes)

        return model

    def set_parameter_requires_grad(self, model: nn.Module) -> None:
        for param in model.parameters():
            param.requires_grad = False

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

        loss = self.criterion(logits, y)
        acc = self.metric(preds, y)

        return preds, loss, acc

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Initialize Adam optimizer"""
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
