from typing import Optional, Tuple

import lightning as L
import torch
import torch.nn as nn
import torchmetrics
from torchvision import models
from torchvision.models._api import WeightsEnum


class LitModel(L.LightningModule):
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (3, 224, 224),
        num_classes: int = 53,
        learning_rate: float = 1e-3,
        transfer: bool = False,
    ):
        super().__init__()

        # Init parameters
        self.channels, self.width, self.height = input_shape

        # Save hyperparameters to self.hparams (auto-logged by WandbLogger)
        self.save_hyperparameters()

        # Design model
        self.model = self.create_model()

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Metric
        self.metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def create_model(self) -> nn.Module:
        """Create model"""
        weights: Optional[WeightsEnum] = None
        if self.hparams.transfer:
            weights = models.ResNet34_Weights.DEFAULT
        backbone = models.resnet34(weights=weights)
        if self.hparams.transfer:
            self.set_parameter_requires_grad(backbone)

        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]

        self.feature_extractor = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Flatten(1),
            nn.Dropout(),
            nn.Linear(num_filters, 512),
            nn.Dropout(),
            nn.Linear(512, self.hparams.num_classes),
        )

        return nn.Sequential(self.feature_extractor, self.fc)

    def set_parameter_requires_grad(self, model: nn.Module) -> None:
        """Freeze feature extractor"""
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
        acc = self.metric(preds, y)  # type: ignore

        return preds, loss, acc

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Initialize Adam optimizer"""
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
        )
