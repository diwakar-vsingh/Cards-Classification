from typing import Any, Dict, Optional, Tuple, Union

import lightning as L
import torch
import torch.nn as nn
import torchmetrics
from torchvision.models import get_model

import wandb

RESNETS: Tuple[str, ...] = ("resnet18", "resnet34", "resnet50", "resnet101")


class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`"

    def __init__(
        self, size: Optional[Union[int, Tuple[Optional[int], Optional[int]]]] = None
    ) -> None:
        super().__init__()
        self.size = size or 1
        self.ap = nn.AdaptiveAvgPool2d(self.size)
        self.mp = nn.AdaptiveMaxPool2d(self.size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.mp(x), self.ap(x)], dim=1)


class LitModel(L.LightningModule):
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (3, 224, 224),
        num_classes: int = 53,
        arch: str = "resnet34",
        learning_rate: float = 1e-3,
        pretrained: bool = True,
        transfer: bool = False,
    ):
        super().__init__()

        # Init parameters
        self.channels, self.width, self.height = input_shape

        # Save hyperparameters to self.hparams (auto-logged by WandbLogger)
        self.save_hyperparameters()

        # Design model
        self.model = self.create_model(arch)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Metric
        self.metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def create_model(self, arch: str) -> nn.Module:
        """Create model"""
        weights: Optional[str] = "DEFAULT" if self.hparams.pretrained else None
        assert arch in RESNETS, f"Only {RESNETS} are supported, got {arch}"
        backbone = get_model(arch, weights=weights)
        if self.hparams.transfer:
            self.set_parameter_requires_grad(backbone)

        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]

        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(
            AdaptiveConcatPool2d(),
            nn.Flatten(1),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(2 * num_filters, 512, bias=False),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, self.hparams.num_classes, bias=False),
        )

        return nn.Sequential(self.body, self.head)

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
        if self.trainer.global_step == 0:
            wandb.define_metric("train_acc", summary="mean")  # type: ignore
            wandb.define_metric("train_loss", summary="mean")  # type: ignore

        _, loss, acc = self._get_preds_loss_acc(batch)

        # Log loss and metric
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Used for logging metrics"""
        if self.trainer.global_step == 0:
            wandb.define_metric("val_acc", summary="mean")  # type: ignore
            wandb.define_metric("val_loss", summary="mean")  # type: ignore

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

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and/or learning rate scheduler"""
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val_acc",
        }
