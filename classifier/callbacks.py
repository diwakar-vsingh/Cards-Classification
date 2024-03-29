from typing import List, Tuple

import lightning as L
import torch
from wandb.data_types import Image

import wandb


class LogPredictionsCallback(L.Callback):
    def __init__(self, num_samples: int = 20) -> None:
        super().__init__()
        self.num_samples = num_samples

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: torch.Tensor,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Called when the validation batch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let"s log {num_sample} sample image predictions from first batch
        if batch_idx == 0:
            x, y = batch

            # log predictions as a Table
            columns: List[str] = ["image", "ground truth", "prediction"]
            data: List[List[Image]] = [
                [wandb.Image(x_i), y_i, y_pred]  # type: ignore[attr-defined]
                for x_i, y_i, y_pred in list(
                    zip(
                        x[: self.num_samples],
                        y[: self.num_samples],
                        outputs[: self.num_samples],
                    )
                )
            ]

            trainer.logger.log_table(key="predictions", columns=columns, data=data)
