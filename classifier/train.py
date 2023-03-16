import os
from pathlib import Path

import click
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from classifier.callbacks import LogPredictionsCallback
from classifier.dataset import CardsDataModule
from classifier.model import Model


@click.command()
@click.option(
    "-d",
    "--data_dir",
    default=Path("data"),
    type=click.Path(),
    show_default=True,
    help="Path to the data directory",
)
@click.option(
    "-n",
    "--normalize",
    default=False,
    is_flag=True,
    help="Whether to normalize the data",
)
@click.option(
    "-b",
    "--batch-size",
    default=64,
    type=int,
    show_default=True,
    help="Batch size for training",
)
@click.option(
    "-lr",
    "--learning-rate",
    default=1e-5,
    help="Learning rate for training",
)
@click.option(
    "-p",
    "--patience",
    default=5,
    help="Patience for early stopping",
)
@click.option(
    "-fe",
    "--feature-extractor",
    is_flag=True,
    help="Whether to freeze the feature extractor",
)
@click.option(
    "-e",
    "--expt-name",
    required=True,
    help="Name of the experiment",
)
def main(
    data_dir: Path,
    normalize: bool,
    batch_size: int,
    learning_rate: float,
    patience: int,
    feature_extractor: bool,
    expt_name: str,
):
    pl.seed_everything(42)

    # Init logger
    wandb_logger = WandbLogger(
        name=expt_name, project="Cards Classifier", log_model="all"
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode="max", verbose=True)
    log_predictions_callback = LogPredictionsCallback(num_samples=10)
    early_stop_callback = EarlyStopping(
        monitor="val_acc",
        patience=patience,
        verbose=True,
        mode="max",
    )

    # Init DataModule
    dm = CardsDataModule(
        data_dir=data_dir,
        num_workers=os.cpu_count(),
        normalize=normalize,
        batch_size=batch_size,
    )

    # Init model from datamodule's attributes
    model = Model(
        num_classes=53, learning_rate=learning_rate, feature_extractor=feature_extractor
    )

    # Init trainer
    trainer = pl.Trainer(
        max_epochs=-1,
        accelerator="auto",
        devices=None,
        logger=wandb_logger,
        callbacks=[early_stop_callback, log_predictions_callback, checkpoint_callback],
    )

    # Pass the datamodule as arg to trainer.fit to override model hooks :)
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
