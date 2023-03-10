import os
from pathlib import Path

import click
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

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
    expt_name: str,
):
    # Init logger
    wandb_logger = WandbLogger(
        name=expt_name, project="Cards Classifier", log_model="all"
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode="max", verbose=True)
    early_stop_callback = EarlyStopping(
        monitor="val_acc",
        min_delta=0.00,
        patience=3,
        verbose=False,
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
    model = Model(num_classes=53, learning_rate=learning_rate)

    # Init trainer
    trainer = pl.Trainer(
        max_epochs=-1,
        accelerator="auto",
        devices=None,
        logger=wandb_logger,
        callbacks=[early_stop_callback, checkpoint_callback],
    )

    # Pass the datamodule as arg to trainer.fit to override model hooks :)
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
