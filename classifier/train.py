import os
from pathlib import Path

import click
import pytorch_lightning as pl

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
def main(
    data_dir: Path,
    normalize: bool,
    batch_size: int,
    learning_rate: float,
):
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
        max_epochs=5,
        accelerator="auto",
        devices=None,
    )

    # Pass the datamodule as arg to trainer.fit to override model hooks :)
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
