import resource
from multiprocessing import cpu_count
from pathlib import Path
from typing import Optional, Union

import click
import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import Logger, WandbLogger
from lightning.pytorch.tuner import Tuner

from classifier.callbacks import LogPredictionsCallback
from classifier.dataset import CardsDataModule
from classifier.io_utils import get_most_recently_edited_file
from classifier.model import LitModel

CKPT_DIR = Path("checkpoints")


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
    "-c",
    "--ckpt",
    default=None,
    type=click.Path(),
    help="Path to the checkpoint to resume training from",
)
@click.option(
    "-r",
    "--resume-training",
    is_flag=True,
    help="Whether to resume training from the most recent checkpoint",
)
@click.option(
    "-a",
    "--arch",
    default="resnet34",
    show_default=True,
    type=click.Choice(["resnet18", "resnet34", "resnet50", "resnet101"]),
    help="Architecture of the model to train",
)
@click.option(
    "-pt",
    "--pretrained",
    is_flag=True,
    help="Whether to use a pretrained model",
)
@click.option(
    "-fe",
    "--feature-extractor",
    is_flag=True,
    help="If True, only train the classifier head, else train the entire model",
)
@click.option(
    "-b",
    "--batch-size",
    default=128,
    type=int,
    show_default=True,
    help="Batch size for training",
)
@click.option(
    "-lr",
    "--learning-rate",
    default=1e-4,
    show_default=True,
    help="Learning rate for training",
)
@click.option(
    "-p",
    "--patience",
    default=5,
    show_default=True,
    help="Patience for early stopping",
)
@click.option(
    "-e",
    "--expt-name",
    required=True,
    help="Name of the experiment",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Whether to run in debug mode. This will turn off Wandb logging",
)
@click.option(
    "-tlr",
    "--tune-learning-rate",
    is_flag=True,
    help="Whether to tune the learning rate",
)
@click.option(
    "-tb",
    "--tune-batch-size",
    is_flag=True,
    help="Whether to tune the batch size",
)
def main(
    data_dir: Path,
    normalize: bool,
    ckpt: Optional[Path],
    resume_training: bool,
    arch: str,
    pretrained: bool,
    feature_extractor: bool,
    batch_size: int,
    learning_rate: float,
    patience: int,
    expt_name: str,
    debug: bool,
    tune_learning_rate: bool,
    tune_batch_size: bool,
):
    assert not (
        tune_batch_size and tune_learning_rate
    ), "Cannot tune both batch size and learning rate at the same time"
    L.seed_everything(42)

    # Set the soft and hard limits for the maximum number of open files
    _, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    new_soft_limit = 10000
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft_limit, hard_limit))

    # Callbacks
    early_stop_callback = EarlyStopping(
        monitor="val_acc",
        patience=patience,
        verbose=True,
        mode="max",
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        verbose=True,
        save_on_train_epoch_end=True,
        auto_insert_metric_name=True,
    )
    lr_monitor = LearningRateMonitor()
    callbacks = [early_stop_callback, checkpoint_callback, lr_monitor]

    log_predictions_callback = LogPredictionsCallback(num_samples=10)
    if not debug:
        callbacks.append(log_predictions_callback)

    # Load checkpoint
    ckpt = ckpt or (
        get_most_recently_edited_file(CKPT_DIR) if resume_training else None
    )

    # Init DataModule
    dm = CardsDataModule(
        data_dir=data_dir,
        num_workers=cpu_count() // 2,
        normalize=normalize,
        batch_size=batch_size,
    )

    # Init model from datamodule's attributes
    model = LitModel(
        num_classes=53,
        learning_rate=learning_rate,
        arch=arch,
        pretrained=pretrained,
        transfer=feature_extractor,
    )

    # Init logger
    logger: Union[Logger, bool] = False
    if not debug:
        logger = WandbLogger(
            name=expt_name, project="Cards Classifier", log_model="all"
        )

    # Init trainer
    trainer = L.Trainer(
        max_epochs=-1,
        accelerator="mps",
        logger=logger,
        log_every_n_steps=25,
        callbacks=callbacks,
    )
    if tune_batch_size or tune_learning_rate:
        # Init tuner
        tuner = Tuner(trainer)

        if tune_learning_rate:
            # Run learning rate finder
            lr_finder = tuner.lr_find(model, datamodule=dm)

            if lr_finder is not None:
                print(f"Suggested learning rate: {lr_finder.suggestion():.2e}")

                # Plot with returned figure, and suggested lr
                fig = lr_finder.plot(suggest=True)
                fig.savefig("lr_finder.png")

                # Pick point based on plot, or get suggestion
                new_lr = lr_finder.suggestion()

                # update hparams of the model
                model.hparams.learning_rate = new_lr

        if tune_batch_size:
            # Auto-scale batch size with binary search
            tuner.scale_batch_size(model, mode="power", datamodule=dm)

    # Train the model ⚡🚅⚡
    trainer.fit(model, datamodule=dm, ckpt_path=ckpt)

    # Evaluate the model on the held-out test set ⚡⚡
    trainer.test()


if __name__ == "__main__":
    main()
