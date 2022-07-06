import datetime
import pytz
import time
from pathlib import Path
import subprocess

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import callbacks
from pytorch_lightning.strategies.ddp import DDPStrategy
from argparse import ArgumentParser
import torch

from model import LongCheckerModel

from data import *
#import lib.longformer.data as dm
#from lib.longformer.model import SciFactModel


def get_timestamp():
    "Store a timestamp for when training started."
    timestamp = time.time()
    timezone = pytz.timezone("America/Los_Angeles")
    dt = datetime.datetime.fromtimestamp(timestamp, timezone)
    return dt.strftime("%Y-%m-%d:%H:%m:%S")


def get_checksum():
    "Keep track of the git checksum for this experiment."
    p1 = subprocess.Popen(["git", "rev-parse", "--short", "HEAD"], stdout=subprocess.PIPE)
    stdout, stderr = p1.communicate()
    res = stdout.decode("utf-8").split("\n")[0]
    return res


def get_folder_names(args):
    """
    Make a folder name as a combination of timestamp and experiment name (if
    given).

    If a slurm ID is given, just name it by its slurm id.
    """
    if args.slurm_job_id is None:
        timestamp = time.time()
        timezone = pytz.timezone("America/Los_Angeles")
        dt = datetime.datetime.fromtimestamp(timestamp, timezone)
        # Seconds past the start of the current day.
        second_past = 3600 * dt.hour + 60 * dt.minute + dt.second
        name = dt.strftime("%y_%m_%d_") + str(second_past)
    else:
        name = str(args.slurm_job_id)

    if args.experiment_name is not None:
        name += f"_{args.experiment_name}"

    # If the out directory exists, start appending integer suffixes until we
    # find a new one.
    out_dir = Path(args.result_dir) / name
    if out_dir.exists():
        suffix = 0
        candidate = Path(f"{str(out_dir)}_{suffix}")
        while candidate.exists():
            suffix += 1
            candidate = Path(f"{str(out_dir)}_{suffix}")
        out_dir = candidate

    # A bunch of annoying path jockeying to make things work out.
    checkpoint_dir = str(out_dir / "checkpoint")
    version = out_dir.name
    parent = out_dir.parent
    name = parent.name
    save_dir = str(parent.parent)

    return save_dir, name, version, checkpoint_dir


def get_num_training_instances(args):
    """
    Need to hard-code the number of training instances for each dataset so that
    we can set the optimizer correctly.
    """
    # Need to load in and set up the datset to figure out how many instances.
    data_module = dm.ConcatDataModule(args)
    data_module.setup()

    return len(data_module.folds["train"])


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_file", type=str, default="data/dataset.csv")
    parser.add_argument("--corpus_file", type=str, default=None)
    parser.add_argument("--train_file", type=str, default="data/train_med_fact.csv")
    parser.add_argument("--val_file", type=str, default="data/train_med_fact.csv")
    parser.add_argument("--slurm_job_id", type=int, default=None)
    parser.add_argument("--starting_checkpoint", type=str, default="checkpoints/fever_sci.ckpt")
    #parser.add_argument("--monitor", type=str, default="valid_sentence_label_f1")
    parser.add_argument("--result_dir", type=str, default="results/lightning_logs")
    parser.add_argument("--experiment_name", type=str, default="training_longformer")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mydata", type=int, default=1)
    #parser.add_argument("--accelerator", type=str, default='gpu')
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LongCheckerModel.add_model_specific_args(parser)
    #parser = dm.ConcatDataModule.add_model_specific_args(parser)

    args = parser.parse_args()
    args.timestamp = get_timestamp()
    args.git_checksum = get_checksum()
    return args


def main():
    pl.seed_everything(76)

    args = parse_args()

    # Get the appropriate dataset.
    #data_module = dm.ConcatDataModule(args)
    train_dataloader = get_dataloader(args, True)
    val_dataloader = get_dataloader(args, False)

    args.num_training_instances = len(train_dataloader.dataset) #get_num_training_instances(args)

    # Create the model.
    if args.starting_checkpoint is not None:
        # Initialize weights from checkpoint and override hyperparams.
        model = LongCheckerModel.load_from_checkpoint(
            args.starting_checkpoint, hparams=args)
    else:
        # Initialize from scratch.
        model = LongCheckerModel(args)

    # Loggers
    save_dir, name, version, checkpoint_dir = get_folder_names(args)

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=save_dir, name=name, version=version)
    csv_logger = pl_loggers.CSVLogger(
        save_dir=save_dir, name=name, version=version)
    loggers = [tb_logger, csv_logger]

    # Checkpointing.
    checkpoint_callback = callbacks.ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=1, save_last=True,
        dirpath=checkpoint_dir)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=3, verbose=False, mode="min")
    lr_callback = callbacks.LearningRateMonitor(logging_interval="step")
    gpu_callback = callbacks.DeviceStatsMonitor()
    trainer_callbacks = [early_stop_callback, checkpoint_callback, lr_callback, gpu_callback]

    # DDP pluging fix to keep training from hanging.
    if args.accelerator == "gpu":
        strategy = DDPStrategy(find_unused_parameters=True)
    else:
        strategy = None

    # Create trainer and fit the model.
    # Need `find_unused_paramters=True` to keep training from randomly hanging.
    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=trainer_callbacks, logger=loggers, strategy=strategy)

    # If asked to scale the batch size, tune instead of fitting.
    #if args.auto_scale_batch_size:
    #    print("Scaling batch size.")
    #    trainer.tune(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    #else:

    gc.collect()
    torch.cuda.empty_cache()
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == "__main__":
    main()
