from tqdm import tqdm
import argparse
from pathlib import Path

from model import LongCheckerModel
from data import get_dataloader
import util
from sklearn.metrics import f1_score
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy

import numpy as np

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--input_file", type=str, default="data/dataset.csv")
    parser.add_argument("--corpus_file", type=str, default=None)
    parser.add_argument("--test_file", type=str, default="data/test.csv")
    parser.add_argument("--output_file", type=str, default="prediction/result.jsonl")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--mydata", default=1, type=int)
    parser.add_argument(
        "--no_nei", action="store_true", help="If given, never predict NEI."
    )
    parser.add_argument("--debug", action="store_true")
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LongCheckerModel.add_model_specific_args(parser)

    return parser.parse_args()


def get_predictions(args):
    args = get_args()

    # Set up model and data.
    model = LongCheckerModel.load_from_checkpoint(checkpoint_path=args.checkpoint_path, hparams=args)
    # If not predicting NEI, set the model label threshold to 0.
    if args.no_nei:
        model.label_threshold = 0.0

    # Since we're not running the training loop, gotta put model on GPU.
    model.to(f"cuda:{args.device}")
    model.eval()
    model.freeze()

    # Grab model hparams and override using new args, when relevant.
    hparams = model.hparams["hparams"]
    del hparams.precision  # Don' use 16-bit precision during evaluation.
    for k, v in vars(args).items():
        if hasattr(hparams, k):
            setattr(hparams, k, v)

    dataloader = get_dataloader(args, args.test_file)

    # Make predictions.
    predicted_labels_all = []
    prediction_all = []
    labels = []

    for batch in tqdm(dataloader):
        batch = util.dict2cuda(batch, f"cuda:{args.device}")
        preds_batch = model( batch["tokenized"], batch["abstract_sent_idx"] )
        labels.extend(batch["label"])
        prediction_all.extend(preds_batch["predicted_labels"])
        predicted_labels_all.extend( model.decode(preds_batch, batch) )

    F1 = f1_score_func(np.array(prediction_all), np.array(labels))

    return predicted_labels_all, F1


def format_predictions(args, predictions_all):
    # Need to get the claim ID's from the original file, since the data loader
    # won't have a record of claims for which no documents were retireved.
    claims = util.load_jsonl(args.input_file)
    claim_ids = [x["id"] for x in claims]
    assert len(claim_ids) == len(set(claim_ids))

    formatted = {claim: {} for claim in claim_ids}

    # Dict keyed by claim.
    for prediction in predictions_all:
        # If it's NEI, skip it.
        if prediction["predicted_label"] == "NEI":
            continue

        # Add prediction.
        formatted_entry = {
            prediction["abstract_id"]: {
                "label": prediction["predicted_label"],
                "sentences": prediction["predicted_rationale"],
            }
        }
        formatted[prediction["claim_id"]].update(formatted_entry)

    # Convert to jsonl.
    res = []
    for k, v in formatted.items():
        to_append = {"id": k, "evidence": v}
        res.append(to_append)

    return res


def main():
    args = get_args()
    outname = Path(args.output_file)
    predictions, F1 = get_predictions(args)
    print("Weighted F1 score is" + F1)

    # Save final predictions as json.
    formatted = format_predictions(args, predictions)
    util.write_jsonl(formatted, outname)

def main_with_Trainer():
    args = get_args()

    # Get the appropriate dataset.
    test_dataloader = get_dataloader(args, args.test_file)
    args.num_training_instances = len(test_dataloader.dataset)  # get_num_training_instances(args)

    # Create the model.
    if args.checkpoint_path is not None:
        # Initialize weights from checkpoint and override hyperparams.
        model = LongCheckerModel.load_from_checkpoint(
            args.checkpoint_path, hparams=args)
    else:
        # Initialize from scratch.
        model = LongCheckerModel(args)

    # DDP pluging fix to keep training from hanging.
    if args.accelerator == "gpu":
        strategy = DDPStrategy(find_unused_parameters=True)
    else:
        strategy = None

    # Create trainer and fit the model.
    # Need `find_unused_paramters=True` to keep training from randomly hanging.
    trainer = pl.Trainer.from_argparse_args(args, strategy=strategy)
    print("Evaluating...")
    trainer.test(model, dataloaders=test_dataloader, verbose=True)

if __name__ == "__main__":
    main_with_Trainer()
