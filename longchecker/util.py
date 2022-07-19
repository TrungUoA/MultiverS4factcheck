"""
Shared utility functions.
"""

import json
import numpy as np
import pathlib
import os
import torch
import pandas as pd
from random import shuffle

from nltk.tokenize import sent_tokenize

def divide_train_val(data, train_frac=0.9, all_frac=1.0):
    train_subsets = []
    val_subsets = []
    for k in data["label"].unique():
        data_k = data[data["label"] == k]
        if all_frac < 1.0:
            data_k = data_k.sample(frac=all_frac)
        train_data_k = data_k.sample(frac=train_frac)
        val_data_k = pd.concat([data_k, train_data_k]).drop_duplicates(keep=False)
        train_subsets.append(train_data_k)
        val_subsets.append(val_data_k)
    train_data = pd.concat(train_subsets, axis=0)
    val_data = pd.concat(val_subsets, axis=0)

    return train_data, val_data

def convert_df2list(data, label_map):
    res = []
    for _, row in data.iterrows():
        to_tensorize = {"claim": row["claim"],
                        "sentences": sent_tokenize(row["exp"])}
        # "title": candidate_doc["title"]}
        entry = {"claim_id": row["id"],
                 "abstract_id": row["id"],
                 "label": label_map[row["label"]],
                 "to_tensorize": to_tensorize}
        res.append(entry)
    shuffle(res)
    return res

def load_jsonl(fname, max_lines=None):
    res = []
    for i, line in enumerate(open(fname)):
        if max_lines is not None and i == max_lines:
            return res
        else:
            res.append(json.loads(line))

    return res


class NPEncoder(json.JSONEncoder):
    "Handles json encoding of Numpy objects."
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NPEncoder, self).default(obj)


def write_jsonl(data, fname):
    with open(fname, "w") as f:
        for line in data:
            print(json.dumps(line, cls=NPEncoder), file=f)


def get_longformer_science_checkpoint():
    current_dir = pathlib.Path(os.path.realpath(__file__)).parent
    fname = current_dir.parent / "checkpoints/longformer_large_science.ckpt"

    return str(fname)


def unbatch(d, ignore=[]):
    """
    Convert a dict of batched tensors to a list of tensors per entry. Ignore any
    keys in the list.
    """
    ignore = set(ignore)

    to_unbatch = {}
    for k, v in d.items():
        # Skip ignored keys.
        if k in ignore:
            continue
        if isinstance(v, torch.Tensor):
            # Detach and convert tensors to CPU.
            new_v = v.detach().cpu().numpy()
        else:
            new_v = v

        to_unbatch[k] = new_v

    # Make sure all entries have same length.
    lengths = [len(v) for v in to_unbatch.values()]
    if len(set(lengths)) != 1:
        raise ValueError("All values must be of same length.")

    res = []
    for i in range(lengths[0]):
        to_append = {}
        for k, v in to_unbatch.items():
            to_append[k] = v[i]

        res.append(to_append)

    return res

def dict2cuda(d, device):
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = dict2cuda(value, device)
        else:
            d[key] = value.to(device)
    return d