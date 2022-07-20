import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import AutoTokenizer

import util
import torch


MAX_LEN = 512

def get_tokenizer():
    "Need to add a few special tokens to the default longformer checkpoint."
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-large-4096", model_max_length=MAX_LEN)
    ADDITIONAL_TOKENS = {
        "section_start": "<|sec|>",
        "section_end": "</|sec|>",
        "section_title_start": "<|sec-title|>",
        "section_title_end": "</|sec-title|>",
        "abstract_start": "<|abs|>",
        "abstract_end": "</|abs|>",
        "title_start": "<|title|>",
        "title_end": "</|title|>",
        "sentence_sep": "<|sent|>",
        "paragraph_sep": "<|par|>",
    }
    tokenizer.add_tokens(list(ADDITIONAL_TOKENS.values()))
    return tokenizer


class LongCheckerDataset(Dataset):
    "Stores and tensorizes a list of claim / document entries."

    def __init__(self, entries, tokenizer):
        self.entries = entries
        self.tokenizer = tokenizer
        self.label_lookup = {"REFUTES": 0, "NOT ENOUGH INFO": 1, "SUPPORTS": 2}

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        "Tensorize a single claim / abstract pair."
        entry = self.entries[idx]
        res = {
            "claim_id": entry["claim_id"],
            "abstract_id": entry["abstract_id"],
            "label": self.label_lookup[ entry["label"] ]
        }
        tensorized = self._tensorize(**entry["to_tensorize"])
        res.update(tensorized)
        return res

    def _tensorize(self, claim, sentences, title=None):
        """
        This function does the meat of the preprocessing work. The arguments
        should be self-explanatory, except `title`. We have abstract titles for
        SciFact, but not FEVER.
        """
        tokenized, abstract_sent_idx = self._tokenize(claim, sentences, title)

        # Get the label and the rationales.
        return {
            "tokenized": tokenized,
            "abstract_sent_idx": abstract_sent_idx,
        }

    def _tokenize(self, claim, sentences, title):
        cited_text = self.tokenizer.eos_token.join(sentences)
        if title is not None:
            cited_text = title + self.tokenizer.eos_token + cited_text
        tokenized = self.tokenizer(claim + self.tokenizer.eos_token + cited_text, truncation=True, max_length=MAX_LEN)
        tokenized["global_attention_mask"] = self._get_global_attention_mask(tokenized)
        abstract_sent_idx = self._get_abstract_sent_tokens(tokenized, title)

        # Make sure we've got the right number of abstract sentence tokens.
        assert len(abstract_sent_idx) == len(sentences) or len(tokenized['input_ids']) == MAX_LEN

        return tokenized, abstract_sent_idx

    def _get_global_attention_mask(self, tokenized):
        "Assign global attention to all special tokens and to the claim."
        input_ids = torch.tensor(tokenized.input_ids)
        # Get all the special tokens.
        is_special = (input_ids == self.tokenizer.bos_token_id) | (
            input_ids == self.tokenizer.eos_token_id
        )
        # Get all the claim tokens (everything before the first </s>).
        first_eos = torch.where(input_ids == self.tokenizer.eos_token_id)[0][0]
        is_claim = torch.arange(len(input_ids)) < first_eos
        # Use global attention if special token, or part of claim.
        global_attention_mask = is_special | is_claim
        # Unsqueeze to put in batch form, and cast like the tokenizer attention mask.
        global_attention_mask = global_attention_mask.to(torch.int64)
        return global_attention_mask.tolist()

    def _get_abstract_sent_tokens(self, tokenized, title):
        "Get the indices of the </s> tokens representing each abstract sentence."
        is_eos = torch.tensor(tokenized["input_ids"]) == self.tokenizer.eos_token_id
        eos_idx = torch.where(is_eos)[0]
        # If there's a title, the first two </s> tokens are for the claim /
        # abstract separator and the title. Keep the rest.
        # If no title, keep all but the first.
        start_ix = 1 if title is None else 2
        return eos_idx[start_ix:].tolist()


class LongCheckerReader:
    """
    Class to handle SciFact with retrieved documents.
    """
    def __init__(self, predict_args, input_file):
        self.data_file = input_file
        self.corpus_file = predict_args.corpus_file
        # Basically, I used two different sets of labels. This was dumb, but
        # doing this mapping fixes it.
        self.label_map = {"supported": "SUPPORTS",
                          "refuted": "REFUTES", "nei": "NOT ENOUGH INFO"}

    def get_mydata(self, tokenizer, data_path, val_file=None, val_div=False):
        """ Get our data """
        # if the csv file is too large, consider reading it as an iterable object with the chunksize argument
        data = pd.read_csv(data_path)
        data.rename(columns={data.columns[0]: 'id'}, inplace=True)
        if val_div:
            if val_file is None:
                train_df, val_df = util.divide_train_val(data, train_frac=0.93)
                train_df.to_csv("data/train_sub.csv")
                val_df.to_csv("data/val_sub.csv")
            else:
                val_data = pd.read_csv(data_path)
                val_data.rename(columns={val_data.columns[0]: 'id'}, inplace=True)
                train_df = util.divide_train_val(data, train_frac=1)
                val_df = util.divide_train_val(val_data, train_frac=1)
            train_lcds = LongCheckerDataset(util.convert_df2list(train_df, self.label_map), tokenizer)
            val_lcds = LongCheckerDataset(util.convert_df2list(val_df, self.label_map), tokenizer)
            return train_lcds, val_lcds

        return LongCheckerDataset(util.convert_df2list(data, self.label_map), tokenizer)

    def get_data(self, tokenizer):
        """
        Get the data for the relevant fold.
        """
        res = []

        corpus = util.load_jsonl(self.corpus_file)
        corpus_dict = {x["doc_id"]: x for x in corpus}
        claims = util.load_jsonl(self.data_file)

        for claim in claims:
            for doc_id in claim["doc_ids"]:
                candidate_doc = corpus_dict[doc_id]
                to_tensorize = {"claim": claim["claim"],
                                "sentences": candidate_doc["abstract"],
                                "title": candidate_doc["title"]}
                entry = {"claim_id": claim["id"],
                         "abstract_id": candidate_doc["doc_id"],
                         "label": self._get_label(claim["evidence"], doc_id),
                         "to_tensorize": to_tensorize}
                res.append(entry)

        return LongCheckerDataset(res, tokenizer)

    def _get_label(self, evidences, doc_id):
        if(len(evidences)) == 0:
            return "NOT ENOUGH INFO"

        if str(doc_id) not in evidences.keys() or (len(evidences[str(doc_id)])) == 0:
            return "NOT ENOUGH INFO"
        votes = [self.label_map[ev["label"]] for ev in evidences[str(doc_id)]]
        assert len(set(votes)) == 1
        return max(set(votes), key=votes.count)

class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        "Collate all the data together into padded batch tensors."
        # NOTE(dwadden) Set missing values to 0 for `abstract_sent_idx` instead
        # of -1 because it's going to be used as an input to
        # `batched_index_select` later on in the modeling code.
        res = {
            "claim_id": self._collate_scalar(batch, "claim_id"),
            "abstract_id": self._collate_scalar(batch, "abstract_id"),
            "label": self._collate_scalar(batch, "label"),
            "tokenized": self._pad_tokenized([x["tokenized"] for x in batch]),
            "abstract_sent_idx": self._pad_field(batch, "abstract_sent_idx", 0),
        }

        # Make sure the keys match.
        assert res.keys() == batch[0].keys()
        return res

    @staticmethod
    def _collate_scalar(batch, field):
        "Collate scalars by concatting."
        return torch.tensor([x[field] for x in batch])

    def _pad_tokenized(self, tokenized):
        """
        Pad the tokenizer outputs. Need to do this manually because the
        tokenizer's default padder doesn't expect `global_attention_mask` as an
        input.
        """
        fields = ["input_ids", "attention_mask", "global_attention_mask"]
        pad_values = [self.tokenizer.pad_token_id, 0, 0]
        tokenized_padded = {}
        for field, pad_value in zip(fields, pad_values):
            tokenized_padded[field] = self._pad_field(tokenized, field, pad_value)

        return tokenized_padded

    def _pad_field(self, entries, field_name, pad_value):
        xxs = [entry[field_name] for entry in entries]
        return self._pad(xxs, pad_value)

    @staticmethod
    def _pad(xxs, pad_value):
        """
        Pad a list of lists to the length of the longest entry, using the given
        `pad_value`.
        """
        res = []
        max_length = max(map(len, xxs))
        for entry in xxs:
            to_append = [pad_value] * (max_length - len(entry))
            padded = entry + to_append
            res.append(padded)

        return torch.tensor(res)


# read in data and return 1 DataLoader for the dataset in data_file
# if used for our created dataset, data_file contains all attributes (set --mydata flag 1)
# if used for existing datasets, data_file has the id(s) to the doc_id in corpus.jsonl (set --mydata flag 0)
def get_dataloader(predict_args, data_file=None):
    "Main entry point to get the data loader. This can only be used at test time."
    reader = LongCheckerReader(predict_args, data_file)
    tokenizer = get_tokenizer()
    if predict_args.mydata:
        ds = reader.get_mydata(tokenizer, data_file)
    else:
        ds = reader.get_data(tokenizer)
    collator = Collator(tokenizer)
    return DataLoader(ds,
                      num_workers=predict_args.num_workers,
                      batch_size=predict_args.batch_size,
                      collate_fn=collator,
                      shuffle=False,
                      pin_memory=True)

# read in data and return two DataLoader(s) for a training set and a validation/test set
# only used for our dataset
def get_dataloaders(predict_args, data_file):
    "Main entry point to get the data loader. This can only be used at test time."
    reader = LongCheckerReader(predict_args, data_file)
    tokenizer = get_tokenizer()
    datasets = reader.get_mydata(tokenizer, data_file, val_div=True)
    collator = Collator(tokenizer)
    return [DataLoader(ds, num_workers=predict_args.num_workers, batch_size=predict_args.batch_size,
                       collate_fn=collator, shuffle=False, pin_memory=True) for ds in datasets]
