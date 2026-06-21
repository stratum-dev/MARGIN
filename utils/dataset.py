"""
PyTorch Dataset for vulnerability-classification source code.

Wraps a HuggingFace dataset and tokenizes code snippets on-the-fly,
producing fixed-length input tensors along with label indices.
"""

from collections import Counter

from torch.utils.data import Dataset
from transformers import AutoTokenizer


class CodeDataset(Dataset):
    """
    Tokenized code dataset for multi-class vulnerability detection.

    Each sample is tokenised with a pretrained transformer tokenizer and
    returned as a dictionary of ``input_ids``, ``attention_mask``,
    ``label_idx``, and ``raw_label``.

    Parameters
    ----------
    backbone : str
        HuggingFace model name or path (e.g. ``"microsoft/unixcoder-base"``).
    hf_dataset : datasets.Dataset
        A HuggingFace dataset split with ``"source"`` and ``"label"`` columns.
    """

    def __init__(self, backbone: str, hf_dataset: str):
        self.dataset = hf_dataset
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        self.max_length = 512
        self.label2idx = {}
        self.idx2label = {}
        self._build_label_mapping()

    def _build_label_mapping(self):
        """
        Build bidirectional ``label2idx`` / ``idx2label`` mappings.

        Labels are sorted so that ``"Non-vul"`` always comes first (index 0);
        all other labels follow in alphabetical order.
        """
        labels = list(set(self.dataset["label"]))

        sorted_labels = sorted(labels, key=lambda x: (x != "Non-vul", x))

        for idx, label in enumerate(sorted_labels):
            self.label2idx[label] = idx
            self.idx2label[idx] = label

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Return the tokenized sample at position *idx*.

        Returns
        -------
        dict
            ``input_ids``, ``attention_mask`` (both ``torch.Tensor`` of shape
            ``(max_length,)``), ``label_idx`` (int), ``raw_label`` (str).
        """
        item = self.dataset[idx]
        code = item["source"]
        label = item["label"]

        encoding = self.tokenizer(
            code,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label_idx": self.label2idx[label],
            "raw_label": label,
        }
