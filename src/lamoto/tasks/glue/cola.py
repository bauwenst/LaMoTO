from enum import Enum
from datasets import DatasetDict

import numpy.random as npr
import datasets

from ._general import ClassifySentenceGLUETask
from ...util.exceptions import ImpossibleBranchError


class BalancingMode(Enum):
    NONE                  = 1
    UPSAMPLING_MINORITY   = 2
    DOWNSAMPLING_MAJORITY = 3


class CoLA(ClassifySentenceGLUETask):
    """Detection of erroneously formed sentences. Note that the actual dataset has a 70-30 positive-negative skew."""
    def __init__(self, balancing: BalancingMode=BalancingMode.UPSAMPLING_MINORITY):
        super().__init__("CoLA")
        self._balancing = balancing

    def _loadDatasetRaw(self) -> DatasetDict:
        dataset_full = super()._loadDatasetRaw()
        if self._balancing == BalancingMode.NONE:
            return dataset_full

        rng = npr.default_rng(self.hyperparameters.SEED)
        def is_negative(example: dict) -> bool:
            return example["label"] == 0

        dataset_negatives = dataset_full.filter(is_negative)
        n_negatives_train = len(dataset_negatives["train"])
        n_negatives_valid = len(dataset_negatives["validation"])
        n_excess_positives_train = len(dataset_full["train"])      - 2*n_negatives_train  # positive - negatives == (all - negatives) - negatives
        n_excess_positives_valid = len(dataset_full["validation"]) - 2*n_negatives_valid

        if self._balancing == BalancingMode.UPSAMPLING_MINORITY:
            return datasets.DatasetDict({
                "train": datasets.concatenate_datasets([
                    dataset_full["train"],
                    dataset_negatives["train"].select(rng.integers(low=0, high=len(dataset_negatives["train"]), size=n_excess_positives_train))
                ]),
                "validation": datasets.concatenate_datasets([
                    dataset_full["validation"],
                    dataset_negatives["validation"].select(rng.integers(low=0, high=len(dataset_negatives["validation"]), size=n_excess_positives_valid))
                ])
            })
        elif self._balancing == BalancingMode.DOWNSAMPLING_MAJORITY:
            raise NotImplementedError()  # To implement this: isolate the positives from the dataset, sample n_negatives from them, and concatenate with the negatives.
        else:
            raise ImpossibleBranchError()
