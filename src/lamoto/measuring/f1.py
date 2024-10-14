from typing import Dict

import evaluate
from ._core import LogitLabelMetric, Tensor


class MacroF1(LogitLabelMetric):
    """
    F1 score macro-averaged across all the classes present in the reference.
    Necessary for multiclass classification (i.e. more than 2 classes), since F1 is only defined for binary data.

    It is not micro-averaged because micro-F1 == accuracy.

    The average is unweighted, meaning the individual F1 scores of the small and big classes have the exact same
    contribution. This is clearly better than a weighted macro-F1. https://stackoverflow.com/a/78163857/9352077
    """
    def compute(self, predictions: Tensor, references: Tensor) -> Dict[str, float]:
        return evaluate.load("f1").compute(predictions=predictions, references=references, average="macro")


class MacroRecall(LogitLabelMetric):
    def compute(self, predictions: Tensor, references: Tensor) -> Dict[str, float]:
        return evaluate.load("recall").compute(predictions=predictions, references=references, average="macro")


class MacroPrecision(LogitLabelMetric):
    def compute(self, predictions: Tensor, references: Tensor) -> Dict[str, float]:
        return evaluate.load("precision").compute(predictions=predictions, references=references, average="macro")
