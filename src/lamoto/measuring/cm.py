from typing import Dict, Set
from torch import Tensor

import torch
import evaluate

from ._core import StreamedMetric


class BinaryConfusionMatrix(StreamedMetric):
    """
    An even better suite of metrics based on a binary confusion matrix.
        - Accuracy;
        - The matrix cells: TP, TN, FP, FN;
        - Pr, Re and F1 for BOTH classes, not just the positive class;
        - Class-macro averages for Pr, Re and F1.
    """

    def __init__(self, _=None, threshold: float=0.5):
        super().__init__(_)
        self.threshold = threshold
        self.cache = None
        self.matrix = evaluate.load("BucketHeadP65/confusion_matrix")  # TP = answerable predicted as answerable.

    def add(self, logits: Tensor, labels: Tensor) -> torch.BoolTensor:
        thresholded_predictions = torch.nn.functional.softmax(logits, dim=-1)[:,1] > self.threshold  # If you didn't have a dedicated UA head, you could impute this with when the QA head says  start_index != end_index.

        for example_idx in range(thresholded_predictions.shape[0]):
            think_has_answer = bool(thresholded_predictions[example_idx])
            has_answer       = bool(labels[example_idx])
            self.matrix.add(prediction=think_has_answer, reference=has_answer)
            self.cache = has_answer

        return thresholded_predictions

    @classmethod
    def keys(cls) -> Set[str]:
        return {
            "accuracy",
            "TN", "FP", "FN", "TP",
            "Pr+", "Re+", "F1+",
            "Pr-", "Re-", "F1-",
            "PrM", "ReM", "F1M"
        }

    def computeFromMemory(self) -> Dict[str, float]:
        if self.cache is None:
            raise RuntimeError("Cannot compute metric across zero examples.")

        matrix = self.matrix.compute()["confusion_matrix"]
        if len(matrix) == 1:  # That means you have a matrix [[n]] where n is either TP or TN.
            if self.cache == True:  # This means you have so far only answered with TPs.
                matrix = [[0,           0],
                          [0, matrix[0,0]]]
            else:
                matrix = [[matrix[0,0], 0],
                          [0,           0]]

        # Name the confusion cells
        tn = matrix[0][0]  # Unanswerable predicted as unanswerable
        fp = matrix[0][1]  # Unanswerable (first row) predicted as answerable (second column)
        fn = matrix[1][0]
        tp = matrix[1][1]

        metrics = dict()
        metrics["TN"] = tn  # Unanswerable predicted as unanswerable
        metrics["FP"] = fp  # Unanswerable (first row) predicted as answerable (second column)
        metrics["FN"] = fn
        metrics["TP"] = tp

        # Compute clean F1 (a.o.t. weird HuggingFace effects which are influenced by the span)
        predicted_answerable = tp + fp
        pr_p = tp/predicted_answerable if predicted_answerable else 1.0
        actually_answerable = tp + fn
        re_p = tp/actually_answerable if actually_answerable else 1.0
        f1_p = 2*pr_p*re_p/(pr_p+re_p)
        metrics["Pr+"] = pr_p
        metrics["Re+"] = re_p
        metrics["F1+"] = f1_p

        # F1 complement (if you switch the classes of F1, its value changes)
        predicted_unanswerable = tn + fn
        pr_n = tn/predicted_unanswerable if predicted_unanswerable else 1.0
        actually_unanswerable = tn + fp
        re_n = tn/actually_unanswerable if actually_unanswerable else 1.0
        f1_n = 2*pr_n*re_n/(pr_n+re_n)
        metrics["Pr-"] = pr_n
        metrics["Re-"] = re_n
        metrics["F1-"] = f1_n

        # Macros
        metrics["PrM"] = (pr_p + pr_n)/2
        metrics["ReM"] = (re_p + re_n)/2  # https://scikit-learn.org/stable/modules/model_evaluation.html#balanced-accuracy-score
        metrics["F1M"] = (f1_p + f1_n)/2

        # Accuracy
        total = tp + tn + fp + fn
        metrics["accuracy"] = (tp+tn)/total      # true predictions out of all predictions

        return metrics
