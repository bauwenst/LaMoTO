"""
Multiple-choice subtasks.
"""
from typing import Tuple, Any

from archit.instantiation.tasks import ForSingleAnswerMultipleChoice
from archit.instantiation.heads import SequenceClassificationHeadForNestedBatches
from datasets import DatasetDict, load_dataset
from transformers import DataCollator, EvalPrediction, AutoModelForMultipleChoice

from ._core import Task, MetricSetup
from .superglue.copa import DataCollatorForMultipleChoice
from ..util.datasets import imputeTestSplit, replaceDatasetColumns_OneExampleToOneExample


class SWAG(Task[SequenceClassificationHeadForNestedBatches]):
    """
    Single-choice "next-sentence selection".
    https://aclanthology.org/D18-1009/
    """

    ENDING_FIELDS = ["ending0", "ending1", "ending2", "ending3"]

    def __init__(self):
        super().__init__(
            task_name="SWAG",
            text_fields=["sent1", "sent2"] + SWAG.ENDING_FIELDS,
            label_field="label",
            metric_config=MetricSetup(
                to_compute=["precision", "recall", "f1", "accuracy"],
                to_track={
                    "precision": {"precision": "Pr"},
                    "recall":    {"recall": "Re"},
                    "f1":        {"f1": "$F_1$"},
                    "accuracy":  {"accuracy": "Acc"}
                }
            ),
            num_labels=1,
            archit_class=ForSingleAnswerMultipleChoice,
            automodel_class=AutoModelForMultipleChoice
        )

    def _loadDataset(self) -> DatasetDict:
        return imputeTestSplit(load_dataset("allenai/swag", "regular"), column_for_stratification="label", seed=self.hyperparameters.SEED)

    def _prepareDataset(self, dataset: DatasetDict) -> DatasetDict:
        def preprocess(example: dict) -> dict:
            return self.tokenizer(len(SWAG.ENDING_FIELDS)*[example["sent1"]], [example["sent2"] + " " + example[ending] for ending in SWAG.ENDING_FIELDS],
                                  add_special_tokens=self.hyperparameters.ADD_SPECIAL_TOKENS, truncation=True, max_length=self._getMaxInputLength())
        return replaceDatasetColumns_OneExampleToOneExample(dataset, preprocess, but_keep={"label"})

    def getCollator(self) -> DataCollator:
        return DataCollatorForMultipleChoice(self.tokenizer, padding="longest", max_length=self._getMaxInputLength())

    def getPredictionsAndReferences(self, eval: EvalPrediction) -> Tuple[Any,Any]:
        return eval.predictions.squeeze().argmax(-1).tolist(), eval.label_ids.squeeze().tolist()

    def adjustHyperparameters(self, hp):
        hp.archit_head_config.num_labels = 1
