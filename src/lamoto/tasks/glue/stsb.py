from archit.instantiation.tasks import ForSequenceRegression, SequenceClassificationHeadConfig
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import load_dataset

from ._general import SequenceTaskHyperparameters
from .._core import *


class STSB(Task[SequenceClassificationHeadConfig]):
    """
    Regressive tasks that tries to score sentence similarity between 0 and 5.
    """

    def __init__(self):
        super().__init__(
            task_name="stsb",
            metric_config=MetricSetup(
                to_compute=["pearsonr", "spearmanr"],
                to_track={
                    "pearsonr": {"pearsonr": "Pearson"},
                    "spearmanr": {"spearmanr": "Spearman"}
                }
            ),
            archit_class=ForSequenceRegression,
            automodel_class=AutoModelForSequenceClassification,

            num_labels=1
        )

    def loadDataset(self) -> DatasetDict:
        return load_dataset("glue", "stsb")

    def prepareDataset(self, dataset: DatasetDict) -> DatasetDict:
        def preprocess(example):
            return self.tokenizer(example["sentence1"], example["sentence2"], add_special_tokens=self.hyperparameters.ADD_SPECIAL_TOKENS,
                                  truncation=True, max_length=self._getMaxInputLength())

        dataset = dataset.map(preprocess, batched=False)
        dataset = dataset.remove_columns(["sentence1", "sentence2", "idx"])
        return dataset

    def adjustHyperparameters(self, hp: SequenceTaskHyperparameters):
        hp.archit_head_config.num_labels = 1

    def getCollator(self) -> DataCollator:
        return DataCollatorWithPadding(self.tokenizer, padding="longest", max_length=self._getMaxInputLength())

    def getPredictionsAndReferences(self, eval: EvalPrediction) -> Tuple[Any,Any]:
        predictions, labels = eval.predictions.squeeze(), eval.label_ids
        return predictions.tolist(), labels.tolist()
