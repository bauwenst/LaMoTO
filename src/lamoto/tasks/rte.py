from datasets import load_dataset
from transformers import DataCollatorWithPadding, AutoModelForSequenceClassification

from archit.instantiation.heads import SequenceClassificationHeadConfig
from archit.instantiation.tasks import ForSingleLabelSequenceClassification

from ._core import *


class RTE(Task[SequenceClassificationHeadConfig]):

    def __init__(self):
        super().__init__(
            task_name="rte",
            metric_config=MetricSetup(
                to_compute=["precision", "recall", "f1", "accuracy"],
                to_track={
                    "precision": {"precision": "Pr"},
                    "recall": {"recall": "Re"},
                    "f1": {"f1": "$F_1$"},
                    "accuracy": {"accuracy": "Acc"}
                }
            ),
            archit_class=ForSingleLabelSequenceClassification,
            automodel_class=AutoModelForSequenceClassification,

            num_labels=2
        )

    def loadDataset(self) -> DatasetDict:
        return load_dataset("glue", "rte")

    def prepareDataset(self, dataset: DatasetDict) -> DatasetDict:
        def preprocess(example):
            return self.tokenizer(example["sentence1"], example["sentence2"], add_special_tokens=self.hyperparameters.ADD_SPECIAL_TOKENS,
                                  truncation=True, max_length=self._getMaxInputLength())

        dataset = dataset.map(preprocess, batched=False)
        dataset = dataset.remove_columns(["sentence1", "sentence2", "idx"])
        return dataset

    def adjustHyperparameters(self, hp: TaskHyperparameters[SequenceClassificationHeadConfig]):
        hp.HEAD_CONFIG.num_labels = 2

    def getCollator(self) -> DataCollator:
        return DataCollatorWithPadding(self.tokenizer, padding="longest", max_length=self._getMaxInputLength())

    def getPredictionsAndReferences(self, eval: EvalPrediction) -> Tuple[Any,Any]:
        predictions, labels = eval.predictions.argmax(-1), eval.label_ids  # The last dimension of predictions (i.e. the logits) is the amount of classes.
        return predictions.tolist(), labels.tolist()
