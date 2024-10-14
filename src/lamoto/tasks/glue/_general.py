from datasets import load_dataset
from transformers import DataCollatorWithPadding, AutoModelForSequenceClassification

from archit.instantiation.heads import SequenceClassificationHeadConfig
from archit.instantiation.tasks import ForSingleLabelSequenceClassification

from lamoto.tasks._core import *


SequenceTaskHyperparameters = TaskHyperparameters[SequenceClassificationHeadConfig]

class GLUETask(Task[SequenceClassificationHeadConfig]):
    """
    Since all GLUE tasks are sequence tasks, they share a bunch of their code.
    """

    def __init__(self, task_name: str, metric_config: MetricSetup, num_labels: int):
        super().__init__(
            task_name=task_name,
            metric_config=metric_config,
            archit_class=ForSingleLabelSequenceClassification,
            automodel_class=AutoModelForSequenceClassification,

            num_labels=num_labels
        )
        self._num_labels = num_labels

    def loadDataset(self) -> DatasetDict:
        return load_dataset("glue", self.task_name)

    def adjustHyperparameters(self, hp: SequenceTaskHyperparameters):
        hp.archit_head_config.num_labels = self._num_labels

    def getCollator(self) -> DataCollator:
        return DataCollatorWithPadding(self.tokenizer, padding="longest", max_length=self._getMaxInputLength())

    def getPredictionsAndReferences(self, eval: EvalPrediction) -> Tuple[Any,Any]:
        predictions, labels = eval.predictions.argmax(-1), eval.label_ids  # The last dimension of predictions (i.e. the logits) is the amount of classes.
        return predictions.tolist(), labels.tolist()


class CompareSentencesGLUETask(GLUETask):
    """
    For all the NLI and similarity tasks in GLUE.
    """

    def __init__(self, task_name: str, num_labels: int, text_field1: str="sentence1", text_field2: str="sentence2"):
        super().__init__(
            task_name=task_name,
            metric_config=MetricSetup(
                to_compute=["precision", "recall", "f1", "accuracy"],
                to_track={
                    "precision": {"precision": "Pr"},
                    "recall":    {"recall": "Re"},
                    "f1":        {"f1": "$F_1$"},
                    "accuracy":  {"accuracy": "Acc"}
                }
            ) if num_labels == 2 else MetricSetup(
                to_compute=["precision_macro", "recall_macro", "f1_macro", "accuracy"],
                to_track={
                    "precision_macro": {"precision": "Macro Pr"},
                    "recall_macro": {"recall": "Macro Re"},
                    "f1_macro": {"f1": "Macro $F_1$"},
                    "accuracy": {"accuracy": "Acc"}
                }
            ),
            num_labels=num_labels
        )
        self._num_labels = num_labels
        self._field1 = text_field1
        self._field2 = text_field2

    def prepareDataset(self, dataset: DatasetDict) -> DatasetDict:
        def preprocess(example):
            return self.tokenizer(example[self._field1], example[self._field2], add_special_tokens=self.hyperparameters.ADD_SPECIAL_TOKENS,
                                  truncation=True, max_length=self._getMaxInputLength())

        dataset = dataset.map(preprocess, batched=False)
        dataset = dataset.remove_columns([self._field1, self._field2, "idx"])
        return dataset


class ClassifySentenceGLUETask(GLUETask):
    """
    For the binary single-sentence classification tasks in GLUE.
    """

    def __init__(self, task_name: str):
        super().__init__(
            task_name=task_name,
            metric_config=MetricSetup(
                to_compute=["precision", "recall", "f1", "accuracy", "matthews_correlation"],
                to_track={
                    "precision": {"precision": "Pr"},
                    "recall":    {"recall": "Re"},
                    "f1":        {"f1": "$F_1$"},
                    "accuracy":  {"accuracy": "Acc"},
                    "matthews_correlation": {"matthews_correlation": "MCC"}
                }
            ),
            num_labels=2
        )

    def prepareDataset(self, dataset: DatasetDict) -> DatasetDict:
        def preprocess(example):
            return self.tokenizer(example["sentence"], add_special_tokens=self.hyperparameters.ADD_SPECIAL_TOKENS,  # return_tensors="pt",  # DO NOT USE THIS OPTION, IT IS EVIL. Will basically make 1-example batches of everything even though things like the collator will expect non-batches, and hence they will think no padding is needed because all features magically have the same length of 1.
                                  truncation=True, max_length=self._getMaxInputLength())

        dataset = dataset.map(preprocess, batched=False)
        dataset = dataset.remove_columns(["sentence", "idx"])
        return dataset
