from typing import Tuple, Any, Optional, Union, List, Dict
from dataclasses import dataclass
from datasets import DatasetDict
from transformers import EvalPrediction, PreTrainedTokenizerBase, DataCollator
from transformers.tokenization_utils_base import PaddingStrategy
from transformers.data.data_collator import DataCollatorMixin

import torch
from transformers import AutoModelForMultipleChoice
from archit.instantiation.tasks import ForSingleAnswerMultipleChoice

from ._general import SuperGLUETask
from .._core import MetricSetup


class COPA(SuperGLUETask):
    """
    Multiple-choice cause/effect inference.

    The task consists of quadruplets
      (premise, question, choice1, choice2)
    where the goal is to choose either choice 1 or choice 2 as the result of applying 'question' (which is "cause" or "effect") to the premise.

    According to Table 2 of the SuperGLUE paper, it is expected that you transform the question into a natural-language question before
    passing it to the model. Scripts like https://github.com/adapter-hub/efficient-task-transfer/blob/main/itrain/datasets/super_glue.py
    store hardcoded question "What was the cause of this?" and "What happened as a result?".

    The SuperGLUE paper says about this task:
    "For COPA, MultiRC, and ReCoRD, for each answer choice, we similarly concatenate(?) the context(?) with that answer choice
     and feed the resulting sequence into BERT to produce an answer representation. For COPA, we project these representations
     into a scalar, and take as the answer the choice with the highest associated scalar."
    """

    HARDCODED_QUESTIONS = {
        "cause": "What was the cause of this?",
        "effect": "What was the result of this?"
    }

    def __init__(self):
        super().__init__(
            task_name="COPA",
            metric_config=MetricSetup(
                to_compute=["precision", "recall", "f1", "accuracy"],
                to_track={
                    "precision": {"precision": "Pr"},
                    "recall":    {"recall": "Re"},
                    "f1":        {"f1": "$F_1$"},
                    "accuracy":  {"accuracy": "Acc"}
                }
            ),
            num_labels=1
        )
        # Need to reset these classes since GLUETask always uses sequence classification as the architecture, while this task is an exception.
        self.archit_class    = ForSingleAnswerMultipleChoice
        self.automodel_class = AutoModelForMultipleChoice

    def _prepareDataset(self, dataset: DatasetDict) -> DatasetDict:
        def preprocess(example):
            return self.tokenizer(2*[example["premise"] + " " + COPA.HARDCODED_QUESTIONS[example["question"]]], [example["choice1"], example["choice2"]],
                                  add_special_tokens=self.hyperparameters.ADD_SPECIAL_TOKENS, truncation=True, max_length=self._getMaxInputLength())

        dataset = dataset.map(preprocess, batched=False)
        dataset = dataset.remove_columns(["premise", "question", "choice1", "choice2", "idx"])
        return dataset

    def getCollator(self) -> DataCollator:
        return DataCollatorForMultipleChoice(self.tokenizer, padding="longest", max_length=self._getMaxInputLength())

    def getPredictionsAndReferences(self, eval: EvalPrediction) -> Tuple[Any,Any]:
        return eval.predictions.squeeze().argmax(-1).tolist(), eval.label_ids.squeeze().tolist()


@dataclass
class DataCollatorForMultipleChoice(DataCollatorMixin):
    """
    For some reason, this only exists in the HuggingFace docs.
    https://github.com/huggingface/transformers/issues/34671
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, examples: List[Dict[str, Any]]):
        # Take labels out of the examples beforehand, because they aren't nested.
        label_name = "label" if "label" in examples[0].keys() else "labels"
        labels = [example.pop(label_name) for example in examples]

        batch_size  = len(examples)
        num_choices = len(examples[0]["input_ids"])

        # Go from 2 examples of 2 choices [{input_ids: [[1], [2]]}, {input_ids: [[3], [4]]}]
        # to 4 examples [{input_ids: [1]}, {input_ids: [2]}] + [{input_ids: [3]}, {input_ids: [4]}]
        flat_examples = sum(
            ([{k: v[i] for k, v in example.items()} for i in range(num_choices)] for example in examples),
            start=[]
        )

        # Pad all choices of all examples as if you're padding examples.
        batch = self.tokenizer.pad(
            flat_examples,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Reshape from B*C x L into B x C x L, and add the labels back in.
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch
