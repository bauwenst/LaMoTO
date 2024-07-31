"""
Morphological boundary recognition (MBR) is a character-level task that predicts whether one morpheme has ended and
another has started between each adjacent pair of characters.

This is pretty much limited to the CANINE model. According to J.H. Clark via personal correspondence:
  > Canine-C + N-grams was not released since it didn't seem to give huge quality gains despite being
    substantially more complicated (it wasn't really the "hero" model of the paper, IMO).
  > We didn't explore Canine-S too much more because we feel that the main research direction for this work
    is not engineering the best subword models we can, but rather showing the pros (and deficiencies) of current
    character-level models.
"""
# TODO: There are two ways to set up training.
#   - Out of context: you give only the word and its labels, and ask the model to predict at each character.
#   - In context: you use the pre-training corpus to get sentences that contain at least one word from the dataset.
#                 then you do the same as MLM where only a select few tokens are actually used for prediction, here
#                 the characters part of the word of interest. Much richer data. Would be loaded completely differently
#                 though (OSCAR but filtered by checking if any word is in the lemma set, and with labels constructed
#                 by padding the given labels with long sequences of -100).

# TODO: There's something to be said for having a split point before the first character and after the last character.
#       This way, the model will better learn compound boundaries.

import re
from copy import deepcopy
from typing import Iterable

from datasets import Dataset, DatasetDict
from transformers import DataCollatorForTokenClassification, AutoModelForTokenClassification

from bpe_knockout.project.config import morphologyGenerator, setupEnglish, KnockoutDataConfiguration

from ._core import *



##################################
SUGGESTED_HYPERPARAMETERS_MBR = deepcopy(SUGGESTED_HYPERPARAMETERS)
SUGGESTED_HYPERPARAMETERS_MBR.CHECKPOINT_OR_CONFIG = "google/canine-c"
SUGGESTED_HYPERPARAMETERS_MBR.TOKENISER_CHECKPOINT = "google/canine-c"
SUGGESTED_HYPERPARAMETERS_MBR.EFFECTIVE_BATCHES_WARMUP = 1000
SUGGESTED_HYPERPARAMETERS_MBR.EVAL_VS_SAVE_INTERVALS.evaluation = NEveryEpoch(per_epoch=9, effective_batch_size=SUGGESTED_HYPERPARAMETERS_MBR.EXAMPLES_PER_EFFECTIVE_BATCH)

DATASET_CONFIG = setupEnglish()
##################################


class MBR(Task):

    def __init__(self, dataset_out_of_context: bool=True):
        super().__init__(
            task_name="MBR" + "-" + DATASET_CONFIG.langTag(),
            metric_config=MetricSetup(
                to_compute=["accuracy", "precision", "recall", "f1"],
                to_track={
                    "accuracy": {"accuracy": "Acc"},
                    "precision": {"precision": "Pr"},
                    "recall": {"recall": "Re"},
                    "f1": {"f1": "$F_1$"}
                }
            ),
            automodel_class=AutoModelForTokenClassification,
            num_labels=2
        )
        # self.tokenizer: CanineTokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)  # There is no unk_token because any Unicode codepoint is mapped via a hash table to an ID (which is better than UTF-8 byte tokenisation although not reversible).

        self.single_word_dataset = dataset_out_of_context

    def _datasetOutOfContext(self) -> Iterable[dict]:
        print("> Building dataset")

        BAR = "|"
        FIND_BAR = re.compile(re.escape(BAR))

        with KnockoutDataConfiguration(DATASET_CONFIG):
            for obj in morphologyGenerator():
                splitstring = obj.morphSplit()
                split_indices = [match.start() // 2 for match in
                                 FIND_BAR.finditer(" ".join(splitstring).replace("   ", BAR))]

                text = obj.lemma()
                labels = torch.zeros(len(text), dtype=torch.int8)
                labels[split_indices] = 1
                yield {"text": text, "labels": labels}

    def _datasetInContext(self) -> Iterable[dict]:
        raise RuntimeError("No in-context dataset exists currently.")

    def loadDataset(self) -> DatasetDict:
        iterable = self._datasetOutOfContext() if self.single_word_dataset else self._datasetInContext()

        # Turn iterable into a Dataset. TODO: Not sure how sane this is. Works for small morphological datasets, but it should be as lazy as possible, rather than loading the whole dataset into memory at once.
        dataset = Dataset.from_list(list(iterable))

        # 80-10-10 split
        datasetdict_train_vs_validtest = dataset.train_test_split(train_size=80 / 100)
        datasetdict_valid_vs_test = datasetdict_train_vs_validtest["test"].train_test_split(train_size=50 / 100)
        return DatasetDict({
            "train": datasetdict_train_vs_validtest["train"],
            "valid": datasetdict_valid_vs_test["train"],
            "test": datasetdict_valid_vs_test["test"]
        })

    def prepareDataset(self, dataset: DatasetDict) -> DatasetDict:
        def preprocess(example):
            output = self.tokenizer(example["text"], add_special_tokens=False,
                                    # return_tensors="pt",  # DO NOT USE THIS OPTION, IT IS EVIL. Will basically make 1-example batches of everything even though things like the collator will expect non-batches, and hence they will think no padding is needed because all features magically have the same length of 1.
                                    truncation=True, max_length=self._getMaxInputLength())
            output["labels"] = example["labels"]
            return output

        dataset = dataset.map(preprocess, batched=False)
        dataset = dataset.remove_columns(["text"])
        return dataset

    def getCollator(self) -> DataCollator:
        # Get the batch generator, a.k.a. collator (https://huggingface.co/docs/transformers/main_classes/data_collator).
        # Note that unlike sequence classification, the labels per example have variable length in token classification.
        # Since you pad input IDs, you must also pad labels. This is one of the reasons you can get the very confusing
        #       ValueError: Unable to create tensor, you should probably activate truncation and/or padding with
        #       'padding=True' 'truncation=True' to have batched tensors with the same length.
        # The other reason is the too-deeply-nested tensors due to return_tensors="pt".
        return DataCollatorForTokenClassification(self.tokenizer, padding="longest", max_length=self._getMaxInputLength())

    def getPredictionsAndReferences(self, eval: transformers.EvalPrediction) -> Tuple[Any,Any]:
        predictions, labels = eval.predictions.argmax(-1), eval.label_ids  # The last dimension of predictions (i.e. the logits) is the amount of classes.
        predictions, labels = predictions.flatten(), labels.flatten()  # Both are EXAMPLES x TOKENS
        mask = labels != -100  # Only select results where the label isn't padding.

        return predictions[mask].tolist(), labels[mask].tolist()

    def train(self, hyperparameters: TaskHyperparameters=CANINE_DEFAULT_HYPERPARAMETERS, model_augmentation: ModelAugmentation=None):
        return super().train(hyperparameters, model_augmentation)