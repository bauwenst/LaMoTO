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
from typing import Iterable

import numpy as np
from datasets import Dataset
from transformers import DataCollatorForTokenClassification, AutoModelForTokenClassification
import torch

from modest.interfaces.morphologies import WordSegmentation
from modest.interfaces.datasets import ModestDataset
from modest.languages.english import English_Celex
from archit.instantiation.heads import TokenClassifierHeadConfig
from archit.instantiation.tasks import ForSingleLabelTokenClassification

from ._core import *
from ..util.datasets import replaceDatasetColumns_OneExampleToOneExample, ListOfField, ClassLabel
from ..util.visuals import log


class MBR(Task[TokenClassifierHeadConfig]):

    def __init__(self, morphologies: ModestDataset[WordSegmentation]=English_Celex(), dataset_out_of_context: bool=True, weighted_labels: bool=True):
        super().__init__(
            task_name="MBR" + "-" + morphologies.identifier(),
            text_fields=["text"],
            label_field=ListOfField(ClassLabel("labels")),
            metric_config=MetricSetup(
                to_compute=["accuracy", "precision", "recall", "f1"],
                to_track={
                    "accuracy": {"accuracy": "Acc"},
                    "precision": {"precision": "Pr"},
                    "recall": {"recall": "Re"},
                    "f1": {"f1": "$F_1$"}
                },
                to_rank=RankingMetricSpec("recall", "recall", higher_is_better=True)
            ),
            archit_class=ForSingleLabelTokenClassification,
            automodel_class=AutoModelForTokenClassification,

            num_labels=2
        )
        self._morphologies = morphologies
        self._is_single_word_dataset = dataset_out_of_context

        self._label_distribution = None
        if weighted_labels:
            _,label_distribution = self.dataset_metadata.getLabelDistribution(self.loadDataset()["train"]).popitem()
            label_distribution = {str(k): float(v) for k,v in label_distribution.items()}
            log(f"Class labels will be weighted proportional to the inverse of the following distribution:\n\t{sorted(label_distribution.items())}")
            label_distribution = [1/p for _, p in sorted(label_distribution.items())]
            label_distribution = [w/sum(label_distribution) for w in label_distribution]
            self._label_distribution = label_distribution

        # self.tokenizer: CanineTokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)  # There is no unk_token because any Unicode codepoint is mapped via a hash table to an ID (which is better than UTF-8 byte tokenisation although not reversible).

    def _datasetOutOfContext(self) -> Iterable[dict]:
        for obj in self._morphologies.generate():
            text = obj.word

            split_indices = np.cumsum([len(t) for t in obj.segment()]) - 1  # E.g.: if the token lengths are [2,3], you want [0,1,0,0,1].
            labels = np.zeros(len(text), dtype=np.int8)
            labels[split_indices] = 1

            yield {"text": text, "labels": labels}

    def _datasetInContext(self) -> Iterable[dict]:
        raise NotImplementedError("No in-context dataset exists currently.")

    def _loadDataset(self) -> DatasetDict:
        iterable = self._datasetOutOfContext() if self._is_single_word_dataset else self._datasetInContext()

        # Turn iterable into a Dataset. TODO: Not sure how sane this is. Works for small morphological datasets, but it should be as lazy as possible, rather than loading the whole dataset into memory at once.
        dataset = Dataset.from_list(list(iterable))

        # 80-10-10 split
        datasetdict_train_vs_validtest = dataset.train_test_split(train_size=80 / 100)
        datasetdict_valid_vs_test = datasetdict_train_vs_validtest["test"].train_test_split(train_size=50 / 100)
        return DatasetDict({
            "train": datasetdict_train_vs_validtest["train"],
            "validation": datasetdict_valid_vs_test["train"],
            "test": datasetdict_valid_vs_test["test"]
        })

    def _prepareDataset(self, dataset: DatasetDict) -> DatasetDict:
        def preprocess(example: dict) -> dict:
            return self.tokenizer(example["text"], add_special_tokens=False,
                                  # return_tensors="pt",  # DO NOT USE THIS OPTION, IT IS EVIL. Will basically make 1-example batches of everything even though things like the collator will expect non-batches, and hence they will think no padding is needed because all features magically have the same length of 1.
                                  truncation=True, max_length=self._getMaxInputLength())
        return replaceDatasetColumns_OneExampleToOneExample(dataset, preprocess, but_keep={"labels"})

    def adjustHyperparameters(self, hp: TaskHyperparameters[TokenClassifierHeadConfig]):
        hp.archit_head_config.n_classes = 2
        hp.archit_head_config.label_loss_weights = self._label_distribution

    def getSuggestedHeadConfig(self):
        from archit.instantiation.heads import SingleLayerHeadConfig
        return SingleLayerHeadConfig()

    def getCollator(self) -> DataCollator:
        # Get the batch generator, a.k.a. collator (https://huggingface.co/docs/transformers/main_classes/data_collator).
        # Note that unlike sequence classification, the labels per example have variable length in token classification.
        # Since you pad input IDs, you must also pad labels. This is one of the reasons you can get the very confusing
        #       ValueError: Unable to create tensor, you should probably activate truncation and/or padding with
        #       'padding=True' 'truncation=True' to have batched tensors with the same length.
        # The other reason is the too-deeply-nested tensors due to return_tensors="pt".
        return DataCollatorForTokenClassification(self.tokenizer, padding="longest", max_length=self._getMaxInputLength())

    def getPredictionsAndReferences(self, eval: EvalPrediction) -> tuple[Any,Any]:
        predictions, labels = eval.predictions.argmax(-1), eval.label_ids  # The last dimension of predictions (i.e. the logits) is the amount of classes.
        predictions, labels = predictions.flatten(), labels.flatten()  # Both are EXAMPLES x TOKENS
        mask = labels != -100  # Only select results where the label isn't padding.

        return predictions[mask].tolist(), labels[mask].tolist()

    @classmethod
    def getDefaultHyperparameters(cls) -> TaskHyperparameters[TokenClassifierHeadConfig]:
        from archit.instantiation.basemodels import CanineBaseModel
        from ..training.auxiliary.hyperparameters import NEveryEpoch, AfterNEpochs
        hp = super().getDefaultHyperparameters()
        hp.archit_base_model_class       = CanineBaseModel
        hp.model_config_or_checkpoint   = "google/canine-c"
        hp.tokeniser                    = "google/canine-c"
        hp.effective_batches_warmup     = 1000  # TODO: No idea if still relevant.
        hp.eval_vs_save_intervals.evaluation = NEveryEpoch(per_epoch=9)
        hp.evals_of_patience       = 30
        hp.hard_stopping_condition = AfterNEpochs(100)
        hp.archit_head_config      = None
        return hp
