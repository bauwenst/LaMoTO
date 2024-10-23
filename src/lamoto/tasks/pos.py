from datasets import load_dataset
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification

from archit.instantiation.heads import TokenClassificationHeadConfig
from archit.instantiation.tasks import ForSingleLabelTokenClassification

from ._core import *
from ..preprocessing.wordlevel import FlattenWordLabels, LabelPooling


class POS(Task[TokenClassificationHeadConfig]):

    def __init__(self):
        self.tagset = ["B-" + tag for tag in self.loadDataset()["train"].features["upos"].feature.names]
        super().__init__(
            task_name="pos",
            metric_config=MetricSetup(
                to_compute=["seqeval"],
                to_track={
                    "seqeval": {"overall_accuracy": "Accuracy"}  # Note that Pr = Re = F1 = Acc without a negative class (BIO's O class which POS doesn't have but NER does).
                }
            ),
            archit_class=ForSingleLabelTokenClassification,
            automodel_class=AutoModelForTokenClassification,

            num_labels=len(self.tagset)
        )

    def loadDataset(self) -> DatasetDict:
        """
        Can use many datasets, since PoS tags sometimes come with a dataset meant for a different purpose.
        For example, you can actually reuse the entirety of the NER task and just replace "ner_tags" by "pos_tags" since
        CoNLL-2003 also has those (although they're not in BIO format).
        """
        return load_dataset("universal-dependencies/universal_dependencies", "en_ewt", trust_remote_code=True)

    def prepareDataset(self, dataset: DatasetDict) -> DatasetDict:
        flattener = FlattenWordLabels(tokenizer=self.tokenizer,
                                      max_tokens=self._getMaxInputLength(),
                                      add_specials=self.hyperparameters.ADD_SPECIAL_TOKENS,
                                      pooling_mode=LabelPooling.LAST)

        def preprocess(example: dict):
            input_ids, labels = flattener.preprocess(example["tokens"], {"labels": example["upos"]})
            return {
                "input_ids": input_ids,
                "labels": labels["labels"],
                "attention_mask": [1]*len(input_ids)
            }

        dataset = dataset.map(preprocess, batched=False)
        dataset = dataset.remove_columns(["tokens", "idx", "text", "lemmas", "upos", "xpos", "feats", "head", "deprel", "deps", "misc"])
        return dataset

    def adjustHyperparameters(self, hp: TaskHyperparameters[TokenClassificationHeadConfig]):
        hp.archit_head_config.num_labels = len(self.tagset)

    def getCollator(self) -> DataCollator:
        return DataCollatorForTokenClassification(self.tokenizer, padding="longest", max_length=self._getMaxInputLength())

    def getPredictionsAndReferences(self, eval: EvalPrediction) -> Tuple[Any,Any]:
        predictions, labels = eval.predictions.argmax(-1), eval.label_ids  # The last dimension of predictions (i.e. the logits) is the amount of classes.
        assert predictions.shape == labels.shape

        all_predictions = []
        all_labels      = []
        for example_id in range(labels.shape[0]):
            all_predictions.append([])
            all_labels.append([])
            for position_id in range(labels.shape[1]):
                if labels[example_id, position_id] != -100:
                    all_predictions[-1].append(self.tagset[predictions[example_id, position_id]])  # seqeval requires string labels rather than ints. It's weird like that.
                    all_labels[-1].append(     self.tagset[labels[example_id, position_id]])

        return all_predictions, all_labels
