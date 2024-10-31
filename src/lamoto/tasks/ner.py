from datasets import load_dataset
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification

from archit.instantiation.heads import TokenClassificationHeadConfig
from archit.instantiation.tasks import ForSingleLabelTokenClassification


from ._core import *
from ..preprocessing.wordlevel import FlattenWordLabels, LabelPooling


class NER(Task[TokenClassificationHeadConfig]):
    """
    Named entity recognition using the SeqEval-IOB2 metrics (which automatically exclude the "O" tag when computing F1,
    otherwise it would just be accuracy).
    """

    def __init__(self):
        self.tagset = self._loadDataset()["train"].features["ner_tags"].feature.names
        super().__init__(
            task_name="NER",
            metric_config=MetricSetup(
                to_compute=["seqeval"],
                to_track={
                    "seqeval": {"overall_precision": "Pr", "overall_recall": "Re", "overall_f1": "$F_1$"}
                }
            ),
            archit_class=ForSingleLabelTokenClassification,
            automodel_class=AutoModelForTokenClassification,

            num_labels=len(self.tagset)  # == 9 == len(['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'])
        )

    def _loadDataset(self) -> DatasetDict:
        return load_dataset("eriktks/conll2003", trust_remote_code=True)

    def _prepareDataset(self, dataset: DatasetDict) -> DatasetDict:
        tokenise_truncate_flatten = FlattenWordLabels(tokenizer=self.tokenizer,
                                                      max_tokens=self._getMaxInputLength(),
                                                      add_specials=self.hyperparameters.ADD_SPECIAL_TOKENS,
                                                      pooling_mode=LabelPooling.FIRST)  # Only learn from the first token, which is easiest for NER (due to capitals): https://aclanthology.org/2021.eacl-main.194.pdf
        def preprocess(example):
            tokens, labels = tokenise_truncate_flatten.preprocess(example["tokens"], {"ner_tags": example["ner_tags"]})
            return {
                "input_ids": tokens,
                "attention_mask": [1]*len(tokens),
                "labels": labels["ner_tags"]
            }

        dataset = dataset.map(preprocess, batched=False)
        dataset = dataset.remove_columns(["tokens", "id", "chunk_tags", "pos_tags", "ner_tags"])
        return dataset

    def adjustHyperparameters(self, hp: TaskHyperparameters[TokenClassificationHeadConfig]):
        hp.archit_head_config.num_labels = len(self.tagset)

    def getCollator(self) -> DataCollator:
        return DataCollatorForTokenClassification(self.tokenizer, padding="longest", max_length=self._getMaxInputLength())

    def getPredictionsAndReferences(self, eval: EvalPrediction) -> Tuple[Any,Any]:
        # print(eval.predictions.shape, eval.predictions[-1,:,:])
        # print(eval.label_ids.shape,   eval.label_ids[-1,:])
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
