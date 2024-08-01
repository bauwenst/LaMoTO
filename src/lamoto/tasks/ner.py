from datasets import load_dataset
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification

from ._core import *


class NER(Task):
    """
    Named entity recognition using the SeqEval-IOB2 metrics (which automatically exclude the "O" tag when computing F1,
    otherwise it would just be accuracy).
    """

    def __init__(self):
        self.tagset = self.loadDataset()["train"].features["ner_tags"].feature.names
        super().__init__(
            task_name="ner",
            metric_config=MetricSetup(
                to_compute=["seqeval"],
                to_track={
                    "seqeval": {"overall_precision": "Pr", "overall_recall": "Re", "overall_f1": "$F_1$"}
                }
            ),
            automodel_class=AutoModelForTokenClassification,

            num_labels=len(self.tagset)  # == 9 == len(['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'])
        )

    def loadDataset(self) -> DatasetDict:
        return load_dataset("conll2003")

    def prepareDataset(self, dataset: DatasetDict) -> DatasetDict:
        def preprocess(example):
            enc = self.tokenizer(example["tokens"], is_split_into_words=True,
                                 add_special_tokens=self.hyperparameters.ADD_SPECIAL_TOKENS, truncation=True, max_length=self._getMaxInputLength())
            word_labels  = example["ner_tags"]
            token_labels = []

            word_ids = enc.word_ids()
            for i in range(len(word_ids)):
                if word_ids[i] is not None and (i == 0 or word_ids[i] != word_ids[i-1]):  # First-only pooling is better for NER (due to capitals): https://aclanthology.org/2021.eacl-main.194.pdf
                    token_labels.append(word_labels[word_ids[i]])
                else:
                    token_labels.append(-100)

            enc["labels"] = token_labels
            return enc

        dataset = dataset.map(preprocess, batched=False)
        dataset = dataset.remove_columns(["tokens", "id", "chunk_tags", "pos_tags", "ner_tags"])
        return dataset

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
