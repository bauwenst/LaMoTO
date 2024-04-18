from datasets import load_dataset
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification

from ._core import *


class POS(FinetuningTask):

    def __init__(self):
        self.tagset = ["B-" + tag for tag in self.loadDataset()["train"].features["upos"].feature.names]
        super().__init__(
            task_name="pos",
            metrics=MetricSetup(
                to_compute=["seqeval"],
                to_track={
                    "seqeval": {"overall_accuracy": "Accuracy"}  # Note that Pr = Re = F1 = Acc without a negative class (BIO's O class which POS doesn't have but NER does).
                }
            ),
            automodel_class=AutoModelForTokenClassification,

            num_labels=len(self.tagset)
        )

    def loadDataset(self) -> DatasetDict:
        """
        Can use many datasets, since PoS tags sometimes come with a dataset meant for a different purpose.
        For example, you can actually reuse the entirety of the NER task and just replace "ner_tags" by "pos_tags" since
        CoNLL-2003 also has those (although they're not in BIO format).
        """
        return load_dataset("universal_dependencies", "en_ewt")

    def prepareDataset(self, dataset: DatasetDict) -> DatasetDict:
        def preprocess(example):
            enc = self.tokenizer(example["tokens"], is_split_into_words=True,
                                 add_special_tokens=ADD_SPECIAL_TOKENS, truncation=True, max_length=MAX_INPUT_LENGTH)
            word_labels  = example["upos"]  # Note: this is already a list of integers.
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
        dataset = dataset.remove_columns(["tokens", "idx", "text", "lemmas", "upos", "xpos", "feats", "head", "deprel", "deps", "misc"])
        return dataset

    def getCollator(self) -> DataCollator:
        return DataCollatorForTokenClassification(self.tokenizer, padding="longest", max_length=MAX_INPUT_LENGTH)

    def getPredictionsAndReferences(self, eval: transformers.EvalPrediction) -> Tuple[Any,Any]:
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
