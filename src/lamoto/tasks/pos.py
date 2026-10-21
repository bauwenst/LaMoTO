from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification

from archit.instantiation.heads import TokenClassificationHeadConfig
from archit.instantiation.tasks import ForSingleLabelTokenClassification

from ._core import *
from ..preprocessing.ud import FilterAndCorrectUDtypes, UniversalDependenciesDataset
from ..preprocessing.wordlevel import FlattenWordLabels, LabelPooling
from ..util.visuals import log
from ..util.datasets import replaceDatasetColumns_OneExampleToOneExample, TextField, ListOfField, ClassLabel


class POS(Task[TokenClassificationHeadConfig]):

    def __init__(self):
        self._id_to_tag = self._generateTagset()
        self._tag_to_id = {tag: i for i, tag in enumerate(self._id_to_tag)}
        self._id_to_tag_seqeval = ["B-" + tag for tag in self._id_to_tag]  # We add the B- because seqeval (in getPredictionsAndReferences) doesn't work with tags that don't start with B- and

        super().__init__(
            task_name="POS",
            text_fields=[ListOfField(TextField("tokens"))],
            label_field=ListOfField(ClassLabel("upos")),
            metric_config=MetricSetup(
                to_compute=["seqeval"],
                to_track={
                    "seqeval": {"overall_accuracy": "Accuracy"}  # Note that Pr = Re = F1 = Acc without a negative class (BIO's O class which POS doesn't have but NER does).
                },
                to_rank=RankingMetricSpec("seqeval", "overall_accuracy", True)
            ),
            archit_class=ForSingleLabelTokenClassification,
            automodel_class=AutoModelForTokenClassification,

            num_labels=len(self._id_to_tag)
        )

    def _generateTagset(self) -> list[str]:  # It used to be that the HF UD dataset exposed a feature where you could just get all the possible class labels. Sadly no longer the case.
        # Taken from https://universaldependencies.org/u/pos/index.html
        return [
            "ADJ",    # adjective
            "ADP",    # adposition
            "ADV",    # adverb
            "AUX",    # auxiliary
            "CCONJ",  # coordinating conjunction
            "DET",    # determiner
            "INTJ",   # interjection
            "NOUN",   # noun
            "NUM",    # numeral
            "PART",   # particle
            "PRON",   # pronoun
            "PROPN",  # proper noun
            "PUNCT",  # punctuation
            "SCONJ",  # subordinating conjunction
            "SYM",    # symbol
            "VERB",   # verb
            "X",      # other
        ]
        # Empirical:
        # # log("Generating PoS tagset...")
        # counter = Counter()
        # dataset = self._loadDataset()
        # for split in dataset:
        #     for example in dataset[split]:
        #         counter.update(label_sequence["upos"])
        # return sorted(counter.keys(), key=counter.get, reverse=True)

    def _loadDataset(self) -> DatasetDict:
        """
        Can use many datasets, since PoS tags sometimes come with a dataset meant for a different purpose.
        For example, you can actually reuse the entirety of the NER task and just replace "ner_tags" by "pos_tags" since
        CoNLL-2003 also has those (although they're not in BIO format).
        """
        return UniversalDependenciesDataset("en_ewt").load()

    def _prepareDataset(self, dataset: DatasetDict) -> DatasetDict:
        sanitiser = FilterAndCorrectUDtypes(pos_tag_mapping=self._tag_to_id)
        flattener = FlattenWordLabels(tokenizer=self.tokenizer,
                                      max_tokens=self._getMaxInputLength(),
                                      add_specials=self.hyperparameters.add_special_tokens,
                                      pooling_mode=LabelPooling.LAST)

        def datasetMap(example: dict) -> dict:
            words, _, _, pos = sanitiser.preprocess(words=example["tokens"], heads=example["head"], pos_tags=example["upos"])
            input_ids, labels = flattener.preprocess(words, {"pos": pos})
            return {
                "input_ids": input_ids,
                "labels": labels["pos"],
                "attention_mask": [1]*len(input_ids)
            }

        return replaceDatasetColumns_OneExampleToOneExample(dataset, datasetMap)

    def adjustHyperparameters(self, hp: TaskHyperparameters[TokenClassificationHeadConfig]):
        hp.archit_head_config.num_labels = len(self._id_to_tag)

    def getCollator(self) -> DataCollator:
        return DataCollatorForTokenClassification(self.tokenizer, padding="longest", max_length=self._getMaxInputLength())

    def getPredictionsAndReferences(self, eval: EvalPrediction) -> tuple[Any,Any]:
        predictions, labels = eval.predictions.argmax(-1), eval.label_ids  # The last dimension of predictions (i.e. the logits) is the amount of classes.
        assert predictions.shape == labels.shape

        all_predictions = []
        all_labels      = []
        for example_id in range(labels.shape[0]):
            all_predictions.append([])
            all_labels.append([])
            for position_id in range(labels.shape[1]):
                if labels[example_id, position_id] != -100:
                    all_predictions[-1].append(self._id_to_tag[predictions[example_id, position_id]])  # seqeval requires string labels rather than ints. It's weird like that.
                    all_labels[-1].append(self._id_to_tag[labels[example_id, position_id]])

        return all_predictions, all_labels
