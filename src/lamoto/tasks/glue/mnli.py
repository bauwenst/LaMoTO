from datasets import DatasetDict, load_dataset

from ._general import CompareSentencesGLUETask, RankingMetricSpec


class MNLI(CompareSentencesGLUETask):
    """
    MNLI has one train set but two validation and test sets: one is "matched" to the kind of data found in the train set,
    the other is "mismatched" and hence harder. Papers usually test on both, but from what I've seen, model performance
    differs by less than 0.5% on these. Hence, I choose to only include the harder, mismatched splits.
    """
    def __init__(self):
        super().__init__("MNLI", rank_by=RankingMetricSpec("f1_macro", "f1", True),
                         num_labels=3, text_field1="premise", text_field2="hypothesis")

    def _loadDataset(self) -> DatasetDict:
        original_datasetdict = load_dataset("glue", self.task_name.lower())
        new_datasetdict      = original_datasetdict["train"].train_test_split(
            test_size=len(original_datasetdict["validation_mismatched"])/len(original_datasetdict["train"]),
            stratify_by_column="label",
            seed=self.hyperparameters.SEED
        )
        new_datasetdict["validation"] = original_datasetdict["validation_mismatched"]
        return new_datasetdict
