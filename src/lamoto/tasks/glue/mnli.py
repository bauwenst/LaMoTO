from datasets import DatasetDict

from ._general import CompareSentencesGLUETask


class MNLI(CompareSentencesGLUETask):
    """
    MNLI has one train set but two validation and test sets: one is "matched" to the kind of data found in the train set,
    the other is "mismatched" and hence harder. Papers usually test on both, but from what I've seen, model performance
    differs by less than 0.5% on these. Hence, I choose to only include the harder, mismatched splits.
    """
    def __init__(self):
        super().__init__("mnli", num_labels=3, text_field1="premise", text_field2="hypothesis")

    def loadDataset(self) -> DatasetDict:
        datasetdict = super().loadDataset()
        datasetdict["validation"] = datasetdict["validation_mismatched"]
        datasetdict["test"]       = datasetdict["test_mismatched"]
        return datasetdict
