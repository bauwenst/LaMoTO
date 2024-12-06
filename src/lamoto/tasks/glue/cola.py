from datasets import DatasetDict

from ._general import ClassifySentenceGLUETask
from ...util.datasets import BalancingStrategy, UpsampleToBiggest, rebalanceLabels


class CoLA(ClassifySentenceGLUETask):
    """Detection of erroneously formed sentences. Note that the actual dataset has a 70-30 positive-negative skew."""
    def __init__(self, balancing: BalancingStrategy=UpsampleToBiggest()):
        super().__init__("CoLA")
        self._balancing = balancing

    def _loadDataset(self) -> DatasetDict:
        """
        Note: we don't override _loadDatasetRaw() to rebalance the labels.
        GLUE always imputes the test split after raw. The reason you don't add rebalancing in there is that duplicating
        examples from the train set and then sampling from the train set means you get data leakage from train to test.
        """
        dataset_with_test = super()._loadDataset()
        return rebalanceLabels(dataset_with_test, label_column="label", strategy=self._balancing, seed=self.hyperparameters.SEED)
