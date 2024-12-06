from lamoto.tasks._core import Task
from lamoto.tasks import BoolQ
from lamoto.training.auxiliary.hyperparameters import getDefaultHyperparameters
from lamoto.util.datasets import *


def tst_strategies(task: Task, label_column: str="label"):
    task._setHyperparameters(getDefaultHyperparameters())

    dataset = task.loadDataset()
    print(dataset)
    print("Before:")
    print("\t", getLabelCounts(dataset, label_column))

    for strategy in [UpsampleToBiggest(), DownsampleToSmallest(), BalanceToMedian()]:
        print(strategy.__class__.__name__)
        print("\t", getLabelCounts(rebalanceLabels(dataset, label_column=label_column, strategy=strategy, seed=0), label_column))


if __name__ == "__main__":
    tst_strategies(BoolQ())
