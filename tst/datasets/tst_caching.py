from transformers import AutoTokenizer
from lamoto.training.auxiliary.hyperparameters import getDefaultHyperparameters
from lamoto.tasks import POS


def tst_caching():
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    task = POS()
    task._setTokenizer(tokenizer)
    task._setHyperparameters(getDefaultHyperparameters())
    print(task.prepareDataset(task.loadDataset()))
    print(task.prepareDataset(task.loadDataset()))


if __name__ == "__main__":
    tst_caching()
