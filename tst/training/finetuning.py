from tst.preamble import *

from archit.instantiation.heads import *
from lamoto.tasks import *
from lamoto.tasks._core import SUGGESTED_HYPERPARAMETERS

SUGGESTED_HYPERPARAMETERS.SEED = 0


def test_pos():
    SUGGESTED_HYPERPARAMETERS.HEAD_CONFIG = TokenClassificationHeadConfig(num_labels=0)  # Filled in automatically.

    task = POS()
    task.train(SUGGESTED_HYPERPARAMETERS)


def test_ner():
    SUGGESTED_HYPERPARAMETERS.HEAD_CONFIG = TokenClassificationHeadConfig(num_labels=0)

    task = NER()
    task.train(SUGGESTED_HYPERPARAMETERS)


def test_dp():
    SUGGESTED_HYPERPARAMETERS.HEAD_CONFIG = DependencyParsingHeadConfig(extended_model_config=BaseModelExtendedConfig())  # Filled in automatically.
    task = DP()
    task.train(SUGGESTED_HYPERPARAMETERS)


def test_cola():
    SUGGESTED_HYPERPARAMETERS.HEAD_CONFIG = SequenceClassificationHeadConfig(num_labels=0)

    task = CoLA()
    task.train(SUGGESTED_HYPERPARAMETERS)


if __name__ == "__main__":
    # test_pos()
    test_ner()
    test_cola()
    # test_dp()