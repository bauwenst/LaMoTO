from tst.preamble import *

from archit.instantiation.heads import *
from lamoto.tasks import *
from lamoto.tasks._core import getDefaultHyperparameters

hp = getDefaultHyperparameters()
hp.SEED = 0


def test_pos():
    hp.HEAD_CONFIG = TokenClassificationHeadConfig()  # Filled in automatically.

    task = POS()
    task.train(hp)


def test_ner():
    hp.HEAD_CONFIG = TokenClassificationHeadConfig()

    task = NER()
    task.train(hp)


def test_dp():
    hp.HEAD_CONFIG = DependencyParsingHeadConfig(extended_model_config=BaseModelExtendedConfig())  # Filled in automatically.
    task = DP()
    task.train(hp)


def test_cola():
    hp.HEAD_CONFIG = SequenceClassificationHeadConfig()

    task = CoLA()
    task.train(hp)


if __name__ == "__main__":
    test_pos()
    # test_ner()
    # test_cola()
    # test_dp()
