from tst.preamble import *

from archit.instantiation.heads import *
from lamoto.tasks import *
from lamoto.tasks._core import getDefaultHyperparameters

hp = getDefaultHyperparameters()
hp.SEED = 0
hp.MODEL_CONFIG_OR_CHECKPOINT = "haisongzhang/roberta-tiny-cased"  # 4 layers, 512 hidden size


def tst_pos():
    hp.archit_head_config = TokenClassificationHeadConfig()  # Filled in automatically.

    task = POS()
    task.train(hp)


def tst_ner():
    hp.archit_head_config = TokenClassificationHeadConfig()

    task = NER()
    task.train(hp)


def tst_dp():
    hp.archit_head_config = DependencyParsingHeadConfig(extended_model_config=BaseModelExtendedConfig())  # Filled in automatically.
    task = DP()
    task.train(hp)


def tst_cola():
    hp.archit_head_config = SequenceClassificationHeadConfig()

    task = CoLA()
    task.train(hp)


def tst_sts():
    hp.archit_head_config = SequenceClassificationHeadConfig()

    task = STSB()
    task.train(hp)


def tst_glue():
    hp.archit_head_config = SequenceClassificationHeadConfig()
    glue_hp = HyperparametersGLUE.setAll(hp)
    task = GLUE()
    task.train(glue_hp)


def tst_qnli():
    hp.archit_head_config = SequenceClassificationHeadConfig()
    task = QNLI()  # https://github.com/huggingface/transformers/issues/33985
    task.train(hp)


if __name__ == "__main__":
    tst_glue()
    # tst_qnli()
    # tst_sts()
    # tst_pos()
    # tst_ner()
    # tst_cola()
    # tst_dp()
