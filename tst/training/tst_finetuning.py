from tst.preamble import *

from archit.instantiation.heads import *
from archit.instantiation.basemodels import RobertaBaseModel, DebertaBaseModel
from lamoto.tasks import *
from lamoto.training.auxiliary.hyperparameters import *
from lamoto.training.training import LamotoPaths


hp = getDefaultHyperparameters()
hp.seed = 0
hp.store_in_hf_cache = False
hp.discard_artifacts = True


def setModel(id: int=1):
    if id == 1:
        hp.model_config_or_checkpoint = "haisongzhang/roberta-tiny-cased"  # 4 layers, 512 hidden size
        hp.archit_basemodel_class = RobertaBaseModel
    elif id == 2:
        hp.model_config_or_checkpoint = "microsoft/deberta-base"
        hp.archit_basemodel_class = DebertaBaseModel
    elif id == 3:  # Will only work on my local machine.
        hp.model_config_or_checkpoint = LamotoPaths.pathToCheckpoints() / "roberta-tiny-cased_DP_2024-12-17_02-17-40" / "checkpoint-3072"
        hp.archit_basemodel_class = RobertaBaseModel
    else:
        raise RuntimeError()


def tst_pos():
    hp.archit_head_config = TokenClassificationHeadConfig()  # Filled in automatically.

    task = POS()
    task.train(hp)


def tst_ner():
    hp.archit_head_config = TokenClassificationHeadConfig()

    task = NER()
    task.train(hp)


def tst_dp():
    hp.archit_head_config = DependencyParsingHeadConfig(extended_model_config=PoolingAndStridingConfig())  # Filled in automatically.
    task = DP()
    task.train(hp)


def tst_cola():
    hp.archit_head_config = SequenceClassificationHeadConfig()
    hp.hard_stopping_condition = Never()

    task = CoLA()
    task.train(hp)


def tst_sts():
    hp.hard_stopping_condition = Never()
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
    task = QNLI()  # Suffers from https://github.com/huggingface/transformers/issues/33985
    task.train(hp)


def tst_qqp():
    hp.archit_head_config = SequenceClassificationHeadConfig()

    hp.hard_stopping_condition = AfterNDescents(descents=5)

    task = QQP()
    task.train(hp)


def tst_copa():
    hp.hard_stopping_condition = Never()
    hp.eval_vs_save_intervals = Intervals(
        evaluation=EveryNDescents(descents=256),
        checkpointing=None,
        backups=EveryNDescents(descents=512)  # Good test to see if this works.
    )
    hp.evals_of_patience = 5

    hp.archit_head_config = SequenceClassificationHeadConfig()
    task = COPA()
    task.train(hp)


def tst_record():
    hp.archit_head_config = SequenceClassificationHeadConfig()
    task = ReCoRD_Binary()
    task.train(hp)


def tst_mbr():
    from modest.languages.english import English_Celex
    task = MBR(morphologies=English_Celex())
    task.train()


def tst_squadv1():
    hp.hard_stopping_condition = Never()
    hp.archit_head_config = ExtractiveQAHeadConfig()
    task = SQuADv1()
    task.train(hp)


def tst_squadv2():
    hp.hard_stopping_condition = Never()
    hp.archit_head_config = ExtractiveAQAHeadConfig(ua_loss_weight=1.0)
    task = SQuADv2()
    task.train(hp)


def tst_wic():
    hp.hard_stopping_condition = Never()
    hp.evals_of_patience  = 10
    hp.archit_head_config = SequenceClassificationHeadConfig()
    task = WiC()
    task.train(hp)


def tst_tuner():
    from lamoto.training.tuning import TaskTuner, MetaHyperparameters, HyperparameterGrid

    tuner = TaskTuner(
        sampling_grid=HyperparameterGrid(
            None,
            None,
            [8,16,32,64],
            [1e-4, 1e-3],
            None
        )
    )
    task = DP()
    hp.archit_head_config = DependencyParsingHeadConfig(extended_model_config=PoolingAndStridingConfig())
    meta = MetaHyperparameters(
        meta_seed=0,
        n_grid_samples=5,

        max_examples_phase_1=16384,
        minmax_evals_phase_1=3,

        max_examples_phase_2=16384,
        minmax_evals_phase_2=3
    )
    tuner.tune(task, hp, meta)


if __name__ == "__main__":
    setModel(id=1)

    # tst_glue()
    # tst_qnli()
    # tst_sts()
    # tst_pos()
    # tst_ner()
    # tst_cola()
    # tst_dp()
    # tst_qqp()
    # tst_copa()
    # tst_record()
    # tst_mbr()

    tst_wic()
    # tst_squadv1()
    # tst_squadv2()

    # tst_tuner()
