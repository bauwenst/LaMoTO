from dataclasses import dataclass

from ._general import SequenceTaskHyperparameters

from .cola import CoLA
from .sst2 import SST2
from .rte import RTE
from .mrpc import MRPC
from .qqp import QQP
from .qnli import QNLI
from .mnli import MNLI
from .wnli import WNLI
from .stsb import STSB

from ...augmenting.augment_model import ModelAugmentation


@dataclass
class HyperparametersGLUE:
    sst2: SequenceTaskHyperparameters
    cola: SequenceTaskHyperparameters

    qqp:  SequenceTaskHyperparameters
    mrpc: SequenceTaskHyperparameters
    rte:  SequenceTaskHyperparameters
    qnli: SequenceTaskHyperparameters
    mnli: SequenceTaskHyperparameters
    wnli: SequenceTaskHyperparameters

    stsb: SequenceTaskHyperparameters

    @classmethod
    def setAll(self, hp: SequenceTaskHyperparameters) -> "HyperparametersGLUE":
        return HyperparametersGLUE(
            sst2=hp.copy(),
            cola=hp.copy(),

            qqp=hp.copy(),
            mrpc=hp.copy(),
            rte=hp.copy(),
            qnli=hp.copy(),
            mnli=hp.copy(),
            wnli=hp.copy(),

            stsb=hp.copy()
        )


class GLUE:
    """
    Task-like object that runs all GLUE tasks one after the other.
    You may not want to do this and rather run each individual task, in parallel runs on your system.
    """

    def __init__(self):
        self.sst2 = SST2()
        self.cola = CoLA()

        self.qqp  = QQP()
        self.mrpc = MRPC()
        self.rte  = RTE()
        self.qnli = QNLI()
        self.mnli = MNLI()
        self.wnli = WNLI()

        self.stsb = STSB()

    def train(self, hyperparameters: HyperparametersGLUE, model_augmentation: ModelAugmentation=None):
        for task, hp in [
            (self.sst2, hyperparameters.sst2),
            (self.cola, hyperparameters.cola),
            (self.qqp,  hyperparameters.qqp),
            (self.mrpc, hyperparameters.mrpc),
            (self.rte,  hyperparameters.rte),
            (self.qnli, hyperparameters.qnli),
            (self.mnli, hyperparameters.mnli),
            (self.wnli, hyperparameters.wnli),
            (self.stsb, hyperparameters.stsb),
        ]:
            task.train(hp, model_augmentation)
