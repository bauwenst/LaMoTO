from typing import Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datasets.arrow_dataset import DatasetInfoMixin
from transformers import PretrainedConfig

from lamoto.util.datasets import getDatasetSize


class LamotoIntervalStrategy(ABC):
    @abstractmethod
    def getSteps(self, *args):
        pass


class NoStrategy(LamotoIntervalStrategy):
    def getSteps(self, *args):
        raise NotImplementedError("No strategy for getting this interval.")


class EveryNDescents(LamotoIntervalStrategy):
    def __init__(self, descents: int):
        self.steps_between_events = descents

    def getSteps(self, *args) -> int:
        return self.steps_between_events


class NEveryEpoch(LamotoIntervalStrategy):
    def __init__(self, per_epoch: int, effective_batch_size: int):
        self.events_per_epoch = per_epoch
        self.examples_per_step = effective_batch_size

    def getSteps(self, train_dataset: DatasetInfoMixin) -> int:  # DatasetInfoMixin is the parent class for Dataset and IterableDataset.
        examples_per_epoch = getDatasetSize(train_dataset, split="train")
        steps_per_epoch = examples_per_epoch // self.examples_per_step
        steps_between_events = steps_per_epoch // self.events_per_epoch

        if steps_between_events == 0:
            raise RuntimeError(f"Too many triggers per epoch ({steps_per_epoch} batches per epoch yet {self.events_per_epoch} events per epoch requested).")

        return steps_between_events


class EveryNMinutes(LamotoIntervalStrategy):
    def __init__(self, minutes: int):
        self.minutes = minutes

    def getSteps(self, *args):
        raise NotImplementedError("Time-based intervals aren't enforced with step arguments, but a custom callback.")


@dataclass
class Intervals:
    evaluation: LamotoIntervalStrategy
    checkpointing: Optional[LamotoIntervalStrategy] = None  # Some tasks, you just want to checkpoint per eval. Sometimes there's too much space between evals though, and you don't want to lose progress.


@dataclass
class TaskHyperparameters:
    WANDB_PROJECT: str

    # Sizes
    # - An "effective batch" is all the examples used to compute the gradient of one gradient descent step.
    #   Classically, the loss function looks like sum_{i=1}^N loss(x_i, y_i). You compute that sum by splitting the effort
    #   across devices and, per device, splitting the work into several runs because of memory limitations.
    # - Training:
    EXAMPLES_PER_EFFECTIVE_BATCH: int
    EXAMPLES_PER_DEVICEBATCH: int  # A devicebatch is just whatever fits on the GPU, not N.

    EFFECTIVE_BATCHES_WARMUP: Union[int, float]  # The RoBERTa paper says, for GLUE tasks, they warm up for 6% of all batches across 10 epochs. That's in the ballpark of 100 batches.
    MAX_TRAINING_EPOCHS: int

    # - Evaluating:
    EXAMPLES_PER_EVALUATION: int
    EVALS_OF_PATIENCE: Optional[int]  # Don't necessary need early stopping. You never know what's around the corner!

    TRACK_BEST_MODEL: bool
    EVAL_VS_SAVE_INTERVALS: Intervals  # Second one will be ignored if the above option is true.

    # Model configuration
    # - Initialising:
    INIT_WEIGHTS: bool
    CHECKPOINT_OR_CONFIG: Union[str, PretrainedConfig]

    # - Gradients:
    LEARNING_RATE: float
    L2_REGULARISATION: float

    # Tokeniser
    ADD_SPECIAL_TOKENS: bool
