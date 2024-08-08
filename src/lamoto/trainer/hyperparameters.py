from typing import Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from transformers import PreTrainedModel, PreTrainedTokenizerBase, PretrainedConfig
from datasets import Dataset
from datasets.arrow_dataset import DatasetInfoMixin

from ..util.datasets import getDatasetSize, totalBatches


class HowManySteps(ABC):
    """
    A "step" is one gradient descent, or equivalently, one effective batch.
    You can use this as a time axis in the training process.
    """
    @abstractmethod
    def getSteps(self, *args) -> int:
        pass


class LamotoIntervalStrategy(HowManySteps):
    pass


class NeverInterval(LamotoIntervalStrategy):
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


class LamotoStoppingStrategy(HowManySteps):
    pass


class NeverStop(LamotoStoppingStrategy):
    def getSteps(self, *args):
        raise NotImplementedError("No maximum amount of steps defined when the rule is to never stop.")


class AfterNDescents(LamotoStoppingStrategy):
    def __init__(self, descents: int):
        self.steps_to_end = descents

    def getSteps(self, *args):
        return self.steps_to_end


class AfterNEpochs(LamotoStoppingStrategy):
    def __init__(self, epochs: int, effective_batch_size: int):
        self.epochs_to_end = epochs
        self.examples_per_batch = effective_batch_size

    def getSteps(self, train_dataset: DatasetInfoMixin) -> int:
        examples_per_epoch = getDatasetSize(train_dataset, split="train")
        steps_per_epoch    = totalBatches(examples_per_epoch, self.examples_per_batch)
        return steps_per_epoch*self.epochs_to_end


class AfterNTokens(LamotoStoppingStrategy):
    def __init__(self, total_tokens: int, tokens_per_packed_example: int, effective_batch_size: int):  # NOTE: This only works for packed datasets. Otherwise, you need to use a TrainerCallback that uses state.num_input_tokens_seen
        self.total_tokens = total_tokens
        self.tokens_per_example = tokens_per_packed_example
        self.examples_per_batch = effective_batch_size

    def getSteps(self, *args) -> int:
        total_examples = totalBatches(self.total_tokens, self.tokens_per_example)
        total_steps    = totalBatches(total_examples, self.examples_per_batch)
        return total_steps


class AfterNMinutes(LamotoStoppingStrategy):
    def __init__(self, minutes: int):
        self.minutes = minutes

    def getSteps(self, *args):
        raise NotImplementedError("Time-based stopping isn't enforced with a step argument, but a custom callback.")


@dataclass
class Intervals:
    evaluation: LamotoIntervalStrategy
    checkpointing: Optional[LamotoIntervalStrategy] = None  # Some tasks, you just want to checkpoint per eval. Sometimes there's too much space between evals though, and you don't want to lose progress.


@dataclass
class TaskHyperparameters:
    SAVE_AS: Optional[str]  # Not necessary if a checkpoint name is given.
    WANDB_PROJECT: Optional[str]

    # Sizes
    # - An "effective batch" is all the examples used to compute the gradient of one step of gradient descent.
    #   Classically, the loss function looks like sum_{i=1}^N loss(x_i, y_i). You compute that sum by splitting the effort
    #   across devices and, per device, splitting the work into several runs because of memory limitations.
    # - Training:
    EXAMPLES_PER_EFFECTIVE_BATCH: int
    EXAMPLES_PER_DEVICEBATCH: int  # A devicebatch is just whatever fits on the GPU, not N.

    EFFECTIVE_BATCHES_WARMUP: Union[int, float]  # The RoBERTa paper says, for GLUE tasks, they warm up for 6% of all batches across 10 epochs. That's in the ballpark of 100 batches.
    HARD_STOPPING_CONDITION: LamotoStoppingStrategy

    # - Evaluating:
    EXAMPLES_PER_EVALUATION: Optional[int]  # If None, use the entire validation set.
    EVALS_OF_PATIENCE: Optional[int]  # Don't necessary need early stopping. You never know what's around the corner!

    TRACK_BEST_MODEL: bool
    EVAL_VS_SAVE_INTERVALS: Intervals  # Second one will be ignored if the above option is true.

    # Model configuration
    # - Initialising:
    INIT_WEIGHTS: bool
    CHECKPOINT_OR_CONFIG: Union[str, PretrainedConfig]
    TOKENISER_CHECKPOINT: Optional[str]  # Must be given if the above checkpoint isn't.

    # - Gradients:
    LEARNING_RATE: float
    L2_REGULARISATION: float

    # Tokeniser
    ADD_SPECIAL_TOKENS: bool


@dataclass
class EvaluationEnvironment:
    model: PreTrainedModel
    tokeniser: PreTrainedTokenizerBase
    validation_dataset: Dataset
    hyperparameters: TaskHyperparameters


__all__ = ["TaskHyperparameters", "Intervals", "EvaluationEnvironment",
           "NeverInterval", "EveryNDescents", "NEveryEpoch", "EveryNMinutes",
           "NeverStop", "AfterNDescents", "AfterNEpochs", "AfterNTokens", "AfterNMinutes"]