"""
FIXME:
    - The dataset size is always taken from the train split. That should also be changed. getSteps() should probably have three arguments: a dataset, a split, and a batch size.
"""
from typing import Optional, Union, Generic, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass

from copy import deepcopy
from transformers import PreTrainedModel, PreTrainedTokenizerBase, PretrainedConfig
from datasets import Dataset
from datasets.arrow_dataset import DatasetInfoMixin

from tktkt.interfaces.tokeniser import TokeniserWithFiniteTypeDomain
from archit.instantiation.abstracts import PC, HC, BaseModel

from ...util.datasets import getDatasetSize, totalBatches


class BatchesPerTriggerStrategy(ABC):
    """
    A "step" is one gradient descent, or equivalently, one effective batch.
    You can use this as a time axis in the training process. Used for both intervalling and stopping strategies.
    """
    @abstractmethod
    def getSteps(self, batch_size: int, dataset: DatasetInfoMixin, split_name: str) -> int:
        """Returns the amount of effective batches that are processed before a trigger (expressed in batches/trigger)."""
        pass


class Never(BatchesPerTriggerStrategy):

    def getSteps(self, *args, **kwargs):
        raise NotImplementedError("No strategy for getting this interval.")


class EveryNExamples(BatchesPerTriggerStrategy):

    def __init__(self, examples: int):
        self._examples_per_trigger = examples

    def getSteps(self, batch_size: int, dataset: DatasetInfoMixin, split_name: str) -> int:
        return totalBatches(self._examples_per_trigger, batch_size)


class EveryNExamplesOrOncePerEpoch(BatchesPerTriggerStrategy):

    def __init__(self, examples: int):
        self._examples_per_trigger = examples

    def getSteps(self, batch_size: int, dataset: DatasetInfoMixin, split_name: str) -> int:
        try:
            examples_per_epoch = getDatasetSize(dataset, split=split_name)
        except:
            examples_per_epoch = self._examples_per_trigger

        examples_per_trigger = min(examples_per_epoch, self._examples_per_trigger)
        return totalBatches(examples_per_trigger, batch_size)


class EveryNDescents(BatchesPerTriggerStrategy):

    def __init__(self, descents: int):
        self._batches_per_trigger = descents

    def getSteps(self, *args, **kwargs) -> int:
        return self._batches_per_trigger


class EveryNDescentsOrOncePerEpoch(BatchesPerTriggerStrategy):
    """
    Same as EveryNDescents except if epochs are smaller than N, you evaluate once per epoch.
    """
    def __init__(self, descents: int):
        self._max_batches_per_trigger = descents

    def getSteps(self, batch_size: int, dataset: DatasetInfoMixin, split_name: str) -> int:
        try:
            examples_per_epoch = getDatasetSize(dataset, split=split_name)
            batches_per_epoch  = totalBatches(examples_per_epoch, batch_size)
        except:
            batches_per_epoch = self._max_batches_per_trigger

        return min(self._max_batches_per_trigger, batches_per_epoch)


class EveryNEpochs(BatchesPerTriggerStrategy):

    def __init__(self, epochs: int):
        self._epochs_per_trigger = epochs

    def getSteps(self, batch_size: int, dataset: DatasetInfoMixin, split_name: str) -> int:
        try:
            examples_per_epoch = getDatasetSize(dataset, split=split_name)
            batches_per_epoch  = totalBatches(examples_per_epoch, batch_size)
            return self._epochs_per_trigger * batches_per_epoch
        except:
            raise RuntimeError("Could not retrieve dataset size.")


class NEveryEpoch(BatchesPerTriggerStrategy):
    def __init__(self, per_epoch: int):
        self._triggers_per_epoch = per_epoch

    def getSteps(self, batch_size: int, dataset: DatasetInfoMixin, split_name: str) -> int:  # DatasetInfoMixin is the parent class for Dataset and IterableDataset.
        examples_per_epoch = getDatasetSize(dataset, split=split_name)
        batches_per_epoch  = totalBatches(examples_per_epoch, batch_size)
        batches_per_trigger = batches_per_epoch // self._triggers_per_epoch

        if batches_per_trigger == 0:
            raise RuntimeError(f"Too many triggers per epoch ({batches_per_epoch} batches per epoch yet {self._triggers_per_epoch} triggers per epoch requested).")

        return batches_per_trigger


class EveryNMinutes(BatchesPerTriggerStrategy):
    def __init__(self, minutes: int):
        self.minutes = minutes

    def getSteps(self, *args, **kwargs) -> int:
        raise NotImplementedError("Time-based intervals aren't enforced with step arguments, but a custom callback.")


class EveryNPackedTokens(BatchesPerTriggerStrategy):
    """
    Only works for packed datasets. Otherwise, you need to use a TrainerCallback that uses state.num_input_tokens_seen.
    """
    def __init__(self, total_tokens: int, tokens_per_packed_example: int):
        self.total_tokens = total_tokens
        self.tokens_per_example = tokens_per_packed_example

    def getSteps(self, batch_size: int, dataset: DatasetInfoMixin, split_name: str) -> int:
        total_examples = totalBatches(self.total_tokens, self.tokens_per_example)
        total_steps    = totalBatches(total_examples, batch_size)
        return total_steps


AfterNPackedTokens = EveryNPackedTokens
AfterNExamples = EveryNExamples
AfterNDescents = EveryNDescents
AfterNEpochs   = EveryNEpochs
AfterNMinutes  = EveryNMinutes


########################################################################################################################


@dataclass
class Intervals:
    evaluation: BatchesPerTriggerStrategy
    checkpointing: Optional[BatchesPerTriggerStrategy] = None  # Some tasks, you just want to checkpoint per eval. Sometimes there's too much space between evals though, and you don't want to lose progress.


@dataclass
class TaskHyperparameters(Generic[HC]):
    SAVE_AS: Optional[str]  # Not necessary if a checkpoint name is given.
    WANDB_PROJECT: Optional[str]
    traceless: bool  # Whether to discard any model and any graph of intermediate results and only the evaluation results, or to keep graphs and the usual two checkpoints.
    store_in_hf_cache: bool  # Whether to store model checkpoints in the HF_HOME cache folder, or just the CWD.

    # Sizes
    # - An "effective batch" is all the examples used to compute the gradient of one step of gradient descent.
    #   Classically, the loss function looks like sum_{i=1}^N loss(x_i, y_i). You compute that sum by splitting the effort
    #   across devices and, per device, splitting the work into several runs because of memory limitations.
    # - Training:
    EXAMPLES_PER_EFFECTIVE_BATCH: int
    EXAMPLES_PER_DEVICEBATCH: int  # A devicebatch is just whatever fits on the GPU, not N.

    EFFECTIVE_BATCHES_WARMUP: Union[int, float]  # The RoBERTa paper says, for GLUE tasks, they warm up for 6% of all batches across 10 epochs. That's in the ballpark of 100 batches.
    HARD_STOPPING_CONDITION: BatchesPerTriggerStrategy

    # - Evaluating:
    EXAMPLES_PER_EVALUATION: Optional[int]  # If None, use the entire validation set.
    EVALS_OF_PATIENCE: Optional[int]  # Don't necessary need early stopping. You never know what's around the corner!

    TRACK_BEST_MODEL: bool
    EVAL_VS_SAVE_INTERVALS: Intervals  # Second one will be ignored if the above option is true.

    # Model configuration
    # - Initialising:
    SEED: int
    MODEL_CONFIG_OR_CHECKPOINT: Union[str, PretrainedConfig]
    archit_basemodel_class: Type[BaseModel]
    archit_head_config: HC

    init_weights: bool  # Whether to initialise any weights at all. Doesn't apply to the cases where HuggingFace is used.
    load_hf_automodel_if_hf_checkpoint_and_matches_task: bool  # You want this to be false for doing inference, e.g. in CLM after training. When you load a checkpoint for token classification in the context of a task that classifies tokens, by default the old head weights will be reused even if that means num_labels is wrong. This is intentional, because too many task-specific checks would otherwise need to be run.
    custom_hf_class: Optional[Type[PreTrainedModel]]  # If set, will be used instead of ArchIt or AutoModel.

    # - Gradients:
    learning_rate: float
    adamw_decay_rate: float  # Not the same as L2 regularisation. That's the whole point of the AdamW paper!

    # Tokeniser
    TOKENISER: Optional[Union[PreTrainedTokenizerBase, TokeniserWithFiniteTypeDomain, str]]  # If not given, will use the HuggingFace tokeniser of the model checkpoint (which can't be a config then).
    ADD_SPECIAL_TOKENS: bool

    def copy(self) -> "TaskHyperparameters[HC]":
        return deepcopy(self)


@dataclass
class EvaluationEnvironment:
    model: PreTrainedModel
    tokeniser: PreTrainedTokenizerBase
    validation_dataset: Dataset
    hyperparameters: TaskHyperparameters


from archit.instantiation.basemodels import RobertaBaseModel

SUGGESTED_HYPERPARAMETERS = TaskHyperparameters(
    SAVE_AS=None,
    WANDB_PROJECT=None,
    traceless=False,
    store_in_hf_cache=False,

    EXAMPLES_PER_EFFECTIVE_BATCH=32,
    EXAMPLES_PER_DEVICEBATCH=32,
    EFFECTIVE_BATCHES_WARMUP=100,  # The RoBERTa paper says, for GLUE tasks, they warm up for 6% of all batches across 10 epochs. That's in the ballpark of 100 batches.
    HARD_STOPPING_CONDITION=AfterNEpochs(epochs=10),

    EXAMPLES_PER_EVALUATION=None,
    EVAL_VS_SAVE_INTERVALS=Intervals(
        evaluation=EveryNDescents(descents=512),  # Not relative to epoch size because epochs can be insanely massive.
        checkpointing=None
    ),
    EVALS_OF_PATIENCE=5,
    TRACK_BEST_MODEL=True,

    SEED=69420,
    init_weights=True,
    load_hf_automodel_if_hf_checkpoint_and_matches_task=True,
    MODEL_CONFIG_OR_CHECKPOINT="roberta-base",
    archit_basemodel_class=RobertaBaseModel,
    archit_head_config=None,
    custom_hf_class=None,

    learning_rate=2e-5,
    adamw_decay_rate=0.01,

    TOKENISER=None,
    ADD_SPECIAL_TOKENS=True
)


def getDefaultHyperparameters() -> TaskHyperparameters:
    return SUGGESTED_HYPERPARAMETERS.copy()


__all__ = ["TaskHyperparameters", "Intervals", "EvaluationEnvironment",
           "Never", "EveryNDescents", "EveryNDescentsOrOncePerEpoch", "NEveryEpoch", "EveryNMinutes", "EveryNPackedTokens",
           "AfterNDescents", "AfterNEpochs", "AfterNPackedTokens", "AfterNMinutes",
           "PC", "HC", "getDefaultHyperparameters"]