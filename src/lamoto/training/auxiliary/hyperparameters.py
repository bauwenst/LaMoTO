from typing import Optional, Union, Generic, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import warnings
from copy import deepcopy

from transformers import PreTrainedModel, PreTrainedTokenizerBase, PretrainedConfig
from datasets import Dataset
from datasets.arrow_dataset import DatasetInfoMixin

from tktkt.interfaces.tokeniser import TokeniserWithFiniteTypeDomain
from tktkt.builders.base import TokeniserBuilder
from archit.instantiation.abstracts import PC, HC, BaseModel

from ...util.datasets import getDatasetSize, totalBatches


class BatchesPerTriggerStrategy(ABC):
    """
    A "step" is one gradient descent, or equivalently, one effective batch.
    You can use this as a time axis in the training process. Used for both intervalling and stopping strategies.

    All descendants of this class are a @dataclass so that they can be serialised with repr() and deserialised with eval().
    The @dataclass decorator is not heritable.
    """
    @abstractmethod
    def getSteps(self, batch_size: int, dataset: DatasetInfoMixin, split_name: str) -> int:
        """Returns the amount of effective batches that are processed before a trigger (expressed in batches/trigger)."""
        pass


@dataclass
class Never(BatchesPerTriggerStrategy):

    def getSteps(self, *args, **kwargs):
        raise NotImplementedError("No strategy for getting this interval.")


@dataclass
class EveryNExamples(BatchesPerTriggerStrategy):
    examples: int

    def getSteps(self, batch_size: int, dataset: DatasetInfoMixin, split_name: str) -> int:
        return totalBatches(self.examples, batch_size)


@dataclass
class EveryNExamplesOrOncePerEpoch(BatchesPerTriggerStrategy):
    max_examples: int

    def getSteps(self, batch_size: int, dataset: DatasetInfoMixin, split_name: str) -> int:
        try:
            examples_per_epoch = getDatasetSize(dataset, split=split_name)
        except:
            examples_per_epoch = self.max_examples

        examples_per_trigger = min(examples_per_epoch, self.max_examples)
        return totalBatches(examples_per_trigger, batch_size)


@dataclass
class EveryNDescents(BatchesPerTriggerStrategy):
    descents: int

    def getSteps(self, *args, **kwargs) -> int:
        return self.descents


@dataclass
class EveryNDescentsOrOncePerEpoch(BatchesPerTriggerStrategy):
    """
    Same as EveryNDescents except if epochs are smaller than N, you evaluate once per epoch.
    """
    max_descents: int

    def getSteps(self, batch_size: int, dataset: DatasetInfoMixin, split_name: str) -> int:
        try:
            examples_per_epoch = getDatasetSize(dataset, split=split_name)
            batches_per_epoch  = totalBatches(examples_per_epoch, batch_size)
        except:
            batches_per_epoch = self.max_descents

        return min(self.max_descents, batches_per_epoch)


@dataclass
class EveryExpDescents(BatchesPerTriggerStrategy):
    """
    Produces triggers that are linearly spaced on a log axis. This is equivalent to using linearly spaced values as
    exponents for an exponential function. Examples:
        Start 1, spacing 1:     1, 10^1, 10^2, 10^3, ...
        Start 10, spacing 0.1: 10, 10^1.1, 10^1.2, 10^1.3, ...
    If you need a sequence that goes like 1, 2, 3, ..., 10, 20, 30, ... This is not the right class.
    """
    start: int = 10
    exp_spacing: float = 1.0

    def getSteps(self, *args, **kwargs) -> int:
        raise NotImplementedError("Log-based intervals aren't enforced with step arguments, but a custom callback.")


@dataclass
class EveryRatchetingDescents(BatchesPerTriggerStrategy):
    """
    Starts at a given amount and ratchets up the step size after every N increases. For example:
        start 10, 9 steps: 10, 20, 30, ..., 100, 200, 300, ..., 1000, 2000, 3000, ...
        start 20, 4 steps: 20, 40, 60, 80, 100, 200, 300, 400, 500, 1000, 1500, 2000, 2500, 5000, 7500, 1000, ...
    """
    start: int = 10
    steps: int = 9

    def getSteps(self, *args, **kwargs) -> int:
        raise NotImplementedError("Ratchet-based intervals aren't enforced with step arguments, but a custom callback.")


@dataclass
class EveryNEpochs(BatchesPerTriggerStrategy):
    epochs: int

    def getSteps(self, batch_size: int, dataset: DatasetInfoMixin, split_name: str) -> int:
        try:
            examples_per_epoch = getDatasetSize(dataset, split=split_name)
            batches_per_epoch  = totalBatches(examples_per_epoch, batch_size)
            return self.epochs * batches_per_epoch
        except:
            raise RuntimeError("Could not retrieve dataset size.")


@dataclass
class NEveryEpoch(BatchesPerTriggerStrategy):
    per_epoch: int

    def getSteps(self, batch_size: int, dataset: DatasetInfoMixin, split_name: str) -> int:  # DatasetInfoMixin is the parent class for Dataset and IterableDataset.
        examples_per_epoch = getDatasetSize(dataset, split=split_name)
        batches_per_epoch  = totalBatches(examples_per_epoch, batch_size)
        batches_per_trigger = batches_per_epoch // self.per_epoch

        if batches_per_trigger == 0:
            raise RuntimeError(f"Too many triggers per epoch ({batches_per_epoch} batches per epoch yet {self.per_epoch} triggers per epoch requested).")

        return batches_per_trigger


@dataclass
class EveryNMinutes(BatchesPerTriggerStrategy):
    minutes: int

    def getSteps(self, *args, **kwargs) -> int:
        raise NotImplementedError("Time-based intervals aren't enforced with step arguments, but a custom callback.")


@dataclass
class EveryNPackedTokens(BatchesPerTriggerStrategy):
    """
    Only works for packed datasets. Otherwise, you need to use a TrainerCallback that uses state.num_input_tokens_seen.
    """
    total_tokens: int
    max_context_length: int

    def getSteps(self, batch_size: int, dataset: DatasetInfoMixin, split_name: str) -> int:
        total_examples = totalBatches(self.total_tokens, self.max_context_length)
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
    MODEL_CONFIG_OR_CHECKPOINT: Union[str, Path, PretrainedConfig]
    archit_basemodel_class: Type[BaseModel]
    archit_head_config: HC

    init_weights: bool  # Whether to initialise any weights at all. Doesn't apply to the cases where HuggingFace is used.
    load_hf_automodel_if_hf_checkpoint_and_matches_task: bool  # You want this to be false for doing inference, e.g. in CLM after training. When you load a checkpoint for token classification in the context of a task that classifies tokens, by default the old head weights will be reused even if that means num_labels is wrong. This is intentional, because too many task-specific checks would otherwise need to be run.
    custom_hf_class: Optional[Type[PreTrainedModel]]  # If set, will be used instead of ArchIt or AutoModel.

    # - Gradients:
    learning_rate: float
    adamw_decay_rate: float  # Not the same as L2 regularisation. That's the whole point of the AdamW paper!

    # Tokeniser
    TOKENISER: Optional[Union[PreTrainedTokenizerBase, str, TokeniserWithFiniteTypeDomain, TokeniserBuilder[TokeniserWithFiniteTypeDomain]]]  # If not given, will use the HuggingFace tokeniser of the model checkpoint (which can't be a config then).
    ADD_SPECIAL_TOKENS: bool

    def copy(self) -> "TaskHyperparameters[HC]":
        return deepcopy(self)

    def toDict(self) -> dict:
        """
        Convert this object to a dictionary that is safe for being stored as a JSON file.
        """
        hp_as_dict = dict(self.__dict__)
        hp_as_dict["_hp_class"]               = self.__class__.__name__
        hp_as_dict["HARD_STOPPING_CONDITION"] = repr(self.HARD_STOPPING_CONDITION)
        hp_as_dict["EVAL_VS_SAVE_INTERVALS"]  = repr(self.EVAL_VS_SAVE_INTERVALS)
        hp_as_dict["archit_basemodel_class"]  = self.archit_basemodel_class.__name__
        hp_as_dict["custom_hf_class"]         = self.custom_hf_class.__name__ if self.custom_hf_class else None
        if isinstance(self.MODEL_CONFIG_OR_CHECKPOINT, PretrainedConfig):
            hp_as_dict["MODEL_CONFIG_OR_CHECKPOINT"] = {
                "_config_class":  self.MODEL_CONFIG_OR_CHECKPOINT.__class__.__name__,
                "_config_fields": self.MODEL_CONFIG_OR_CHECKPOINT.to_dict(),
            }
        if self.TOKENISER is not None and not isinstance(self.TOKENISER, str):
            hp_as_dict["TOKENISER"] = repr(self.TOKENISER)
        return hp_as_dict


def hyperparametersFromDict(hp_as_dict: dict) -> TaskHyperparameters:
    """
    Inverse of TaskHyperparameters.toDict(), with the possible exception of not restoring the tokeniser.
    """
    hp_class = eval(hp_as_dict.pop("_hp_class"))
    hp = hp_class(**hp_as_dict)
    hp.HARD_STOPPING_CONDITION = eval(hp.HARD_STOPPING_CONDITION)
    hp.EVAL_VS_SAVE_INTERVALS  = eval(hp.EVAL_VS_SAVE_INTERVALS)
    hp.archit_basemodel_class  = eval(hp.archit_basemodel_class)
    hp.custom_hf_class         = eval(hp.custom_hf_class) if hp.custom_hf_class is not None else None
    if isinstance(hp.MODEL_CONFIG_OR_CHECKPOINT, dict):
        config_class = eval(hp.MODEL_CONFIG_OR_CHECKPOINT["_config_class"])
        config_fields = hp.MODEL_CONFIG_OR_CHECKPOINT["_config_fields"]
        hp.MODEL_CONFIG_OR_CHECKPOINT = config_class(**config_fields)
    if hp.TOKENISER is not None:
        try:
            hp.TOKENISER = eval(hp.TOKENISER)
        except:
            hp.TOKENISER = None
            warnings.warn(f"Tokeniser set to 'None' (so the model's default tokeniser will be used) because it could not be reconstructed from the given value:\n{hp.TOKENISER}")
    return hp


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
           "Never", "EveryNEpochs", "EveryNDescents", "EveryNDescentsOrOncePerEpoch", "NEveryEpoch", "EveryNMinutes", "EveryNPackedTokens",
           "AfterNDescents", "AfterNEpochs", "AfterNPackedTokens", "AfterNMinutes",
           "PC", "HC", "getDefaultHyperparameters"]