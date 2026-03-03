from typing import Optional, Union, Generic, Type
from typing_extensions import Self
from abc import ABC, abstractmethod, ABCMeta
from dataclasses import dataclass
from pathlib import Path

import warnings
from copy import deepcopy

from transformers import PreTrainedModel, PreTrainedTokenizerBase, PretrainedConfig
from datasets.arrow_dataset import DatasetInfoMixin

from tktkt.interfaces import TokeniserWithVocabulary, TokeniserFactory
from tktkt.interfaces.identifiers import _ProhibitDeclaringConstructor
from archit.instantiation.abstracts import PC, HC, BaseModel

from ...util.datasets import getDatasetSize, totalBatches
from .callbacks import CombinedCallback, EventType, CallbackAtExpInterval, CallbackAtRatchetingInterval, \
    CallbackAtTimeInterval


class _AbstractableWithoutConstructor(ABCMeta, _ProhibitDeclaringConstructor):
    pass


class _IntervalConfig(metaclass=_AbstractableWithoutConstructor):
    """
    A configuration dataclass for a certain way of deciding how much space there is between triggers of some kind.
    Also used for stopping strategies.

    All descendants of this class are a @dataclass so that they can be serialised with repr() and deserialised with eval().
    The @dataclass decorator is not heritable, so we have a metaclass to prevent normal constructors.
    """
    pass


class _FixedBatchesInterval(_IntervalConfig):
    """
    For intervals where the amount of steps between triggers is constant. A "step" is one gradient descent, or
    equivalently, one effective batch.
    """
    @abstractmethod
    def getSteps(self, batch_size: int, dataset: DatasetInfoMixin, split_name: str) -> int:
        """Returns the amount of effective batches that are processed before a trigger (expressed in batches/trigger)."""
        pass


class _CallbackInterval(_IntervalConfig):
    """
    For intervalling that is more difficult than just "every XYZ steps".
    """
    @abstractmethod
    def getCallback(self, events: Union[EventType,set[EventType]]) -> CombinedCallback:
        pass


@dataclass
class Never(_IntervalConfig):
    pass


@dataclass
class EveryNExamples(_FixedBatchesInterval):
    examples: int

    def getSteps(self, batch_size: int, dataset: DatasetInfoMixin, split_name: str) -> int:
        return totalBatches(self.examples, batch_size)


@dataclass
class EveryNExamplesOrOncePerEpoch(_FixedBatchesInterval):
    max_examples: int

    def getSteps(self, batch_size: int, dataset: DatasetInfoMixin, split_name: str) -> int:
        try:
            examples_per_epoch = getDatasetSize(dataset, split=split_name)
        except:
            examples_per_epoch = self.max_examples

        examples_per_trigger = min(examples_per_epoch, self.max_examples)
        return totalBatches(examples_per_trigger, batch_size)


@dataclass
class EveryNDescents(_FixedBatchesInterval):
    descents: int

    def getSteps(self, *args, **kwargs) -> int:
        return self.descents


@dataclass
class EveryNDescentsOrOncePerEpoch(_FixedBatchesInterval):
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
class EveryExpDescents(_CallbackInterval):
    """
    Produces triggers that are linearly spaced on a log axis. This is equivalent to using linearly spaced values as
    exponents for an exponential function. Examples:
        Start 1, spacing 1:     1, 10^1, 10^2, 10^3, ...
        Start 10, spacing 0.1: 10, 10^1.1, 10^1.2, 10^1.3, ...
    In general, start * 10^{i*spacing} for i = 0, 1, ...
    If you need a sequence that goes like 1, 2, 3, ..., 10, 20, 30, ... This is not the right class.
    """
    start: float = 10
    base: int = 10
    exp_spacing: float = 1.0

    def getCallback(self, events: Union[EventType,set[EventType]]) -> CombinedCallback:
        return CallbackAtExpInterval(start=self.start, base=self.base, spacing=self.exp_spacing, events=events)


@dataclass
class EveryRatchetingDescents(_CallbackInterval):
    """
    Starts at a given amount and ratchets up the step size after every N increases. For example:
        start 10, 9 steps: 10, 20, 30, ..., 100, 200, 300, ..., 1000, 2000, 3000, ...
        start 20, 4 steps: 20, 40, 60, 80, 100, 200, 300, 400, 500, 1000, 1500, 2000, 2500, 5000, 7500, 1000, ...
    """
    start: int = 10
    steps: int = 9

    def getCallback(self, events: Union[EventType,set[EventType]]) -> CombinedCallback:
        return CallbackAtRatchetingInterval(start=self.start, steps_between_ratchets=self.steps, events=events)


@dataclass
class EveryNEpochs(_FixedBatchesInterval):
    epochs: int

    def getSteps(self, batch_size: int, dataset: DatasetInfoMixin, split_name: str) -> int:
        try:
            examples_per_epoch = getDatasetSize(dataset, split=split_name)
            batches_per_epoch  = totalBatches(examples_per_epoch, batch_size)
            return self.epochs * batches_per_epoch
        except:
            raise RuntimeError("Could not retrieve dataset size.")


@dataclass
class NEveryEpoch(_FixedBatchesInterval):
    per_epoch: int

    def getSteps(self, batch_size: int, dataset: DatasetInfoMixin, split_name: str) -> int:  # DatasetInfoMixin is the parent class for Dataset and IterableDataset.
        examples_per_epoch = getDatasetSize(dataset, split=split_name)
        batches_per_epoch  = totalBatches(examples_per_epoch, batch_size)
        batches_per_trigger = batches_per_epoch // self.per_epoch

        if batches_per_trigger == 0:
            raise RuntimeError(f"Too many triggers per epoch ({batches_per_epoch} batches per epoch yet {self.per_epoch} triggers per epoch requested).")

        return batches_per_trigger


@dataclass
class EveryNMinutes(_CallbackInterval):
    minutes: int

    def getCallback(self, events: Union[EventType,set[EventType]]) -> CombinedCallback:
        return CallbackAtTimeInterval(minutes=self.minutes, events=events)


@dataclass
class EveryNPackedTokens(_FixedBatchesInterval):
    """
    Only works for packed datasets. Otherwise, you need to use a CombinedCallback that uses state.num_input_tokens_seen.
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
Immediately    = lambda: AfterNExamples(0)


########################################################################################################################


@dataclass
class Intervals:
    evaluation: _IntervalConfig
    checkpointing: Optional[_IntervalConfig] = None  # Checkpoints contain the model, the optimiser, the rng, ... so that you can resume training from them. HuggingFace allows checkpointing to differ from evaluation IF AND ONLY IF saving and evaluation are done every fixed number of steps AND a save step happens on every evaluation. That means save_steps % eval_steps == 0.
    backups: Optional[_IntervalConfig] = None  # Stores the weights of the model (and nothing else!) to a permanent backup, i.e. a folder that falls outside of the "checkpoint rotation" (the system that removes older checkpoints to keep a fixed amount stored).


@dataclass
class TaskHyperparameters(Generic[HC]):
    # Naming (not necessary in case a checkpoint name is given)
    save_as: Optional[str]  # Results in checkpoint names of the form "partialname+augmentation_taskname+taskaugmentation_2024-01-23_01-02-03"

    # Side-effects
    wandb_project: Optional[str]
    discard_artifacts: bool  # If true, any model and any graph of intermediate results will be removed after training. Does not apply to backups.
    discard_results: bool    # If true, the eval/test results will not be stored on disk, only returned as a Python dictionary after training.
    store_in_hf_cache: bool  # Whether to store model checkpoints in the HF_HOME cache folder, or just the CWD.

    # Sizes
    # - An "effective batch" is all the examples used to compute the gradient of one step of gradient descent.
    #   Classically, the loss function looks like sum_{i=1}^N loss(x_i, y_i). You compute that sum by splitting the effort
    #   across devices and, per device, splitting the work into several runs because of memory limitations.
    # - Training:
    examples_per_effective_batch: int
    examples_per_device_batch: int  # A devicebatch is just whatever fits on the GPU, not N.

    effective_batches_warmup: Union[int, float]  # The RoBERTa paper says, for GLUE tasks, they warm up for 6% of all batches across 10 epochs. That's in the ballpark of 100 batches.
    hard_stopping_condition: _IntervalConfig

    # - Evaluating:
    examples_per_evaluation: Optional[int]  # If None, use the entire validation set.
    evals_of_patience: Optional[int]  # Don't necessary need early stopping. You never know what's around the corner!

    track_best_checkpoint: bool
    rank_checkpoints_using_loss: bool
    eval_vs_save_intervals: Intervals  # Second one will be ignored if the above option is true.

    # Model configuration
    # - Initialising:
    seed: int
    model_config_or_checkpoint: Union[str, Path, PretrainedConfig]
    archit_basemodel_class: Type[BaseModel]
    archit_head_config: HC

    init_weights: bool  # Whether to initialise any weights at all. Doesn't apply to the cases where HuggingFace is used.
    load_hf_automodel_if_hf_checkpoint_and_matches_task: bool  # You want this to be false for doing inference, e.g. in CLM after training. When you load a checkpoint for token classification in the context of a task that classifies tokens, by default the old head weights will be reused even if that means num_labels is wrong. This is intentional, because too many task-specific checks would otherwise need to be run.
    custom_hf_class: Optional[Type[PreTrainedModel]]  # If set, will be used instead of ArchIt or AutoModel.

    # - Gradients:
    learning_rate: float
    adamw_decay_rate: float  # Not the same as L2 regularisation. That's the whole point of the AdamW paper!
    gradient_clipping_norm: Optional[float]  # The gradient norm past which the magnitude of the gradient vector is clipped.
    gradient_checkpointing_if_possible: bool

    # Tokeniser
    tokeniser: Optional[Union[PreTrainedTokenizerBase, str, TokeniserWithVocabulary, TokeniserFactory[TokeniserWithVocabulary]]]  # If not given, will use the HuggingFace tokeniser of the model checkpoint (which can't be a config then).
    add_special_tokens: bool

    def copy(self) -> Self:
        return deepcopy(self)

    def withHeadConfig(self, config: HC):
        hp = self.copy()
        hp.archit_head_config = config
        return hp

    def toDict(self) -> dict:
        """
        Convert this object to a dictionary that is safe for being stored as a JSON file.
        """
        hp_as_dict = dict(self.__dict__)
        hp_as_dict["_hp_class"]               = self.__class__.__name__
        hp_as_dict["hard_stopping_condition"] = repr(self.hard_stopping_condition)
        hp_as_dict["eval_vs_save_intervals"]  = repr(self.eval_vs_save_intervals)
        hp_as_dict["archit_basemodel_class"]  = self.archit_basemodel_class.__name__
        hp_as_dict["custom_hf_class"]         = self.custom_hf_class.__name__ if self.custom_hf_class else None
        if isinstance(self.model_config_or_checkpoint, PretrainedConfig):
            hp_as_dict["model_config_or_checkpoint"] = {
                "_config_class":  self.model_config_or_checkpoint.__class__.__name__,
                "_config_fields": self.model_config_or_checkpoint.to_dict(),
            }
        if self.tokeniser is not None and not isinstance(self.tokeniser, str):
            hp_as_dict["tokeniser"] = repr(self.tokeniser)
        return hp_as_dict

    def __setattr__(self, key, value):
        """Ensures that all field assignments are case-insensitive. (The only part that is not case-insensitive are constructor arguments.)"""
        super().__setattr__(key.lower(), value)

    def __getattr__(self, item):
        """Ensures that all field accesses are case-insensitive."""
        if item.lower() != item:
            return getattr(self, item.lower())
        else:
            raise AttributeError(item)


def hyperparametersFromDict(hp_as_dict: dict) -> TaskHyperparameters:
    """
    Inverse of TaskHyperparameters.toDict(), with the possible exception of not restoring the tokeniser.
    """
    hp_class = eval(hp_as_dict.pop("_hp_class"))
    hp = hp_class(**hp_as_dict)
    hp.hard_stopping_condition = eval(hp.hard_stopping_condition)
    hp.eval_vs_save_intervals  = eval(hp.eval_vs_save_intervals)
    hp.archit_basemodel_class  = eval(hp.archit_basemodel_class)
    hp.custom_hf_class         = eval(hp.custom_hf_class) if hp.custom_hf_class is not None else None
    if isinstance(hp.model_config_or_checkpoint, dict):
        config_class = eval(hp.model_config_or_checkpoint["_config_class"])
        config_fields = hp.model_config_or_checkpoint["_config_fields"]
        hp.model_config_or_checkpoint = config_class(**config_fields)
    if hp.tokeniser is not None:
        try:
            hp.tokeniser = eval(hp.tokeniser)
        except:
            hp.tokeniser = None
            warnings.warn(f"Tokeniser set to 'None' (so the model's default tokeniser will be used) because it could not be reconstructed from the given value:\n{hp.tokeniser}")
    return hp


from archit.instantiation.basemodels import RobertaBaseModel

SUGGESTED_HYPERPARAMETERS = TaskHyperparameters(
    save_as=None,
    wandb_project=None,
    discard_artifacts=False,
    discard_results=False,
    store_in_hf_cache=False,

    examples_per_effective_batch=32,
    examples_per_device_batch=32,
    effective_batches_warmup=100,  # The RoBERTa paper says, for GLUE tasks, they warm up for 6% of all batches across 10 epochs. That's in the ballpark of 100 batches.
    hard_stopping_condition=AfterNEpochs(epochs=10),

    examples_per_evaluation=None,
    eval_vs_save_intervals=Intervals(
        evaluation=EveryNDescents(descents=512),  # Not relative to epoch size because epochs can be insanely massive.
        checkpointing=None
    ),
    evals_of_patience=5,
    track_best_checkpoint=True,
    rank_checkpoints_using_loss=True,

    seed=69420,
    init_weights=True,
    load_hf_automodel_if_hf_checkpoint_and_matches_task=True,
    model_config_or_checkpoint="roberta-base",
    archit_basemodel_class=RobertaBaseModel,
    archit_head_config=None,
    custom_hf_class=None,

    learning_rate=2e-5,
    adamw_decay_rate=0.01,
    gradient_clipping_norm=None,
    gradient_checkpointing_if_possible=False,

    tokeniser=None,
    add_special_tokens=True
)


def getDefaultHyperparameters() -> TaskHyperparameters:
    return SUGGESTED_HYPERPARAMETERS.copy()


__all__ = ["TaskHyperparameters", "Intervals",
           "Never", "EveryNEpochs", "EveryNDescents", "EveryNDescentsOrOncePerEpoch", "NEveryEpoch", "EveryNMinutes", "EveryNPackedTokens",
           "AfterNDescents", "AfterNEpochs", "AfterNPackedTokens", "AfterNMinutes",
           "PC", "HC", "getDefaultHyperparameters"]