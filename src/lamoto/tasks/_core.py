"""
Core fine-tuning script for any task.
"""
from typing import Any, Dict, Type, List, Tuple, Generic
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from datasets import DatasetDict
from transformers import DataCollator, PreTrainedTokenizerBase, PretrainedConfig, EvalPrediction
from transformers.models.auto.auto_factory import _BaseAutoModelClass
from archit.instantiation.abstracts import ModelWithHead

from ..augmenting.augment_model import ModelAugmentation
from ..measuring._core import Metric
from ..training.auxiliary.hyperparameters import TaskHyperparameters, getDefaultHyperparameters, PC, HC
from ..util.visuals import warn, log


@dataclass
class RankingMetricSpec:
    """Specification of the metric used for determining the best model, if turned on in the hyperparameters."""
    metric_name: str
    result_name: str
    higher_is_better: bool

    def fullName(self) -> str:
        return self.metric_name + "_" + self.result_name if self.metric_name else self.result_name


@dataclass
class MetricSetup:
    to_compute: List[str]               # Names of all the HuggingFace evaluate metrics to load and compute in the end.
    to_track: Dict[str, Dict[str,str]]  # metric name -> result name -> formatted name, used for graphing intermediate evaluations.
    to_rank: RankingMetricSpec = None   # Which of these to measure for finding the best model with the validation set. Defaults to loss.


class Task(ABC, Generic[HC]):

    def __init__(self, task_name: str, metric_config: MetricSetup,
                 archit_class: Type[ModelWithHead[PC,HC]], automodel_class: Type[_BaseAutoModelClass], **automodel_args):
        self.task_name       = task_name
        self.metric_config   = metric_config
        self.archit_class    = archit_class
        self.automodel_class = automodel_class
        self.automodel_args  = automodel_args

        # Fields that can be used by method implementations, but are only instantiated once .train() is called, to
        # avoid loading heavy objects that would be duplicated by the super() call of a task wrapper.
        self.hyperparameters: TaskHyperparameters[HC] = None
        self.tokenizer: PreTrainedTokenizerBase = None
        self.model_config: PretrainedConfig = None
        self.metrics: Dict[str, Metric] = None

        # Caches
        self._dataset_cache_raw: DatasetDict      = None
        self._dataset_cache_prepared: DatasetDict = None

    @abstractmethod
    def _loadDataset(self) -> DatasetDict:
        pass

    @abstractmethod
    def _prepareDataset(self, dataset: DatasetDict) -> DatasetDict:
        pass

    @abstractmethod
    def getCollator(self) -> DataCollator:
        pass

    @abstractmethod
    def adjustHyperparameters(self, hp: TaskHyperparameters[HC]):
        pass

    @abstractmethod
    def getPredictionsAndReferences(self, eval: EvalPrediction) -> Tuple[Any,Any]:
        pass

    def sneakyLogitTransform(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return logits

    ####################################################################################################################

    def loadDataset(self) -> DatasetDict:
        if self._dataset_cache_raw is None:
            log("Dataset cache will be imputed.")
            self._dataset_cache_raw = self._loadDataset()
        return self._dataset_cache_raw

    def prepareDataset(self, dataset: DatasetDict) -> DatasetDict:
        if self._dataset_cache_prepared is None:
            log("Prepared dataset cache will be imputed.")
            self._dataset_cache_prepared = self._prepareDataset(dataset)
        return self._dataset_cache_prepared

    def _computeMetrics(self, eval: EvalPrediction) -> dict:
        predictions, references = self.getPredictionsAndReferences(eval)
        results = dict()
        for metric_name, metric in self.metrics.items():
            subresults = metric.compute(predictions=predictions, references=references)
            for key, value in subresults.items():
                results[metric_name + "_" + key] = value

        # Sanity checks
        if self.metric_config.to_rank.fullName() != "loss" and self.metric_config.to_rank.fullName() not in results:
            raise RuntimeError(f"The ranking metric '{self.metric_config.to_rank.metric_name}' did not compute the required result '{self.metric_config.to_rank.result_name}'. Results we did compute: {results}")
        for metric_name, result_names in self.metric_config.to_track.items():
            for result_name in result_names:
                if metric_name + "_" + result_name not in results:
                    warn(f"Metric '{metric_name}' did not compute the tracked result '{result_name}'.")

        return results  # To this dictionary, the eval loss will be added post-hoc, and all keys will be prefixed by "eval_".

    def _setHyperparameters(self, hp: TaskHyperparameters[HC]):
        self.hyperparameters = hp
    def _setModelConfig(self, mc: PretrainedConfig):
        self.model_config = mc
    def _setMetrics(self, m: Dict[str, Metric]):
        self.metrics = m
    def _setTokenizer(self, tk: PreTrainedTokenizerBase):
        self.tokenizer = tk

    def _getMaxInputLength(self) -> int:
        """
        Helper function to find the amount of tokens the model can accept at most.
        """
        # First try the tokeniser itself. For CANINE, this is the only place where you find the correct number (2048).
        try:
            n = self.tokenizer.model_max_length
            if n < 1e12:  # Due to very persistent issue where the model config is right and the tokeniser is wrong: https://github.com/huggingface/transformers/issues/14561
                return n
        except:
            pass

        # Alternatively try the model config. This name was standardised late, so it is possible that you can't find it.
        try:
            n = self.model_config.max_position_embeddings
            if n:
                return n
        except:
            if "max_position_embeddings" in self.model_config.attribute_map:  # All PretrainedConfig classes have an attribute map.
                return getattr(self.model_config, self.model_config.attribute_map["max_position_embeddings"])
            else:
                raise RuntimeError("Couldn't find maximum input length in the tokeniser nor the model config.")

    def _isHfCheckpointForThisTask(self, architecture_name: str):
        """
        HuggingFace architectures look like "[Base]For[Task]" while ArchIt architectures can in principle be anything,
        although they conventionally look like "For[Task]".
        ArchIt architectures define which HuggingFace [Task] string is equivalent for them. If an architecture hence
        contains that string (but isn't equal to it, because in that case it is not HuggingFace and hence must be ArchIt)
        it comes from HuggingFace and is tailored to this task.
        """
        return self.archit_class.head_class.hfEquivalentSuffix() in architecture_name and \
               self.archit_class.head_class.hfEquivalentSuffix() != architecture_name

    def train(self, hyperparameters: TaskHyperparameters[HC]=getDefaultHyperparameters(), model_augmentation: ModelAugmentation=None, resume_from_folder: Path=None) -> Dict[str, float]:
        from ..training.core import TaskTrainer  # Import happens here to prevent circular importing.
        return TaskTrainer().train(
            task=self, hyperparameters=hyperparameters, model_augmentation=model_augmentation, resume_from_folder=resume_from_folder
        )


class TaskWrapper(Task[HC]):
    """
    A task which, by default, steals all the implementations from an underlying task.
    """

    def __init__(self, task: Task[HC], wrapper_name: str):
        super().__init__(
            task_name=task.task_name + "+" + wrapper_name,
            metric_config=task.metric_config,
            archit_class=task.archit_class,
            automodel_class=task.automodel_class,
            **task.automodel_args
        )
        self._method_implementations: Task[HC] = task

    def _loadDataset(self) -> DatasetDict:
        return self._method_implementations._loadDataset()

    def _prepareDataset(self, dataset: DatasetDict) -> DatasetDict:
        return self._method_implementations._prepareDataset(dataset)

    def getCollator(self) -> DataCollator:
        return self._method_implementations.getCollator()

    def adjustHyperparameters(self, hp: TaskHyperparameters[HC]):
        return self._method_implementations.adjustHyperparameters(hp)

    def getPredictionsAndReferences(self, eval: EvalPrediction) -> Tuple[Any,Any]:
        return self._method_implementations.getPredictionsAndReferences(eval)

    def sneakyLogitTransform(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self._method_implementations.sneakyLogitTransform(logits, labels)

    # Finally, four methods to communicate the runtime fields with the underlying task, so it can use them in its implementations:

    def _setHyperparameters(self, hp: TaskHyperparameters[HC]):
        super()._setHyperparameters(hp)
        self._method_implementations._setHyperparameters(hp)

    def _setMetrics(self, m: Dict[str, Metric]):
        super()._setMetrics(m)
        self._method_implementations._setMetrics(m)

    def _setModelConfig(self, mc: PretrainedConfig):
        super()._setModelConfig(mc)
        self._method_implementations._setModelConfig(mc)

    def _setTokenizer(self, tk: PreTrainedTokenizerBase):
        super()._setTokenizer(tk)
        self._method_implementations._setTokenizer(tk)


__all__ = ["Task", "MetricSetup", "RankingMetricSpec", "TaskHyperparameters", "getDefaultHyperparameters", "TaskWrapper",
           "DatasetDict", "DataCollator", "Any", "Tuple", "Path", "ModelAugmentation", "EvalPrediction"]