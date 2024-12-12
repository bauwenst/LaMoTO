from typing import Protocol, Any, Dict, Type, Optional, Set, Union
from typing_extensions import Self  # https://stackoverflow.com/a/77247460/9352077
from abc import ABC, abstractmethod
from dataclasses import dataclass
from numpy import ndarray

import evaluate
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase, Trainer

from ..training.auxiliary.hyperparameters import TaskHyperparameters


@dataclass
class EvaluationEnvironment:
    model: PreTrainedModel
    tokeniser: PreTrainedTokenizerBase
    hyperparameters: TaskHyperparameters
    trainer: Trainer

    validation_dataset: Dataset
    test_dataset: Dataset
    use_test_not_validation: bool=False

    def getDatasetWithoutCollator(self):
        return self.test_dataset if self.use_test_not_validation else self.validation_dataset

    def getDatasetWithCollator(self):
        return self.trainer.get_eval_dataloader(self.getDatasetWithoutCollator())


class MetricHyperparameters:

    @classmethod
    def extractFromTask(cls, hyperparameters: TaskHyperparameters) -> Self:
        """
        Because HuggingFace's metric architecture replaces object instantiation with
        strings (it's not evaluate.F1(), it's evaluate.load("f1")), we can't call
        constructors and hence need this abomination here to pass user arguments, which are
        only known AFTER the metrics are defined, to a string-constructed class.

        In a sane world, you just instantiate Metric objects with parameters given in their
        constructors, rather than using a factory pattern like a registry.

        (The other way to link user arguments to an existing metric would be passing
        a Dict[str, PARAMCLASS] to the trainer, with no autocompletion on the string
        and the parameter class. At least when the task parameters have a dedicated field
        for each metric, you know what hyperparameters to fill in and don't need to know the metric name.)
        """
        for field_value in hyperparameters.__dict__.values():
            if isinstance(field_value, cls):
                return field_value


class Metric(Protocol):  # Protocol so that HuggingFace's evaluate.Metric is made part of the hierarchy and hence its .compute() actually counts as THE SAME method as this one, which isn't possible with a Union[LamotoMetric, HuggingfaceMetric].
    def compute(self, predictions: Any, references: Any) -> Dict[str, Any]:
        pass


class LamotoMetric(Metric, ABC):

    def __init__(self, environment: Optional[EvaluationEnvironment]):
        """
        The environment is necessary if
            (a) you are autonomous and need the model to do your own computations, or
            (b) your metric needs to be configured with hyperparameters, e.g. the beta in F_beta, the weights in BLEU, ...
        """
        self.environment = environment

    @abstractmethod
    def compute(self, predictions: Union[list,ndarray], references: Union[list,ndarray]) -> Dict[str, float]:  # LaMoTO's type signature is more predictable.
        pass

    @abstractmethod
    def isAutonomous(self) -> bool:
        pass

    @classmethod
    @abstractmethod
    def keys(cls) -> Set[str]:
        """The keys to expect in the dictionary produced by .compute()."""
        pass


class LogitLabelMetric(LamotoMetric):
    """
    Metric that uses predictions and labels, like a HuggingFace metric.
    """

    def isAutonomous(self) -> bool:
        return False


class StreamedMetric(LamotoMetric):
    """
    Metric to which data are added at runtime, so there is no point passing any predictions to it at compute time.
    """

    def compute(self, predictions: Union[list,ndarray], references: Union[list,ndarray]) -> Dict[str, float]:
        return self.computeFromMemory()

    @abstractmethod
    def computeFromMemory(self) -> Dict[str, float]:
        pass

    def isAutonomous(self) -> bool:
        return False


class AutonomousMetric(LamotoMetric):
    """
    Metric that generates its own data, rather than just using logits and labels.
    """

    def compute(self, predictions: Union[list,ndarray], references: Union[list,ndarray]) -> Dict[str, float]:
        return self.computeFromEnvironment()

    @abstractmethod
    def computeFromEnvironment(self) -> Dict[str, float]:
        pass

    def isAutonomous(self) -> bool:
        return True


class MetricRegistry:

    def __init__(self):
        self.custom_metrics: Dict[str,Type[LamotoMetric]] = dict()

    def registerMetric(self, name: str, metric: Type[LamotoMetric]):
        if name in self.custom_metrics:
            raise ValueError(f"Cannot register custom metric {name} because it already exists.")

        self.custom_metrics[name] = metric

    def load(self, name: str, environment: EvaluationEnvironment=None) -> Metric:
        if name in self.custom_metrics and environment is not None:
            return self.custom_metrics[name](environment)
        else:
            return evaluate.load(name)
