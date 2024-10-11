from typing import Set, TypeVar, Dict
from abc import abstractmethod, ABC

from datasets.iterable_dataset import IterableDataset, Dataset
from tktkt.preparation.mappers import TextMapper

from ..tasks._core import *
from ..tasks._core import HC, Metric, PreTrainedTokenizerBase, PretrainedConfig
from ..util.datasets import HuggingfaceDataset

DS = TypeVar("DS", bound=HuggingfaceDataset)


class DatasetAugmentation(ABC):

    @abstractmethod
    def augment(self, dataset: DS) -> DS:
        """
        Transform a given dataset split.
        """
        pass


class MappingDatasetAugmentation(DatasetAugmentation):

    @abstractmethod
    def mapDatasetExample(self, example: dict) -> dict:
        pass

    def augment(self, dataset: DS) -> DS:
        return dataset.map(self.mapDatasetExample, batched=False)


class PerturbWords(MappingDatasetAugmentation):

    def __init__(self, mapping: TextMapper, text_field_name: str):
        self._text_field_name = text_field_name
        self._mapping = mapping

    def mapDatasetExample(self, example: dict) -> dict:
        original = example[self._text_field_name]
        if isinstance(original, str):
            example[self._text_field_name] = " ".join(map(self._mapping.convert, original.split()))
        elif isinstance(original, list):
            example[self._text_field_name] = [self._mapping.convert(word) for word in original]
        else:
            raise ValueError(f"Could not process text field '{self._text_field_name}': not a list or string.")
        return example


class Truncate(DatasetAugmentation):

    def __init__(self, max_examples: int):
        self._amount = max_examples

    def augment(self, dataset: DS) -> DS:
        if isinstance(dataset, Dataset):
            return dataset.select(range(self._amount))
        elif isinstance(dataset, IterableDataset):
            return dataset.take(self._amount)
        else:
            raise TypeError(f"Unrecognised dataset type: {type(dataset)}")


########################################################################################################################


class TaskWrapper(Task[HC]):
    """
    A task which, by default, steals all the implementations from an underlying task.
    """

    def __init__(self, task: Task[HC]):
        super().__init__(
            task_name=task.task_name,
            metric_config=task.metric_config,
            archit_class=task.archit_class,
            automodel_class=task.automodel_class,
            **task.automodel_args
        )
        self._method_implementations = task

    def loadDataset(self) -> DatasetDict:
        return self._method_implementations.loadDataset()

    def prepareDataset(self, dataset: DatasetDict) -> DatasetDict:
        return self._method_implementations.prepareDataset(dataset)

    def getCollator(self) -> DataCollator:
        return self._method_implementations.getCollator()

    def getPredictionsAndReferences(self, eval: EvalPrediction) -> Tuple[Any,Any]:
        return self._method_implementations.getPredictionsAndReferences(eval)

    def adjustHyperparameters(self, hp: TaskHyperparameters[HC]):
        return self._method_implementations.adjustHyperparameters(hp)

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


class TaskWithAugmentedDataset(TaskWrapper):
    """
    Wrapper around a fine-tuning task that augments certain splits of its dataset.
    """

    def __init__(self, task: Task[HC], augmentation: DatasetAugmentation, splits: Set[str]):
        super().__init__(task)
        self._augmentation = augmentation
        self._splits = splits

    def loadDataset(self) -> DatasetDict:
        splits = self._method_implementations.loadDataset()
        for split_name in self._splits:
            splits[split_name] = self._augmentation.augment(splits[split_name])
        return splits
