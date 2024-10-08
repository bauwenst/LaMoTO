from typing import Set, Union
from abc import abstractmethod, ABC

from datasets.arrow_dataset import IterableDataset, Dataset
from tktkt.preparation.mappers import TextMapper

from ..tasks._core import *
from ..tasks._core import HC


HuggingfaceDataset = Union[Dataset, IterableDataset]


class DatasetAugmentation(ABC):

    @abstractmethod
    def augment(self, dataset: HuggingfaceDataset) -> HuggingfaceDataset:
        """
        Transform a given dataset split.
        """
        pass


class MappingDatasetAugmentation(DatasetAugmentation):

    @abstractmethod
    def mapDatasetExample(self, example: dict) -> dict:
        pass

    def augment(self, dataset: HuggingfaceDataset) -> HuggingfaceDataset:
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

    def augment(self, dataset: HuggingfaceDataset) -> HuggingfaceDataset:
        if isinstance(dataset, Dataset):
            return dataset.select(range(self._amount))
        elif isinstance(dataset, IterableDataset):
            return dataset.take(self._amount)
        else:
            raise TypeError(f"Unrecognised dataset type: {type(dataset)}")


########################################################################################################################


class TaskWithAugmentedDataset(Task[HC]):
    """
    Wrapper around a fine-tuning task that augments its dataset.
    """

    def __init__(self, task: Task[HC], augmentation: DatasetAugmentation, splits: Set[str]):
        super().__init__(
            task_name=task.task_name,
            metric_config=task.metric_config,
            archit_class=task.archit_class,
            automodel_class=task.automodel_class,
            **task.automodel_args
        )
        self._method_implementations = task
        self._augmentation = augmentation
        self._splits = splits

    def loadDataset(self) -> DatasetDict:
        splits = self._method_implementations.loadDataset()
        for split_name in self._splits:
            splits[split_name] = self._augmentation.augment(splits[split_name])
        return splits

    def prepareDataset(self, dataset: DatasetDict) -> DatasetDict:
        return self._method_implementations.prepareDataset(dataset)

    def getCollator(self) -> DataCollator:
        return self._method_implementations.getCollator()

    def getPredictionsAndReferences(self, eval: EvalPrediction) -> Tuple[Any,Any]:
        return self._method_implementations.getPredictionsAndReferences(eval)

    def adjustHyperparameters(self, hp: TaskHyperparameters[HC]):
        pass
