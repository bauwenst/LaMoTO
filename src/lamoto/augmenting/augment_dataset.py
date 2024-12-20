from typing import Set, TypeVar, Optional
from abc import abstractmethod, ABC

from datasets.iterable_dataset import IterableDataset, Dataset
from tktkt.interfaces.preparation import TextMapper
from tktkt.util.printing import roundHuman
from tktkt.preparation.perturbers import ConstantSampler, ParallelPerturber, Substitute, Insert, Pop

from ..tasks._core import *
from ..tasks._core import HC
from ..util.datasets import HuggingfaceDatasetSplit, sortSplits

DS = TypeVar("DS", bound=HuggingfaceDatasetSplit)


class DatasetAugmentation(ABC):

    @abstractmethod
    def augment(self, dataset: DS) -> DS:
        """
        Transform a given dataset split.
        """
        pass

    @abstractmethod
    def getName(self) -> str:
        pass


class Truncate(DatasetAugmentation):
    """
    Truncates a dataset to a given amount of examples.
    Optionally shuffles the data beforehand to end up with a more diverse set of examples; shuffling is not a separate
    dataset augmentation because the trainer already shuffles whatever data is kept after all augmentations are done.
    """

    def __init__(self, max_examples: int, shuffle_seed: Optional[int]=None):
        self._amount = max_examples
        self._shuffle_seed = shuffle_seed  # For IterableDatasets, this is not a fully global shuffle.

    def augment(self, dataset: DS) -> DS:
        if self._shuffle_seed is not None:
            dataset = self._shuffle(dataset)

        if isinstance(dataset, Dataset):
            return dataset.select(range(self._amount))  # Will error if self._amount > len(dataset)
        elif isinstance(dataset, IterableDataset):
            return dataset.take(self._amount)
        else:
            raise TypeError(f"Unrecognised dataset type: {type(dataset)}")

    def _shuffle(self, dataset: DS) -> DS:
        if isinstance(dataset, Dataset):
            return dataset.shuffle(seed=self._shuffle_seed)
        elif isinstance(dataset, IterableDataset):
            return dataset.shuffle(seed=self._shuffle_seed, buffer_size=100_000)
        else:
            raise TypeError(f"Unrecognised dataset type: {type(dataset)}")

    def getName(self) -> str:
        return f"trunc{roundHuman(self._amount)}"


class MappingDatasetAugmentation(DatasetAugmentation):

    @abstractmethod
    def mapDatasetExample(self, example: dict) -> dict:
        pass

    def augment(self, dataset: DS) -> DS:
        return dataset.map(self.mapDatasetExample, batched=False)


class MapWords(MappingDatasetAugmentation):
    """
    Map the words (i.e. space-separated strings) in a column of the dataset to a different string using a TkTkT mapper.
    """

    def __init__(self, mapping: TextMapper, text_field_name: str, mapping_name: str="mapwords"):
        self._text_field_name = text_field_name
        self._mapping = mapping
        self._mapping_name = mapping_name

    def mapDatasetExample(self, example: dict) -> dict:
        original = example[self._text_field_name]
        if isinstance(original, str):
            example[self._text_field_name] = " ".join(map(self._mapping.convert, original.split()))
        elif isinstance(original, list):
            example[self._text_field_name] = [self._mapping.convert(word) for word in original]
        else:
            raise ValueError(f"Could not process text field '{self._text_field_name}': not a list or string.")
        return example

    def getName(self) -> str:
        return self._mapping_name


class TyposLevenshtein1(MapWords):
    def __init__(self, text_field_name: str, p: float):
        sampler = ConstantSampler(n=1)
        super().__init__(
            text_field_name=text_field_name,
            mapping=ParallelPerturber(
                p=p,
                perturbations=[
                    Substitute(0, sampler=sampler),
                    Insert(0, sampler=sampler),
                    Pop(0, sampler=sampler)
                ]
            ),
            mapping_name="typosLD1"
        )


########################################################################################################################


class TaskWithAugmentedDataset(TaskWrapper):
    """
    Wrapper around a fine-tuning task that augments certain splits of its dataset.
    """

    def __init__(self, task: Task[HC], augmentation: DatasetAugmentation, splits: Set[str]):
        super().__init__(task, augmentation.getName() + "(" + ",".join(sortSplits(splits)) + ")")
        self._augmentation = augmentation
        self._splits = splits

    def _loadDataset(self) -> DatasetDict:
        splits = self._method_implementations._loadDataset()
        for split_name in self._splits:
            splits[split_name] = self._augmentation.augment(splits[split_name])
        return splits


class TaskWithTypos(TaskWithAugmentedDataset):

    def __init__(self, task: Task, text_fields: Set[str], splits: Set[str], p: float):
        text_fields = set(text_fields)
        if text_fields:
            field = text_fields.pop()
            if text_fields:
                super().__init__(TaskWithTypos(task, text_fields, splits=splits, p=p),
                                 augmentation=TyposLevenshtein1(field, p),
                                 splits=splits)
            else:
                super().__init__(task,
                                 augmentation=TyposLevenshtein1(field, p),
                                 splits=splits)
        else:
            raise ValueError("At least one text field is required to apply typos to.")
