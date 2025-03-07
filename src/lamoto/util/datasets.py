from typing import Iterable, Union, TypeVar, Generator, List, Optional, Dict, Any, Callable, Generic, Tuple, Type
from abc import ABC, abstractmethod

from transformers import PreTrainedTokenizerBase

import time
import warnings
from copy import deepcopy
import numpy as np
import numpy.random as npr
from collections import Counter
import datasets
from datasets import Dataset, IterableDataset, DatasetDict, IterableDatasetDict
from datasets.arrow_dataset import DatasetInfoMixin

from tktkt.util.iterables import mapExtend
from tktkt.util.printing import pluralise, ordinal
from tktkt.util.environment import IS_NOT_LINUX

from .schedules import Schedule
from .exceptions import ImpossibleBranchError


N_THREADS_DATASET_MAP = 1 if IS_NOT_LINUX else 6

HuggingfaceExample     = Dict[str,Any]
HuggingfaceBatch       = Dict[str,List[Any]]

HuggingfaceDatasetSplit = Union[Dataset, IterableDataset]
HuggingfaceDatasetDict  = Union[DatasetDict, IterableDatasetDict]
HuggingfaceDataset      = Union[HuggingfaceDatasetSplit, HuggingfaceDatasetDict]

Dataset_or_DatasetDict = TypeVar("Dataset_or_DatasetDict", bound=Union[Dataset,DatasetDict])  # Non-iterable.
T = TypeVar("T")
T2 = TypeVar("T2")

class DictOfLists:
    """
    Utility class that helps to turn dataset of the format
        [{}, {}, {}]
    into
        {[], [], []}
    """

    def __init__(self, keys: Iterable[str], append_none_for_missing_keys: bool=True):
        self._dict: Dict[str,List[Any]] = {key: [] for key in keys}
        self._append_none = append_none_for_missing_keys

    def append(self, items: Dict[str,Any]):
        for k,v in items.items():
            self._dict[k].append(v)  # Will error for keys unknown at construction.

        if self._append_none:
            missing_keys = set(self._dict.keys()) - set(items.keys())
            for k in missing_keys:
                self._dict[k].append(None)

    def toDict(self) -> Dict[str,List[Any]]:
        return self._dict


def replaceDatasetColumns_OneExampleToOneExample(dataset: Dataset_or_DatasetDict, mapping: Callable[[HuggingfaceExample],HuggingfaceExample], but_keep: Iterable[str]=None) -> Dataset_or_DatasetDict:
    old_columns = getDatasetColumnNames(dataset)
    if isinstance(dataset, (IterableDataset, IterableDatasetDict)):
        dataset = dataset.map(mapping, batched=False)
    else:
        dataset = dataset.map(mapping, batched=False, num_proc=N_THREADS_DATASET_MAP)
    return dataset.remove_columns([c for c in set(old_columns) - set(but_keep or [])])  # Note: Since .map doesn't remove columns, all the old columns will still be part of the new columns. We assume .map did not overwrite any columns, and otherwise, that the overwritten columns are declared by the caller.


def replaceDatasetColumns_ManyExamplesToManyExamples(dataset: Dataset_or_DatasetDict, mapping: Callable[[HuggingfaceBatch],HuggingfaceBatch], but_keep: Iterable[str]=None) -> Dataset_or_DatasetDict:
    old_columns = getDatasetColumnNames(dataset)
    if isinstance(dataset, (IterableDataset, IterableDatasetDict)):
        dataset = dataset.map(mapping, batched=True)
    else:
        dataset = dataset.map(mapping, batched=True, num_proc=N_THREADS_DATASET_MAP)
    return dataset.remove_columns([c for c in set(old_columns) - set(but_keep or [])])  # Note: Since .map doesn't remove columns, all the old columns will still be part of the new columns. We assume .map did not overwrite any columns, and otherwise, that the overwritten columns are declared by the caller.


def getDatasetColumnNames(dataset: HuggingfaceDataset) -> List[str]:
    if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
        first_split_name = list(dataset.keys())[0]
        return dataset.column_names[first_split_name]
    else:
        return dataset.column_names


def getDatasetSize(dataset: HuggingfaceDatasetSplit, split: str= "train") -> int:
    """
    Get the amount of examples in a HuggingFace dataset, whether it is a regular Dataset or a streamed IterableDataset.
    TODO: I wonder if .num_rows works even better.
    """
    if isinstance(dataset, dict):
        try:
            dataset = dataset[split]
        except:
            raise TypeError("Expected a dataset but a dictionary was given (DatasetDict or IterableDatasetDict) was given.")

    try:
        return len(dataset)
    except:
        try:  # Should work for both Dataset and IterableDataset.
            return dataset.info.splits[split].num_examples
        except:
            raise ValueError(f"Could not resolve size of dataset split '{split}'.")  # This is e.g. the case for the SlimPajama dataset.


def getRowCount(dataset: Dataset_or_DatasetDict) -> Union[int, Dict[str,int]]:
    if isinstance(dataset, DatasetDict):
        return {split: getRowCount(dataset[split]) for split in dataset}
    return dataset.num_rows


def sortSplits(splits: Iterable[str]) -> List[str]:
    def splitSortKey(split: str):
        if "tr" in split:
            return 0
        elif "va" in split:
            return 1
        elif "te" in split:
            return 2
        else:
            return 3

    return sorted(splits, key=splitSortKey)


def totalBatches(total_examples: int, batch_size: int):
    return 1 + (total_examples-1)//batch_size  # For example: if you have batch size 8 and 15 examples, you have 2 batches.


def shuffleAndTruncate(dataset: DatasetInfoMixin, truncate_to: int=None, seed: int=None):
    """
    French software engineering: .select() for one, .take() for the other.
    """
    if isinstance(dataset, Dataset):
        dataset = dataset.shuffle(seed=seed)
        if truncate_to:
            dataset = dataset.select(range(truncate_to))
    elif isinstance(dataset, IterableDataset):
        dataset = dataset.shuffle(seed=seed, buffer_size=10_000)  # https://huggingface.co/docs/datasets/stream#shuffle
        if truncate_to:
            dataset = dataset.take(truncate_to)
    else:
        raise TypeError(f"Cannot shuffle a {type(dataset).__name__}.")

    return dataset


def imputeTestSplit(datasetdict: DatasetDict, column_for_stratification: Optional[str], seed: int) -> DatasetDict:
    """
    Some tasks have a test set where all labels are -1 so that the only way to verify performance is to send
    model predictions to a privately owned server. For experiments, we want a proper test set. So, we take a
    sample from the train set that is of the size of the validation set, and use that as the test set.
    """
    new_datasetdict = datasetdict["train"].train_test_split(
        test_size=len(datasetdict["validation"]) / len(datasetdict["train"]),
        stratify_by_column=column_for_stratification,
        seed=seed
    )
    new_datasetdict["validation"] = datasetdict["validation"]
    return new_datasetdict


class BalancingStrategy(ABC):
    """
    Given for each label value the amount of examples in a dataset with that label value,
    compute the target amounts for each label value according to some balancing strategy.
    """
    @abstractmethod
    def getTargetCounts(self, label_counts: List[int]) -> List[int]:
        pass


class NoBalancing(BalancingStrategy):
    def getTargetCounts(self, label_counts: List[int]) -> List[int]:
        return label_counts


class UpsampleToBiggest(BalancingStrategy):
    def __init__(self, max_repeats: float=1_000_000_000_000):
        self._max_repeats = max_repeats

    def getTargetCounts(self, label_counts: List[int]) -> List[int]:
        target = max(label_counts)
        return [min(target, int(count*self._max_repeats)) for count in label_counts]


class DownsampleToSmallest(BalancingStrategy):
    def __init__(self, min_fraction: float=0.0):
        self._min_fraction = min_fraction

    def getTargetCounts(self, label_counts: List[int]) -> List[int]:
        target = min(label_counts)
        return [max(target, int(count*self._min_fraction)) for count in label_counts]


class BalanceToMedian(BalancingStrategy):
    """Balance every label to the median of all the class labels, which will include some upsampling and some downsampling."""
    def getTargetCounts(self, label_counts: List[int]) -> List[int]:
        return [int(np.median(label_counts))]*len(label_counts)


def rebalanceLabels(dataset: Dataset_or_DatasetDict, label_column: str, strategy: BalancingStrategy, seed: int) -> Dataset_or_DatasetDict:
    """
    Per split, do the following:
      1. Iterate over it once, separating indices by their label.
      2. According to the balancing strategy, get the shortage/excess of each class.
      3. For each class of size n with a shortage to get to N: duplicate N//n times and sample without replacement for the remaining N - N//n samples.
      4. For each class of size n with an excess to get to N: sample N examples without replacement, equivalent to shuffling and truncating.
    """
    if isinstance(strategy, NoBalancing):
        return dataset
    if isinstance(dataset, DatasetDict):
        return DatasetDict({split: rebalanceLabels(dataset[split], label_column, strategy, seed) for split in dataset})

    rng = npr.default_rng(seed)

    labels = dataset.with_format("numpy")[label_column]
    unique_label_values, label_counts = np.unique(labels, return_counts=True)
    target_counts = strategy.getTargetCounts(label_counts)

    reordered_label_indices = np.argsort(labels, kind="stable")  # In this array, all indices of class 0 come before class 1 come before class 2 ... but they're still all in one array that needs to be split.
    split_label_indices_at  = np.cumsum(label_counts)[:-1]  # cumsum ends with the total, which we don't need.

    sampled_indices = []
    for label_indices, old_count, new_count in zip(np.split(reordered_label_indices, indices_or_sections=split_label_indices_at), label_counts, target_counts):
        assert len(label_indices) == old_count

        if new_count > old_count:  # Upsample by copying the indices as many times as possible, and then sample without replacement.
            for _ in range(new_count // old_count):
                sampled_indices.append(label_indices)
            sampled_indices.append(rng.choice(label_indices, size=new_count % old_count, replace=False))
        elif new_count < old_count:  # Downsample without replacement.
            sampled_indices.append(rng.choice(label_indices, size=new_count, replace=False))
        else:  # Keep indices unchanged.
            sampled_indices.append(label_indices)

    return dataset.select(np.concatenate(sampled_indices))


def applyPerSplit(dataset_dict: Union[DatasetDict, Dict[str,Dataset]], function: Callable[[Dataset], T]) -> Dict[str,T]:
    """Take a function meant for an individual dataset and apply it to every split in a DatasetDict."""
    return {name: function(split) for name,split in dataset_dict.items()}


class FieldType(ABC, Generic[T]):
    """Describes a piece of data that can be extracted from a dataset example with result of type T."""
    @abstractmethod
    def extract(self, example: dict) -> T:
        pass

    def involvesType(self, cls: "Type[FieldType[Any]]") -> bool:
        """Use this when you want to know if a field is 'actually' e.g. text, even when it is wrapped as a list or tagged as a foreign field."""
        return isinstance(self, cls)


class _ExplicitField(FieldType[T]):
    """Field that can be looked up as a column in a dataset."""
    def __init__(self, field: str):
        self.field_name = field

    def extract(self, example: dict) -> T:
        return example[self.field_name]

class TextField(_ExplicitField[str]):  # TODO: Should rename to TextColumn.
    pass

class ClassLabel(_ExplicitField[int]):
    pass

class RegressiveLabel(_ExplicitField[float]):
    pass

class UnboundedIntegerLabel(_ExplicitField[int]):
    pass


class ListOfField(FieldType[List[T2]]):
    """Field that is a list of a certain type. The constructor asks for a field type that pretends that the list is
       replaced by a single element (it knows how to extract the list, but its type annotation describes the element type).
       For example, a column that is a list of tokens is a ListOfField(TextField("tokens"))."""
    def __init__(self, field: FieldType[T2]):
        self.field_itself = field

    def extract(self, example: dict) -> T2:
        return self.field_itself.extract(example)

    def involvesType(self, cls) -> bool:
        return super().involvesType(cls) or self.field_itself.involvesType(cls)


class ForeignField(FieldType[T]):
    """Field that somehow refers to another field. E.g.: in extractive QA, you have 'context' field which is just text,
       and then you could have a field that refers to the starting character of the answer IN that context and/or the full answer string itself.
       This means that you should never modify these fields, except you should always modify them when their referent is modified."""
    def __init__(self, field: FieldType[T], referent: FieldType):
        self.field_itself = field
        self.referent     = referent

    def extract(self, example: dict) -> T:
        return self.field_itself.extract(example)

    def involvesType(self, cls) -> bool:
        return super().involvesType(cls) or self.field_itself.involvesType(cls)


class SubstringLabel(ForeignField[str]):
    """E.g.: answer text in SQuAD."""
    def __init__(self, field: Union[str,FieldType[str]], referent: str):
        super().__init__(TextField(field) if isinstance(field, str) else field, TextField(referent))

class _IndexLabel(ForeignField[int]):
    def __init__(self, field: Union[str,FieldType[int]], referent: str):
        super().__init__(UnboundedIntegerLabel(field) if isinstance(field, str) else field, TextField(referent))

class CharacterIndex(_IndexLabel):
    pass

class WordIndex(_IndexLabel):
    pass

class SpanLabel(FieldType[Tuple[int,int]]):
    def __init__(self, label_start: _IndexLabel, label_end: _IndexLabel):
        assert type(label_start) == type(label_end)
        # assert label_start.target.field == label_end.target.field

        self.start = label_start
        self.end   = label_end

    def extract(self, example: dict):
        return self.start.extract(example), self.end.extract(example)

    def involvesType(self, cls) -> bool:
        return super().involvesType(cls) or self.start.involvesType(cls) or self.end.involvesType(cls)


class ImplicitLabel(FieldType[T]):
    """E.g.: answerability in SQuAD v2. It's a binary label you get by counting the amount of answers in the example."""
    def __init__(self, from_example: Callable[[dict],T]):
        self.extraction_function = from_example

    def extract(self, example: dict) -> T:
        return self.extraction_function(example)

class NestedLabel(ImplicitLabel[T]):
    """E.g.: span start index in SQuAD v1."""
    def __init__(self, field: str, nested_field_type: FieldType[T]):
        super().__init__(lambda ex: nested_field_type.extract(ex[field]))
        self._nested = nested_field_type

    def involvesType(self, cls) -> bool:
        return super().involvesType(cls) or self._nested.involvesType(cls)


class DatasetMetadata:

    def __init__(self, text_fields: List[FieldType], label_fields: List[FieldType]):
        self.text_fields  = text_fields
        self.label_fields = label_fields

    def getLabelCounts(self, dataset: Dataset) -> Dict[str, Dict[Any, int]]:
        field_to_valuecounts = dict()
        for label_field in self.label_fields:
            if isinstance(label_field, ListOfField):
                unique_label_values, label_counts = np.unique(np.fromiter(mapExtend(label_field.extract, dataset), dtype=int), return_counts=True)
            else:
                unique_label_values, label_counts = np.unique(np.fromiter(      map(label_field.extract, dataset), dtype=int), return_counts=True)  # dtype is int because if you apply this to any other label, it's probably just wrong.

            field_to_valuecounts[label_field] = dict(zip(unique_label_values, label_counts))

        return field_to_valuecounts

    def getLabelDistribution(self, dataset: Dataset) -> Dict[str, Dict[Any, float]]:
        field_to_valuefractions = dict()
        for field, value_counts in self.getLabelCounts(dataset).items():
            total = sum(value_counts.values())
            field_to_valuefractions[field] = {v: c/total for v,c in value_counts.items()}

        return field_to_valuefractions


def packedDatasetGenerator(dataset: Iterable[dict], tokenizer: PreTrainedTokenizerBase, context_length: int, key='text'):
    """Concatenate ("pack") samples from a dataset into tokenized chunks of `context_length`.

    Used for efficient training of causal models without padding. No special measures are taken
    to disallow a sequence attending to a previous sequence. The model is left to learn the
    unrelatedness of sequences from the presence of the start- and end-of-sequence-tokens
    between the samples, following a similar convention from GPT-3 and T5.
    See https://github.com/huggingface/transformers/issues/17726 for a feature request for
    HuggingFace Transformers.

    The incomplete final chunk at the end of the dataset is discarded.

    Source: Edwin Rijgersberg's script to train GEITje (https://github.com/Rijgersberg/GEITje)

    :param dataset: Dataset of samples (iterable of dict-like, e.g. Hugging Face dataset)
    :param tokenizer: Callable that tokenizes the samples (e.g. Hugging Face tokenizer)
    :param context_length: number of tokens in packed sequences
    :param key: key of the text field in the sample. Defaults to 'text'
    :yield: dicts of packed input_ids, attention_masks and (self-supervised) labels
    """
    cache = []
    for row in dataset:  # NOTE: This can be affected by a network error if the given iterable is streamed. Wrap it in a IterableDatasetWithSkippingBackoff if HF doesn't fix their retries. https://github.com/huggingface/datasets/pull/6844
        # Add extra IDs to cache
        new_ids = tokenizer(row[key], max_length=1_000_000, truncation=True, add_special_tokens=True)['input_ids']  # max_length=None will give a warning because it assumes tokeniser output is passed to the model without further processing.
        if not new_ids[-1] == tokenizer.eos_token_id:  # You need an EOS between examples. TODO: Not good if your tokeniser uses [SEP]. Same problem as WiC has.
            new_ids.append(tokenizer.eos_token_id)
        cache.extend(new_ids)

        # If the cache is now bigger than the fixed length we want to output: empty it in chunks of that size until
        # the cache needs more data.
        while len(cache) >= context_length:
            chunk = cache[:context_length]
            yield {"input_ids": chunk,
                   # "labels": chunk,  # For CLM, the labels ARE the inputs. (The collator generates these though.)
                   "attention_mask": [1] * context_length}  # If multi-class attention masks are supported, you need a mechanism for that here.
            cache = cache[context_length:]


def transferDatasetMetadata(source: HuggingfaceDatasetSplit, target: HuggingfaceDatasetSplit):
    info = source._info.copy()
    info.features = None  # Otherwise the target will ignore the generated dictionaries and instead give {"text": None} for all examples.
    target._info = info
    target._split = source._split


def PackedDataset(dataset: Iterable[dict], tokenizer: PreTrainedTokenizerBase, context_length: int) -> IterableDataset:
    iterable_dataset = IterableDataset.from_generator(
        generator=packedDatasetGenerator,
        gen_kwargs={"dataset": dataset,
                    "tokenizer": tokenizer,
                    "context_length": context_length}
    )
    if isinstance(dataset, (Dataset,IterableDataset)):  # Set the DatasetInfoMixin fields.
        transferDatasetMetadata(dataset, iterable_dataset)

    return iterable_dataset


########################################################################################################################

# Timeout configuration (I tested this and it works, but it sure is a weird use of Python imports... https://github.com/huggingface/datasets/issues/6172#issuecomment-1794876229)
datasets.config.STREAMING_READ_RETRY_INTERVAL = 60   # Seconds between retries; ideally this would work with exponential backoff, but it doesn't, because... HuggingFace engineers.
datasets.config.STREAMING_READ_MAX_RETRIES    = 120  # Retry for up to 2 hours.

def IterableDatasetWithSkippingBackoff(dataset: IterableDataset, backoff_minutes: Schedule) -> IterableDataset:
    """
    HuggingFace datasets has a two-tiered back-off system and neither of them works to protect against hub outages
    parce que les incompétents ont fait du caca. https://github.com/huggingface/datasets/issues/6843
        - huggingface_hub.utils provides the low-level http_backoff function which is used for all HTTP requests.
          It retries first with 1 second delay, then 2, then 4, then 8, then 8, and then it crashes.
        - datasets.utils.file_utils provides a function _add_retries_to_file_obj_read_method that monkey-patches the
          read method of an HfFileSystemFile to have constant-time backoff on certain exceptions. The amount of retries
          and constant delay are both customisable, EXCEPT IT DOESN'T CATCH THE RELEVANT EXCEPTION so NOTHING IS RETRIED.

    When HuggingFace goes down, a chain of three exceptions is raised:
        - TimeoutError in the ssl package
        - => ReadTimeoutError in the urllib3 package
        - => ReadTimeout in the requests package, which is not caught by the retry mechanism.

    Because `datasets` is heavy on passing around functions (in fact, they use monkey patching to even just add the
    retry mechanism, see datasets.utils.file_utils._add_retries_to_file_obj_read_method) and because the IterableDataset
    gets its data many __iter__ calls deep, you can't losslessly add backoff. The only way to do this would be to add it
    UNDERNEATH the deepest call to next(), because when next() is called and it fails, you can't retry it. The iterator
    has advanced. Best you can do is forget about that example and hope the next one doesn't fail.
    """
    safe_iterable = IterableDataset.from_generator(
        generator=LossyBackoff(iterable=dataset, minutes_between_tries=backoff_minutes).__iter__,  # <-- Call this to get a thing that supports next().
        gen_kwargs=dict()
    )
    transferDatasetMetadata(dataset, safe_iterable)
    return safe_iterable


class LossyBackoff(Iterable[T], ABC):
    """
    FIXME: Unfortunately, this is just not how Python works. When a generator raise an error, you can't step back into it.
           You can only smooth over exceptions if you can catch them INSIDE the loop that yields, not externally, making
           you entirely dependent on HuggingFace.
           Best you can do to salvage this class (but not for usage in LaMoTO) is to turn it into a None smoother, rather
           than an exception smoother, because exception can't be smoothed over externally while Nones can.

    Wraps a given iterable, and when the next() raises an exception during iteration, a back-off is applied rather than
    raising the exception immediately.

    Results that should have been generated by the failed next() calls are lost.

    The schedule should provide a good trade-off between the following two considerations:
        - Time between tries: shorter means less computing time is lost, but has a higher chance of encountering the same
                              outage and hence losing more data.
        - Time to end: longer means more likelihood of overcoming the outage, but if a crash is held out for too long,
                       the compute service may forcibly terminate the run in which case nothing is saved unlike in the case of a crash.
    """

    def __init__(self, iterable: Iterable[T], minutes_between_tries: Schedule):
        self._iterable = iterable
        self._schedule = minutes_between_tries

    def __iter__(self) -> Generator[T,None,None]:
        iterator = self._iterable.__iter__()
        schedule = deepcopy(self._schedule)  # Just like the iterable can produce many active iterators at the same time, the schedule can by copying it.

        schedule.reset()
        while True:
            successful_iteration = True

            try:
                yield next(iterator)
            except StopIteration:  # We don't catch GeneratorExit because that should be converted to StopIteration (I think).
                break
            except Exception as e:
                if schedule.isDone():  # This is not the 'while' condition because the iterator is allowed to have depleted the entire schedule, as long as it has no further errors.
                    warnings.warn(f"Backoff limit exceeded (waited {pluralise(schedule.sum(), 'minute')} across {pluralise(schedule.count(), 'try', plural='tries')}, thus {schedule.count()+1} consecutive iterations have failed). Final exception will be raised.")
                    raise e

                successful_iteration = False
                minutes_to_wait, tries = schedule.next()

                warnings.warn(f"Lost an iterator iteration ({ordinal(tries+1)} consecutive fail). Exception:")
                print(e)
                warnings.warn(f"Waiting {minutes_to_wait} minutes...")
                time.sleep(60*minutes_to_wait)

            if successful_iteration:  # This is not immediately under the 'yield' because we don't want bugs in schedule.reset() to be caught and cause backoff.
                schedule.reset()
