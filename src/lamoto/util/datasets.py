import time
import warnings
from copy import deepcopy
from typing import Iterable, Union, TypeVar, Generator, List
from transformers import PreTrainedTokenizerBase
from abc import ABC

import datasets
from datasets import Dataset, IterableDataset
from datasets.arrow_dataset import DatasetInfoMixin

from tktkt.util.printing import pluralise, ordinal

from .schedules import Schedule

HuggingfaceDataset = Union[Dataset, IterableDataset]


def getDatasetSize(dataset: DatasetInfoMixin, split: str="train"):  # DatasetInfoMixin is the parent class for Dataset and IterableDataset.
    """
    Get the amount of examples in a HuggingFace dataset, whether it is a regular Dataset or a streamed IterableDataset.
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
        raise TypeError("")

    return dataset


# Timeout configuration (I tested this and it works, but it sure is a weird use of Python imports... https://github.com/huggingface/datasets/issues/6172#issuecomment-1794876229)
datasets.config.STREAMING_READ_RETRY_INTERVAL = 60   # Seconds between retries; ideally this would work with exponential backoff, but it doesn't, because... HuggingFace engineers.
datasets.config.STREAMING_READ_MAX_RETRIES    = 120  # Retry for up to 2 hours.


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
        if not new_ids[-1] == tokenizer.eos_token_id:  # You need an EOS between examples.
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


def transferDatasetMetadata(source: HuggingfaceDataset, target: HuggingfaceDataset):
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
    if isinstance(dataset, HuggingfaceDataset):  # Set the DatasetInfoMixin fields.
        transferDatasetMetadata(dataset, iterable_dataset)

    return iterable_dataset


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


T = TypeVar("T")
class LossyBackoff(Iterable[T], ABC):
    """
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
