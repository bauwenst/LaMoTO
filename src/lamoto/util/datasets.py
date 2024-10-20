from typing import Iterable, Union
from transformers import PreTrainedTokenizerBase

import datasets
from datasets import Dataset, IterableDataset
from datasets.arrow_dataset import DatasetInfoMixin


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
    for row in dataset:  # TODO: <--- This can be affected by a network error. If HF doesn't fix their retries, wrap this. https://github.com/huggingface/datasets/pull/6844
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


def PackedDataset(dataset: Iterable[str], tokenizer: PreTrainedTokenizerBase, context_length: int) -> IterableDataset:
    iterable_dataset = IterableDataset.from_generator(
        generator=packedDatasetGenerator,
        gen_kwargs={"dataset": dataset,
                    "tokenizer": tokenizer,
                    "context_length": context_length}
    )
    if isinstance(dataset, HuggingfaceDataset):  # Set the DatasetInfoMixin fields.
        info = dataset._info.copy()
        info.features = None  # Otherwise the IterableDataset will ignore the generated dictionaries and instead give {"text": None} for all examples.
        iterable_dataset._info  = info
        iterable_dataset._split = dataset._split

    return iterable_dataset