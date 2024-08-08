from datasets.arrow_dataset import DatasetInfoMixin
from datasets import Dataset, IterableDataset


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
            raise ValueError(f"Could not resolve size of dataset split '{split}'.")


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
