from datasets.arrow_dataset import DatasetInfoMixin


def getDatasetSize(dataset: DatasetInfoMixin, split: str="train"):  # DatasetInfoMixin is the parent class for Dataset and IterableDataset.
    """
    Get the amount of examples in a HuggingFace dataset, whether it is a regular Dataset or a streamed IterableDataset.
    """
    if isinstance(dataset, dict):
        raise TypeError("Expected a dataset but a dictionary was given (DatasetDict or IterableDatasetDict) was given.")

    try:
        return len(dataset)
    except:
        try:  # Should work for both Dataset and IterableDataset.
            return dataset.info.splits[split].num_examples
        except:
            raise ValueError(f"Could not resolve size of training dataset split '{split}'.")
