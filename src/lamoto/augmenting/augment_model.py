from abc import ABC, abstractmethod
from typing import List

from transformers import PreTrainedModel, PreTrainedTokenizerBase


class ModelAugmentation(ABC):
    """
    Defines an in-place modification for HuggingFace models.
    The reason it is in-place is that it makes little sense to make a copy of the model every time you modify it slightly.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def augment(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
        pass


class ModelAugmentationSequence(ModelAugmentation):

    def __init__(self, augmentations: List[ModelAugmentation]):
        super().__init__("+".join([a.name for a in augmentations]))
        self.augmentations = augmentations

    def augment(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
        for a in self.augmentations:
            a.augment(model, tokenizer)
