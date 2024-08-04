from abc import ABC, abstractmethod
from typing import List

from transformers import PreTrainedModel, PreTrainedTokenizerBase


class ModelAugmentation(ABC):

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def augment(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> PreTrainedModel:
        pass


class ModelAugmentationSequence(ModelAugmentation):

    def __init__(self, augmentations: List[ModelAugmentation]):
        super().__init__("+".join([a.name for a in augmentations]))
        self.augmentations = augmentations

    def augment(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> PreTrainedModel:
        for a in self.augmentations:
            model = a.augment(model, tokenizer)
        return model
