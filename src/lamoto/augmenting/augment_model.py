from abc import ABC, abstractmethod
from transformers import PreTrainedModel, PreTrainedTokenizerBase


class ModelAugmentation(ABC):

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def augment(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> PreTrainedModel:
        pass
