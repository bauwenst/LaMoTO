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
        """Apply an in-place augmentation to the given PyTorch model."""
        pass

    def augmentAndLoad(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, checkpoint: str):
        """Augment the model and load augmentation parameters."""
        self.augment(model, tokenizer)
        self._load_pretrained(model, tokenizer, checkpoint)

    @abstractmethod
    def _load_pretrained(self, augmented_model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, checkpoint: str):
        """
        If the augmentation adds trainable parameters, they will be captured in checkpoints of the augmented model,
        but .from_pretrained() will not load them since it only loads parameters declared beforehand in the model class.
        To load a checkpoint, HF randomly initialises the unaugmented model and loads the checkpoint parameters for that
        model. We then apply the model augmentation, randomly initialising the added parameters, and then call this method.

        There is no guarantee that the given checkpoint contains the weights needed for this augmentation. (This is
        the case when augmenting a checkpoint that existed before augmentation.) You will have to check for that.
        To convert a checkpoint into a state dict, see archit.util.

        TODO: It may be possible to remove some boilerplate work from this method. augmentAndLoad() could detect which
              keys in the state dict change before vs. after the augmentation, particularly the ones that are new. Then,
              you could look for those keys in the checkpoint and load them in immediately (which HF or PT has a function for already, I think).
        """
        pass


class ModelAugmentationSequence(ModelAugmentation):

    def __init__(self, augmentations: List[ModelAugmentation]):
        super().__init__("+".join([a.name for a in augmentations]))
        self.augmentations = augmentations

    def augment(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
        for a in self.augmentations:
            a.augment(model, tokenizer)

    def augmentAndLoad(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, checkpoint: str):
        for a in self.augmentations:
            a.augmentAndLoad(model, tokenizer, checkpoint)

    def _load_pretrained(self, augmented_model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, checkpoint: str):
        pass
