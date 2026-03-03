from typing import Union
from abc import abstractmethod, ABC
from dataclasses import dataclass
from datasets import IterableDatasetDict, IterableDataset

import torch
from transformers import DataCollatorForLanguageModeling
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from archit.instantiation.tasks import ModelWithHead

from tktkt.util.types import Languish, L, Language

from .._core import *
from ...util.datasets import PackedDataset, FieldType
from ...training.auxiliary.hyperparameters import HC


@dataclass
class TaskMetadata:
    task_name: str
    text_fields: list[Union[str,FieldType]]
    label_field: Union[Union[str,FieldType], list[Union[str,FieldType]]]
    metric_config: MetricSetup
    archit_class: type[ModelWithHead]
    automodel_class: type[_BaseAutoModelClass]
    automodel_kwargs: dict[str,Any]


class Corpus(ABC):
    """Object which has an argumentless method for loading a specific dataset."""
    def __init__(self, language: Languish):
        language = L(language)
        if not self._isValidLanguage(language):
            raise ValueError(f"Invalid language: {language}")
        self._language = language

    @abstractmethod
    def _isValidLanguage(self, language: Language) -> bool:
        pass

    @abstractmethod
    def _loadDataset(self, language: Language) -> IterableDatasetDict:
        pass

    def loadDataset(self) -> IterableDatasetDict:
        return self._loadDataset(self._language)


class TokenPredictionTask(Task[HC]):
    """
    Classification task where the goal is to predict one or more tokens from a vocabulary of tens of thousands of possibilities.
    """

    def __init__(self, corpus: Corpus, packing: bool=False, drop_train_examples: int=0, use_perplexity: bool=False):
        """
        :param packing: Whether to concatenate tokens from multiple dataset examples to fill up the model's context length.
        :param use_pppl: Whether to evaluate with PPPL. Note that this takes way more time than the usual evaluation, which is
                         just computing NLL on masked evaluation examples.
        :param drop_train_examples: How many training examples to advance by before starting training.
                                    Note: a *training example* is not the same as a *dataset example*. Training examples
                                    are given to models, and are the unit we measure batch size in. Dataset examples can
                                    be much larger (e.g. for datasets of long articles) or much smaller (e.g. tweets)
                                    than training examples. Since you only know how many batches have been trained for,
                                    rather than how many examples have been used from the underlying dataset, we skip
                                    in units of train examples.
        """
        metadata = self._metadata(use_perplexity=use_perplexity)
        super().__init__(
            task_name=metadata.task_name,
            text_fields=metadata.text_fields,
            label_field=metadata.label_field,
            metric_config=metadata.metric_config,
            archit_class=metadata.archit_class,
            automodel_class=metadata.automodel_class
        )
        self._corpus = corpus
        self._drop_train = max(0,drop_train_examples)
        self._do_packing = packing
        self._do_perplexity = use_perplexity

    @classmethod
    @abstractmethod
    def _metadata(cls, use_perplexity: bool) -> TaskMetadata:
        pass

    @abstractmethod
    def _isAutoregressive(self) -> bool:
        pass

    def _loadDataset(self) -> IterableDatasetDict:
        return self._corpus.loadDataset()

    def _prepareDataset(self, dataset: IterableDatasetDict) -> IterableDatasetDict:
        def non_packing_preprocessor(example):
            return self.tokenizer(example["text"], is_split_into_words=False, add_special_tokens=self.hyperparameters.add_special_tokens, truncation=True, max_length=self._getMaxInputLength())

        if self._do_packing:  # Train split is not tokenised here but in the packer.
            dataset["train"] = PackedDataset(dataset["train"], self.tokenizer, context_length=self._getMaxInputLength())
            if not self._do_perplexity:  # When you use a perplexity metric, the validation split is tokenised by the metric itself. Without PPPL, you need to tokenise the validation set yourself for HuggingFace's logit calculation. As is customary, this involves truncation (i.e. data is lost for examples that are too long), which is not the case when packing.
                validation_set: IterableDataset = dataset["validation"]
                validation_set = validation_set.map(non_packing_preprocessor, batched=False)
                validation_set = validation_set.remove_columns(["text"])
                dataset["validation"] = validation_set
        else:  # If you don't pack, you tokenise the whole corpus and that's it. Does have truncation to the context length, as per above.
            dataset = dataset.map(non_packing_preprocessor, batched=False)
            dataset = dataset.remove_columns(["text"])

        if self._drop_train:  # Drop AFTER packing.
            dataset["train"] = dataset["train"].skip(self._drop_train)

        return dataset

    def adjustHyperparameters(self, hp: TaskHyperparameters[HC]):
        pass

    def getCollator(self) -> DataCollator:
        if self._isAutoregressive():
            return DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        else:
            return DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=self.hyperparameters.mlm_probability)

    def sneakyLogitTransform(self, logits, labels):
        return torch.tensor([[1]], device=logits[0].device)

    def getPredictionsAndReferences(self, eval: EvalPrediction) -> Tuple[Any, Any]:
        return None, None


###################################
### FIXING HUGGINGFACE DATASETS ###
###################################
# See https://github.com/huggingface/datasets/issues/7440 for what the issue is.
# Until recently, nothing could be done about this, but since then the offending code has been put into a class method,
# which can be monkey-patched externally.
from typing import Union, Optional, Generator, Type
from datasets.utils.file_utils import FilesIterable, DownloadConfig, xisfile, xisdir, xbasename, xjoin, xwalk, logger
import time


def _iter_from_urlpaths(cls: Type[FilesIterable], urlpaths: Union[str, list[str]], download_config: Optional[DownloadConfig]=None) -> Generator[str, None, None]:
    MAX_TRIES = 6
    SKIP_EXCEPTION = False

    if not isinstance(urlpaths, list):
        urlpaths = [urlpaths]

    for urlpath in urlpaths:
        n_tries = 0
        while True:
            if xisfile(urlpath, download_config=download_config):  # This function just flat-out lies sometimes lmao
                yield urlpath
                break
            elif xisdir(urlpath, download_config=download_config):
                for dirpath, dirnames, filenames in xwalk(urlpath, download_config=download_config):
                    # in-place modification to prune the search
                    dirnames[:] = sorted([dirname for dirname in dirnames if not dirname.startswith((".", "__"))])
                    if xbasename(dirpath).startswith((".", "__")):
                        # skipping hidden directories
                        continue
                    for filename in sorted(filenames):
                        if filename.startswith((".", "__")):
                            # skipping hidden files
                            continue
                        yield xjoin(dirpath, filename)
                break

            n_tries += 1
            if n_tries >= MAX_TRIES:
                if SKIP_EXCEPTION:
                    break
                else:
                    raise FileNotFoundError(urlpath)
            else:
                seconds = 1.875 * 2**(n_tries-1)
                logger.warning(f"Supposedly could not find URL path {urlpath} (this was try {n_tries}/{MAX_TRIES}). Retrying in {seconds} seconds.")
                time.sleep(seconds)

# Monkey-patching a class method means that old and new instances are affected.
FilesIterable._iter_from_urlpaths = classmethod(_iter_from_urlpaths)  # You need this wrapper because the replacement method behaves as a static method otherwise, where the first argument you give it will be assigned to 'cls'.
