import datasets
from datasets import IterableDatasetDict

from ._general import Corpus, Language, L


class C4(Corpus):
    def _isValidLanguage(self, language: Language) -> bool:
        return language == L("English")
    
    def _loadDataset(self, language: Language) -> datasets.IterableDatasetDict:
        dataset: datasets.IterableDatasetDict = datasets.load_dataset("allenai/c4", "en", streaming=True)
        return dataset.remove_columns(["timestamp", "url"])


class SlimPajama(Corpus):
    def _isValidLanguage(self, language: Language) -> bool:
        return language == L("English")

    def _loadDataset(self, language: Language) -> datasets.IterableDatasetDict:
        dataset: datasets.IterableDatasetDict = datasets.load_dataset("cerebras/SlimPajama-627B", streaming=True, trust_remote_code=True)
        return dataset.remove_columns(["meta"])


class FineWeb(Corpus):

    # FIXME: Add Fineweb-2 languages (i.e. all non-English Fineweb), but note that these only have a test set, not a validation set.

    _ENGLISH_VALIDATION_SIZE = 100_000

    def _isValidLanguage(self, language: Language) -> bool:
        return language == L("English")

    def _loadDataset(self, language: Language) -> datasets.IterableDatasetDict:
        # if self._language == L("English"):
        dataset: datasets.IterableDatasetDict = datasets.load_dataset("HuggingFaceFW/fineweb", "default", streaming=True, trust_remote_code=True)
        full_iterable: datasets.IterableDataset = dataset["train"]
        dataset["train"]      = full_iterable.skip(FineWeb._ENGLISH_VALIDATION_SIZE)
        dataset["validation"] = full_iterable.take(FineWeb._ENGLISH_VALIDATION_SIZE)
        # else:
        #     dataset: IterableDatasetDict = datasets.load_dataset("HuggingFaceFW/fineweb-2", ..., streaming=True, trust_remote_code=True)
        return dataset.remove_columns(["id", "dump", "url", "date", "file_path", "language", "language_score", "token_count"])


class FineWiki(Corpus):

    def _isValidLanguage(self, language: Language) -> bool:
        return True  # Assumed true for all languages because FineWiki is really that large.

    def _loadDataset(self, language: Language) -> IterableDatasetDict:
        dataset: datasets.IterableDatasetDict = datasets.load_dataset("HuggingFaceFW/finewiki", name=language.to_tag(), streaming=True)
        return dataset.select_columns(["text"])
