from typing import List, Dict, Union
from dataclasses import dataclass
from collections import Counter

from datasets import load_dataset
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin
import torch
from supar.utils.fn import pad

from archit.instantiation.tasks import ForDependencyParsing, DependencyParsingHeadConfig

from ._core import *
from ..measuring import DependencyParsingMetrics, Metric
from ..preprocessing.ud import FilterAndCorrectUDtypes
from ..preprocessing.wordlevel import WordLevelPreprocessorWithDummies
from ..util.visuals import log


def relu(x):
    return max(0,x)


@dataclass
class DataCollatorForDependencyParsing(DataCollatorMixin):
    """
    Takes care of two preprocessing responsibilities:
        - Since dependency parsing has two label sequences instead of one, it pads both of them, not just one.
        - Handles 3D padding instead of 2D padding. Since the supar dependency parser uses token-to-word pooling, it
          expects not a 2D (batch x subwords) input_ids tensor but a 3D one (batch x words x subwords).
          This also includes truncating token sequences for one word that are too long. Truncation is normally handled
          by the tokeniser, but it makes more sense to do truncation and padding in the same place.

    What is expected of the incoming examples:
        - They should individually already be ready to be fed to a transformer. That means a tokeniser must have already
          converted strings to integer IDs.
        - Special tokens have to be added already and the labels must reflect this by having a -100 for those words.

    Why have this collator if supar already works, so it has some kind of built-in collator?
        - supar doesn't allow choosing the special token template, it does it for the user, no matter the tokeniser.
        - supar pads its labels somewhere deep inside obscure classes. The labels are padded when a list of sentences
          is converted to a batch, since Batch.compose calls Field.compose which is just a redirect to the pad() function.
          Batch.compose is called inside the "PrefetchGenerator" that underlies supar's DataLoader
          (https://github.com/yzhangcs/parser/blob/bebdd350e034c517cd5b71185e056503290164fa/supar/utils/data.py#L343).
          So in short: yeah there is some kind of collation happening, but holy fuck is it hidden deep.
    """

    tokenizer: PreTrainedTokenizerBase
    max_subwords_per_word: int=None
    label_pad_value: int = -100
    return_tensors: str = "pt"

    def torch_call(self, examples: List[dict]):
        # First handle the tokens.
        #   - Truncate words with many tokens, and pad those with few tokens.
        for example in examples:
            subword_dim = min(self.max_subwords_per_word or 1e99, max(map(len, example["words"])))  # Keep the per-example tensors as small as possible.
            example["words"] = [subwords[:subword_dim] + relu(subword_dim - len(subwords))*[self.tokenizer.pad_token_id]
                                for subwords in example["words"]]

        #   - Take tensors of size words_1 x max(tokens_1), ..., words_n x max(tokens_n) and turn them into a single tensor
        #     n x max(words_i) x max(max(tokens_i)_i)
        ids_tensor = pad([torch.tensor(example["words"], dtype=torch.long) for example in examples], self.tokenizer.pad_token_id)

        # For the labels, we assume there is going to be at least one sequence that has as many labels as ids_tensor.size(1).
        # (Because we assume that for all examples, len(example["words"]) == len(example["labels_arcs"]) == len(example["labels_rels"])).
        labels_arcs = pad([torch.tensor(example["labels_arcs"], dtype=torch.long) for example in examples], self.label_pad_value)
        labels_rels = pad([torch.tensor(example["labels_rels"], dtype=torch.long) for example in examples], self.label_pad_value)

        # Attention mask is at the word level and should hence work the same as the labels.
        attention_mask = pad([torch.tensor(example["attention_mask"], dtype=torch.long) for example in examples], self.label_pad_value)
        assert ids_tensor.size(1) == labels_arcs.size(1) == labels_rels.size(1) == attention_mask.size(1)  # == the amount of words.

        return {
            # "words": ids_tensor,
            # "labels_arcs": labels_arcs,
            # "labels_rels": labels_rels
            "input_ids": ids_tensor,
            "attention_mask": attention_mask,
            "labels": (labels_arcs, labels_rels)
        }


class DP(Task[DependencyParsingHeadConfig]):
    """
    Dependency parsing measured by UAS and LAS.
    """

    def __init__(self):
        # The dataloader doesn't provide a domain for the labels. Since UD allows using "sublabels", we can't even hardcode the 37 standard labels ["nsubj", "obj", "iobj", "csubj", "ccomp", "xcomp", "obl", "vocative", "expl", "dislocated", "advcl", "advmod", "discourse", "aux", "cop", "mark", "nmod", "appos", "nummod", "acl", "amod", "det", "clf", "case", "conj", "cc", "fixed", "flat", "list", "parataxis", "compound", "orphan", "goeswith", "reparandum", "punct", "root", "dep"]
        tags = self.getTagset()
        self.tagset = list(sorted(tags, key=tags.get, reverse=True))
        self.reltag_to_id = {tag: i for i, tag in enumerate(self.tagset)}
        super().__init__(
            task_name="DP",
            metric_config=MetricSetup(
                to_compute=["attachment"],
                to_track={
                    "attachment": {
                        "uas": "UAS",
                        "las": "LAS",
                        "ucm": "UCM",
                        "lcm": "LCM"
                    }
                }
            ),
            archit_class=ForDependencyParsing,
            automodel_class=None,  # Technically there is no HuggingFace-compatible AutoModelForDP. Best we have is supar, but it can only load from a base model checkpoint, not from a supar checkpoint!

            num_labels=len(self.tagset)
        )

        # Extra temporary field that must be reset every time you reset the metrics.
        self._metric: DependencyParsingMetrics = None

    def resetTemporaryFields(self):
        super().resetTemporaryFields()
        self._metric = None

    def _setMetrics(self, m: Dict[str, Metric]):
        super()._setMetrics(m)
        self._metric = self.metrics["attachment"]

    def sneakyLogitTransform(self, logits: Tuple[torch.Tensor,torch.Tensor], labels: Tuple[torch.Tensor,torch.Tensor]):
        self._metric.add(DependencyParsingMetrics.logitsAndLabelsToMetric(logits, labels))
        return torch.tensor([[1]], device=logits[0].device)

    def _loadDataset(self) -> DatasetDict:
        return load_dataset("universal-dependencies/universal_dependencies", "en_ewt", trust_remote_code=True)

    def getTagset(self) -> Counter:
        log("Generating tagset manually...")
        counter = Counter()
        dataset = self._loadDataset()
        for split in dataset:
            for label_sequence in dataset[split]["deprel"]:
                counter.update(label_sequence)
        log("Finished generating tagset.")
        return counter

    def adjustHyperparameters(self, hp: TaskHyperparameters[DependencyParsingHeadConfig]):
        hp.archit_head_config.num_labels = len(self.tagset)

    def _prepareDataset(self, dataset: DatasetDict) -> DatasetDict:
        filter_and_correct = FilterAndCorrectUDtypes(self.reltag_to_id)
        truncator = WordLevelPreprocessorWithDummies(self.tokenizer, max_tokens=self._getMaxInputLength(), add_specials=self.hyperparameters.ADD_SPECIAL_TOKENS, redirect_to_dummy_if_index_was_truncated=False)

        def datasetMap(example: dict):
            # Filter entries with None head, map strings to ints, and truncate everything to a fixed token limit.
            words, heads, deprels, _ = filter_and_correct.preprocess(words=example["tokens"], heads=example["head"], relations=example["deprel"])
            tokens, labels1, labels2 = truncator.preprocess(words, {"labels_arcs": heads}, {"labels_rels": deprels})

            # Marshal the results.
            return {
                "words": tokens,  # List of lists, because this is the format expected by ArchIt's BaseModelExtended class.
                "labels_arcs": labels1["labels_arcs"],
                "labels_rels": labels2["labels_rels"],
                "attention_mask": [1]*len(tokens)  # word-level attention mask for the head
            }

        dataset = dataset.map(datasetMap, batched=False)
        dataset = dataset.remove_columns(["tokens", "idx", "text", "lemmas", "upos", "xpos", "feats", "head", "deprel", "deps", "misc"])
        return dataset

    def getCollator(self) -> DataCollator:
        return DataCollatorForDependencyParsing(self.tokenizer)

    def getPredictionsAndReferences(self, eval: EvalPrediction) -> Tuple[Any,Any]:
        return [], []
