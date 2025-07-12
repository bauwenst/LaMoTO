"""
FIXME: It probably makes more sense to have
       [CLS] ... [SEP1] ... [SEP2] ... [SEP3]
       rather than
       [CLS] ... [SEP] ... [SEP] ... [SEP]
       but this will require embedding matrix resizing etc.

TODO: The architecture described by the authors is not bad. Mean-pooling the word+s1+s2 embeddings may teach the model
      to eventually neutralise all words except those between CLS and SEP, but much more salient would be to actually
      have an extra ArchIt architecture that mean-pools the ranges between specials (here, generating 3 embeddings)
      and then either sends them through a 3H x C linear head or something more complex.
"""
from typing import Tuple, List
from datasets import DatasetDict
from transformers import PreTrainedTokenizerBase

from tktkt.util.iterables import indexSpan

from ._general import ClassifySentenceSuperGLUETask
from .._core import RankingMetricSpec
from ...util.datasets import replaceDatasetColumns_OneExampleToOneExample


class WiC(ClassifySentenceSuperGLUETask):
    """
    The task consists of triplet inputs
        (sentence1, sentence2, word)
    where the goal is to predict whether the word's sense is the same in both sentences.

    "For classification tasks with sentence-pair inputs (BoolQ, CB, RTE, WiC), we concatenate the
    sentences with a [SEP] token, feed the fused input to BERT, and use a logistic regression classifier that
    sees the representation corresponding to [CLS]. For WiC only, we also concatenate the representation
    of the marked word to the [CLS] representation."

    itrain fine-tunes this with a template [CLS] s1 [SEP] s2 [SEP] word [SEP]. Clearly not the same as described,
    but it is contextualised. Easier to hack the tokeniser to do this than to hack inference to concatenate an embedding to the pooled output.
    """
    def __init__(self):
        super().__init__(task_name="WiC", rank_by=RankingMetricSpec("matthews_correlation", "matthews_correlation", True))

    def _prepareDataset(self, dataset: DatasetDict) -> DatasetDict:
        bos_ids, sep_ids, eos_ids = deduceTemplateSpecials(self.tokenizer)
        bos_ids, sep_ids, eos_ids = self.tokenizer.convert_tokens_to_ids(bos_ids), self.tokenizer.convert_tokens_to_ids(sep_ids), self.tokenizer.convert_tokens_to_ids(eos_ids)

        def preprocess(example):
            input_ids = \
                bos_ids + \
                self.tokenizer(example["word"], add_special_tokens=False, return_attention_mask=False)["input_ids"] + \
                sep_ids + \
                self.tokenizer(example["sentence1"], add_special_tokens=False, return_attention_mask=False)["input_ids"] + \
                sep_ids + \
                self.tokenizer(example["sentence2"], add_special_tokens=False, return_attention_mask=False)["input_ids"] + \
                eos_ids
            return {"input_ids": input_ids, "attention_mask": [1]*len(input_ids)}

        return replaceDatasetColumns_OneExampleToOneExample(dataset, preprocess, but_keep={"label"})


def deduceTemplateSpecials(hf_tokeniser: PreTrainedTokenizerBase) -> Tuple[List[str], List[str], List[str]]:
    x_tokens = hf_tokeniser.tokenize("x", add_special_tokens=False)
    y_tokens = hf_tokeniser.tokenize("y", add_special_tokens=False)

    tokens = hf_tokeniser.tokenize("x", "y", add_special_tokens=True)

    try:
        x_start, x_end = indexSpan(x_tokens, tokens)
    except:
        x_start, x_end = 0, 0

    try:
        y_start, y_end = indexSpan(y_tokens, tokens)
    except:
        y_start, y_end = 0, 0

    if y_start < x_start:
        raise NotImplementedError("It seems that the given tokeniser reverses its input!")

    bos = tokens[:x_start]
    sep = tokens[x_end:y_start]
    eos = tokens[y_end:]

    return bos, sep, eos
