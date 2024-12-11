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
from datasets import DatasetDict

from ._general import ClassifySentenceSuperGLUETask
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
        super().__init__(task_name="WiC")

    def _prepareDataset(self, dataset: DatasetDict) -> DatasetDict:
        def preprocess(example):  # FIXME: This doesn't work for CLS-SEP-based vocabularies rather than BOS-EOS.
            input_ids = \
                [self.tokenizer.bos_token_id] + \
                self.tokenizer(example["word"], add_special_tokens=False, return_attention_mask=False)["input_ids"] + \
                [self.tokenizer.eos_token_id] + \
                self.tokenizer(example["sentence1"], add_special_tokens=False, return_attention_mask=False)["input_ids"] + \
                [self.tokenizer.eos_token_id] + \
                self.tokenizer(example["sentence2"], add_special_tokens=False, return_attention_mask=False)["input_ids"] + \
                [self.tokenizer.eos_token_id]
            return {"input_ids": input_ids, "attention_mask": [1]*len(input_ids)}

        return replaceDatasetColumns_OneExampleToOneExample(dataset, preprocess, but_keep={"label"})
