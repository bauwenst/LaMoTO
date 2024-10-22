from tst.preamble import *

from lamoto.tasks.dp import TruncatingWordLabelPreprocessor

from transformers import AutoTokenizer, PreTrainedTokenizerBase


tk: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained("roberta-base")
sentence = ["sentence", "arguably", "discombobulated", "overexaggerated", "superduperlong"]
print(tk(sentence, add_special_tokens=False)["input_ids"])
prep = TruncatingWordLabelPreprocessor(tokenizer=tk, max_tokens=10, redirect_to_dummy_if_index_was_truncated=True)
print(prep.preprocess(
    sentence,
    {"head": [4, 0, 3, 1, 2]},
    {"rel": [100, 200, 300, 400, 500]}))
