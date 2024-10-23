from tktkt.util.printing import gridify

from tst.preamble import *

from lamoto.preprocessing.wordlevel import WordLevelPreprocessorWithDummies, FlattenWordLabels, LabelPooling

from transformers import AutoTokenizer, PreTrainedTokenizerBase


def tst_dp():
    tk: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained("roberta-base")
    sentence = ["sentence", "arguably", "discombobulated", "overexaggerated", "superduperlong"]
    print(tk(sentence, add_special_tokens=False)["input_ids"])
    prep = WordLevelPreprocessorWithDummies(tokenizer=tk, max_tokens=10, add_specials=False, redirect_to_dummy_if_index_was_truncated=True)
    print(prep.preprocess(
        sentence,
        {"head": [4, 0, 3, 1, 2]},
        {"rel": [100, 200, 300, 400, 500]}))


def tst_pos():
    tk: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained("roberta-base")
    sentence = ["sentence", "arguably", "discombobulated", "overexaggerated", "superduperlong"]
    print(tk(sentence, add_special_tokens=False)["input_ids"])
    print()

    prep = FlattenWordLabels(tokenizer=tk, max_tokens=10, add_specials=True, pooling_mode=LabelPooling.FIRST)
    tokens, labels = prep.preprocess(
        sentence,
        {"labels": [4, 0, 3, 1, 2]})

    print(gridify([tokens, labels["labels"]]))


if __name__ == "__main__":
    tst_pos()