from enum import Enum
from typing import List, Dict, Tuple, Any, Union, Optional
from transformers import PreTrainedTokenizerBase

from ..util.exceptions import ImpossibleBranchError


class WordLevelPreprocessor:
    """
    Handles tokenisation, truncation, and special-token padding for tasks which have their input pre-segmented into words
    (like PoS, NER, DP ... often token-level tasks) where you need to keep the tokens separate and some labels are word indices
    rather than class numbers.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_tokens: int, add_specials: bool):
        self._tokenizer    = tokenizer
        self._truncate_to  = max_tokens
        self._add_specials = add_specials

    def reservedForSpecials(self) -> int:
        return self._tokenizer.num_special_tokens_to_add(pair=False)*self._add_specials

    def preprocess(self, words: List[str], indexing_labels: Dict[str,List[int]], other_labels: Dict[str,List[Any]]) -> Tuple[List[List[int]], Dict[str,List[int]], Dict[str,List[Any]]]:
        tokens, indexing_labels, other_labels = self.tokenizeAndTruncate(words, indexing_labels, other_labels)
        if self._add_specials:
            tokens, indexing_labels, other_labels = self.addSpecialsAndShiftIndices(tokens, indexing_labels, other_labels)
        return tokens, indexing_labels, other_labels

    def tokenizeAndTruncate(self, words: List[str], indexing_labels: Dict[str,List[int]], other_labels: Dict[str,List[Any]]) -> Tuple[List[List[int]], Dict[str,List[int]], Dict[str,List[Any]]]:
        max_tokens = self._truncate_to - self.reservedForSpecials()
        tokens_so_far = 0

        subword_ids_per_word = []
        for word_idx, word in enumerate(words):
            subwords = self._tokenizer(word, add_special_tokens=False)["input_ids"]
            tokens_so_far += len(subwords)
            if tokens_so_far < max_tokens:
                subword_ids_per_word.append(subwords)
            else:
                excess = tokens_so_far - max_tokens
                subword_ids_per_word.append(subwords[:len(subwords) - excess])

                # Cut away the rest of the labels.
                for field in indexing_labels:
                    indexing_labels[field] = indexing_labels[field][:word_idx+1]
                for field in other_labels:
                    other_labels[field] = other_labels[field][:word_idx+1]
                break

        return subword_ids_per_word, indexing_labels, other_labels

    def addSpecialsAndShiftIndices(self, tokens: List[List[int]], indexing_labels: Dict[str,List[int]], other_labels: Dict[str,List[Any]]) -> Tuple[List[List[int]], Dict[str,List[int]], Dict[str,List[Any]]]:
        # Now we introduce specials anywhere in the sequence.
        tokens_with_specials: List[Union[List[int],int]] = self._tokenizer.build_inputs_with_special_tokens(tokens)

        # The tokeniser's build_inputs_with_special_tokens has inserted special tokens in places we don't know. We now
        # want to transfer this to the labels, but the labels currently have the wrong length. The only way to correct
        # them is with a loop that remembers both the index in the original labels and in the new tokens.
        # First, find how much these specials have offset each word.
        mask = self._tokenizer.get_special_tokens_mask(tokens_with_specials, already_has_special_tokens=True)  # I trust the implementation for already_has_special_tokens=False more, but you're not allowed to use it with a fast tokenizer apparently.
        offsets = [sum(mask[:i]) for i in range(len(mask)) if mask[i] != 1]  # Indexable on word indices (where "word" can also be the start or end dummy, since they do not contribute to offsets), so the new head index is head + offsets[head].

        # Insert -100 as the labels of the special tokens, and turn [special, [tokens], special] into [[special], [tokens], [special]].
        indexing_labels_with_specials = {field: [] for field in indexing_labels}
        other_labels_with_specials    = {field: [] for field in other_labels}

        old_i = 0
        for new_i, e in enumerate(tokens_with_specials):
            is_special = bool(mask[new_i])
            if is_special:  # Don't advance the cursor in the old label lists. Make up a label of -100 on the spot.
                assert isinstance(e, int)
                tokens_with_specials[new_i] = [tokens_with_specials[new_i]]
                for field in indexing_labels_with_specials:
                    indexing_labels_with_specials[field].append(-100)
                for field in other_labels_with_specials:
                    other_labels_with_specials[field].append(-100)
            else:
                assert isinstance(e, list)
                for field in indexing_labels_with_specials:  # Offset these labels
                    label = indexing_labels[field][old_i]
                    indexing_labels_with_specials[field].append(label + offsets[old_i]  if label != -100 else  -100)
                for field in other_labels_with_specials:  # Copy these labels
                    label = other_labels[field][old_i]
                    other_labels_with_specials[field].append(label)

                old_i += 1

        return tokens_with_specials, indexing_labels_with_specials, other_labels_with_specials


class WordLevelPreprocessorWithDummies(WordLevelPreprocessor):

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_tokens: int, add_specials: bool, redirect_to_dummy_if_index_was_truncated: bool):
        """
        :param redirect_to_dummy_if_index_was_truncated: Words that are truncated cannot be referred to by their index
                                                         anymore, so for the remaining words, if such an index is used
                                                         as label, either you no longer predict anything (False) or you
                                                         add a dummy token at the end (True) which you predict if you
                                                         want to signify "the index I had to predict is further away than this".
        """
        super().__init__(tokenizer, max_tokens, add_specials)
        self._end_dummy = redirect_to_dummy_if_index_was_truncated

    def reservedForSpecials(self) -> int:
        return self._tokenizer.num_special_tokens_to_add(pair=False)*self._add_specials + (1 + self._end_dummy)

    def preprocess(self, words: List[str], indexing_labels: Dict[str,List[int]], other_labels: Dict[str,List[Any]]) -> Tuple[List[List[int]], Dict[str,List[int]], Dict[str,List[Any]]]:
        tokens, indexing_labels, other_labels = self.tokenizeAndTruncate(words, indexing_labels, other_labels)
        tokens, indexing_labels, other_labels = self.addDummies(tokens, indexing_labels, other_labels)
        if self._add_specials:
            tokens, indexing_labels, other_labels = self.addSpecialsAndShiftIndices(tokens, indexing_labels, other_labels)
        return tokens, indexing_labels, other_labels

    def addDummies(self, tokens: List[List[int]], indexing_labels: Dict[str,List[int]], other_labels: Dict[str,List[Any]]) -> Tuple[List[List[int]], Dict[str,List[int]], Dict[str,List[Any]]]:
        """
        :param indexing_labels: Lists of labels, with each label being the index of one of the words in the inputs. The
                                assumption is that
                                    1. These labels are 1-based (the index used to refer to the first word in the input is 1);
                                    2. The label for "index points to a special unit outside the sentence" is 0.
        """
        # Make the 1-based heads (with 0 the root) actually correspond to indices. We add a dummy at index 0 with all labels -100.
        tokens.insert(0, [self._tokenizer.unk_token_id])  # Needs to be a single-element list because otherwise in the specials mask below it is counted for offsets.
        for field in indexing_labels:
            indexing_labels[field].insert(0, -100)
        for field in other_labels:
            other_labels[field].insert(0, -100)

        if self._end_dummy:
            tokens.append([self._tokenizer.unk_token_id])
            for field in indexing_labels:
                indexing_labels[field].append(-100)
            for field in other_labels:
                other_labels[field].append(-100)

        # Due to truncation, some of the incoming arrows for the remaining words have no starting point (some heads have disappeared).
        # Set these to -100 if there is no overflow dummy.
        maximal_index     = len(tokens) - 1  # If a dummy has been added, len(tokens)-1 is the index of the dummy. Word index len(tokens)-1 has been truncated and hence those labels should indeed point to the dummy.
        replacement_index = len(tokens) - 1 if self._end_dummy else -100
        for field in indexing_labels:
            indexing_labels[field] = [index if index <= maximal_index else replacement_index
                                      for index in indexing_labels[field]]

        # At this point in the code, all index values are correct.
        return tokens, indexing_labels, other_labels


class LabelPooling(Enum):  # Not the same as subword pooling, but similar; subword pooling requires a BaseModelExtended, generates N subword embeddings but sends <N embeddings to the head. With label pooling everything is sent to the head, just with some labels blacked out.
    FIRST = 1
    LAST  = 2
    ALL   = 3


class FlattenWordLabels(WordLevelPreprocessor):
    """
    Flattens a list of lists of tokens, duplicates the corresponding labels, and disable some of those.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_tokens: int, add_specials: bool, pooling_mode: LabelPooling):
        super().__init__(tokenizer, max_tokens, add_specials)
        self._pooling = pooling_mode

    def preprocess(self, words: List[str], labels: Dict[str,List[Any]]) -> Tuple[List[Any], Dict[str,List[Any]]]:
        tokens, _, labels = self.tokenizeAndTruncate(words, dict(), labels)
        if self._add_specials:
            tokens, _, labels = self.addSpecialsAndShiftIndices(tokens, _, labels)
        return self.flattenTokensAndPoolLabels(tokens, labels)

    def flattenTokensAndPoolLabels(self, tokens: List[List[int]], labels: Dict[str,List[Any]]) -> Tuple[List[int], Dict[str,List[Any]]]:
        flattened_tokens = []
        flattened_labels: Dict[str,list] = {field: [] for field in labels}

        for word_idx, subword_tokens in enumerate(tokens):
            flattened_tokens.extend(subword_tokens)
            n_subwords = len(subword_tokens)
            for field in labels:
                actual_label = labels[field][word_idx]
                extra_flattened_labels = [-100]*n_subwords
                if self._pooling == LabelPooling.FIRST:
                    extra_flattened_labels[0] = actual_label
                elif self._pooling == LabelPooling.LAST:
                    extra_flattened_labels[-1] = actual_label
                elif self._pooling == LabelPooling.ALL:
                    extra_flattened_labels = [actual_label]*n_subwords
                else:
                    raise ImpossibleBranchError()
                flattened_labels[field].extend(extra_flattened_labels)

        return flattened_tokens, flattened_labels
