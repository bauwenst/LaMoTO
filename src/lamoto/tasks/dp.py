from typing import List, Dict, Union
from dataclasses import dataclass
from collections import Counter

from datasets import load_dataset
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin
import torch
from supar.utils.fn import pad

from tktkt.util.printing import gridify
from archit.instantiation.tasks import ForDependencyParsing, DependencyParsingHeadConfig

from ._core import *
from ..measuring import DependencyParsingMetrics


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


class TruncatingWordLabelPreprocessor:
    """
    Handles tokenisation, truncation, and special-token padding for tasks which have their input pre-segmented into words
    (like PoS, NER, DP ... often token-level tasks) where possibly some labels are to word indices.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_tokens: int, redirect_to_dummy_if_index_was_truncated: bool):
        """
        :param redirect_to_dummy_if_index_was_truncated: Words that are truncated cannot be referred to by their index
                                                         anymore, so for the remaining words, if such an index is used
                                                         as label, either you no longer predict anything (False) or you
                                                         add a dummy token at the end (True) which you predict if you
                                                         want to signify "the index I had to predict is further away than this".
        """
        self._tokenizer = tokenizer
        self._truncate_to = max_tokens
        self._end_dummy = redirect_to_dummy_if_index_was_truncated

    def preprocess(self, words: List[str], indexing_labels: Dict[str,List[int]], other_labels: Dict[str,List[Any]]) -> Tuple[List[List[int]], Dict[str,List[int]], Dict[str,List[Any]]]:
        tokens, indexing_labels, other_labels = self._tokenizeAndTruncate(words, indexing_labels, other_labels)
        return self._addSpecialsAndShiftIndices(tokens, indexing_labels, other_labels)

    def _tokenizeAndTruncate(self, words: List[str], indexing_labels: Dict[str,List[int]], other_labels: Dict[str,List[Any]]) -> Tuple[List[List[int]], Dict[str,List[int]], Dict[str,List[Any]]]:
        max_tokens = self._truncate_to - self._tokenizer.num_special_tokens_to_add(pair=False) - 1 - self._end_dummy
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

    def _addSpecialsAndShiftIndices(self, tokens: List[List[int]], indexing_labels: Dict[str,List[int]], other_labels: Dict[str,List[Any]]) -> Tuple[List[List[int]], Dict[str,List[int]], Dict[str,List[Any]]]:
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

        # At this point in the code, all index values are correct. Now we introduce specials anywhere in the sequence.
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


class DP(Task[DependencyParsingHeadConfig]):
    """
    Dependency parsing measured by UAS and LAS.
    """

    def __init__(self):
        # The dataloader doesn't have a domain for the labels. Since UD allows using "sublabels", we can't even hardcode the 37 standard labels.
        # tagset = ["nsubj", "obj", "iobj", "csubj", "ccomp", "xcomp", "obl", "vocative", "expl", "dislocated", "advcl", "advmod", "discourse", "aux", "cop", "mark", "nmod", "appos", "nummod", "acl", "amod", "det", "clf", "case", "conj", "cc", "fixed", "flat", "list", "parataxis", "compound", "orphan", "goeswith", "reparandum", "punct", "root", "dep"]
        tags = self.getTagset()
        self.tagset = list(sorted(tags, key=tags.get, reverse=True))
        self.tag_to_id = {tag: i for i,tag in enumerate(self.tagset)}
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

    def loadDataset(self) -> DatasetDict:
        return load_dataset("universal-dependencies/universal_dependencies", "en_ewt", trust_remote_code=True)

    def getTagset(self) -> Counter:
        print("Generating tagset manually...")
        counter = Counter()
        dataset = self.loadDataset()
        for split in dataset:
            for label_sequence in dataset[split]["deprel"]:
                counter.update(label_sequence)
        print("Finished generating tagset.")
        return counter

    def adjustHyperparameters(self, hp: TaskHyperparameters[DependencyParsingHeadConfig]):
        hp.archit_head_config.num_labels = len(self.tagset)

    def sneakyLogitTransform(self, logits, labels):
        """
        Here's the reasoning behind this method.

        The issue to solve is that when you are evaluating in the HuggingFace trainer, all predictions and labels are
        concatenated into one big tensor. The problem for DP arc predictions is that the amount of possible labels
        CHANGES every batch, because the labels are positions inside the given sentence. Hence, you can't concatenate the
        predictions for several batches because they don't have same amount of prediction classes.
        You don't see this problem in training nor the first evaluation batch, because there is no batch interaction (yet) there.

        As an example of the error you get:
           RuntimeError: The expanded size of the tensor (60) must match the existing size (58) at non-singleton dimension 2.
           Target sizes: [32, 58, 60].  Tensor sizes: [32, 58, 58]
        The first batch had 60 positions, the second batch had 58.

        Any solution for this has to basically force the Trainer to not accumulate logits, and instead commit them to the
        evaluation metric immediately (which is how supar does it). Here's how you could do that:
            1. Write a custom Trainer that has an evaluation_loop that just doesn't accumulate.
               The problem with this approach is that Trainer actually has a bunch of useful acceleration code.
            2. Use the existing Trainer but with empty validation dataset and use some kind of callback to evaluate the model inside
               the callback instead of in Trainer's evaluation loop.
            3. Trainer has an argument preprocess_logits_for_metrics that is called like
                    logits = preprocess_logits_for_metrics(logits,labels)
               before saving the logits. Here's how you could use that:
                    - Instantiate the UAS/LAS metric
                    - Capture it inside the preprocess_logits_for_metrics function and compute it immediately
                    - Let preprocess_logits_for_metrics return empty tensors as logits and labels
                    - Let computeMetrics return that metric's value.
               Another approach:
                    - Let preprocess_logits_for_metrics flatten the B x L x L logits into a B x L² tensor.
                    - When the time comes to computeMetrics, have some way to identify the different lengths L and turn
                      them back into squares inside a Metric.compute.
               Yet another approach:
                    - Compress the B x L x L logits into B x L x 1 class argmaxes, rather than letting the metric do this.
                    - Let computeMetrics finish the process with UAS/LAS.

        This sneakyLogitTransform() method is used to perform the third approach, except more elegantly, we capture `self`
        (the only time Python allows currying is in expressions `self.method`) and then we access the metric instance
        that is already present in `self` anyway.

        The first two approaches have their place too, namely when you want metrics that aren't logit-based, like strided
        PPL in causal LM. That's not the case for DP though.
        """
        self.metrics["attachment"].add(DependencyParsingMetrics.logitsAndLabelsToMetric(logits, labels))
        return torch.tensor([[1]], device=logits[0].device)

    def prepareDataset(self, dataset: DatasetDict) -> DatasetDict:
        truncator = TruncatingWordLabelPreprocessor(self.tokenizer, max_tokens=self._getMaxInputLength(), redirect_to_dummy_if_index_was_truncated=False)

        def preprocess(example):
            # First of all: take out all unlabelled words and their corresponding labels. The heads are counted as the
            # indices in the sequence WITHOUT these words. There are two classes of them:
            #   - MWEs, in which case the words that follow are the decomposition of said word. For example, the MWE
            #     "doctor's" would produce the UD words ["doctor's", "doctor", "'s"] and all arcs referring to the index
            #     of "doctor's" or words after it actually refer to one word in the future (here it would be "doctor").
            #   - Repetition, like the "excited" in "Grace is more excited to see her than she is excited to see me."
            words = []
            heads = []
            deprels = []
            for i in range(len(example["tokens"])):
                if example["head"][i] != "None":
                    words.append(                 example["tokens"][i])
                    heads.append(             int(example["head"  ][i]))
                    deprels.append(self.tag_to_id[example["deprel"][i]])

            tokens, labels1, labels2 = truncator.preprocess(words, {"labels_arcs": heads}, {"labels_rels": deprels})
            enc = dict()
            enc["words"] = tokens  # List of lists, because this is the format expected by ArchIt's BaseModelExtended class.
            enc["labels_arcs"] = labels1["labels_arcs"]
            enc["labels_rels"] = labels2["labels_rels"]
            enc["attention_mask"] = [1]*len(tokens)  # word-level attention mask for the head
            return enc

        def preprocess2(example):
            # First of all: take out all unlabelled words and their corresponding labels. The heads are counted as the
            # indices in the sequence WITHOUT these words. There are two classes of them:
            #   - MWEs, in which case the words that follow are the decomposition of said word. For example, the MWE
            #     "doctor's" would produce the UD words ["doctor's", "doctor", "'s"] and all arcs referring to the index
            #     of "doctor's" or words after it actually refer to one word in the future (here it would be "doctor").
            #   - Repetition, like the "excited" in "Grace is more excited to see her than she is excited to see me."
            words = []
            heads = []
            deprels = []
            for i in range(len(example["tokens"])):
                if example["head"][i] == "None":
                    pass
                    # if example["upos"][i] == 13 and example["xpos"][i] is None and example["deprel"][i] == "_":
                    #     print("\nMWE found:")
                    # else:
                    #     print("\nRepetition found:")
                    # print(gridify([example["tokens"], example["xpos"], example["upos"], example["head"], example["deprel"]]))
                else:
                    words.append(                 example["tokens"][i])
                    heads.append(             int(example["head"  ][i]))
                    deprels.append(self.tag_to_id[example["deprel"][i]])

            # Tokenise all words that were kept.
            subword_ids_per_word = [self.tokenizer(word, add_special_tokens=False)["input_ids"] for word in words]

            # We need to do truncation manually because we made many tokenizer() calls without concatenating.
            # The way you compute how much to truncate is by computing how much room you have minus how much is reserved by specials.
            max_tokens = self._getMaxInputLength() - self.tokenizer.num_special_tokens_to_add(pair=False) - 1
            tokens_so_far = 0
            for word_idx in range(len(subword_ids_per_word)):
                subwords = subword_ids_per_word[word_idx]
                tokens_so_far += len(subwords)
                if tokens_so_far >= max_tokens:
                    excess = tokens_so_far - max_tokens
                    subword_ids_per_word[word_idx] = subwords[:len(subwords)-excess]

                    # Cut away the rest of the words and labels.
                    subword_ids_per_word = subword_ids_per_word[:word_idx+1]
                    heads                = heads[:word_idx+1]
                    deprels              = deprels[:word_idx+1]
                    break

            # TODO: For very very long sentences, you probably shouldn't just truncate the example, but you should use
            #       overflow with a stride. Otherwise you're just losing too much interesting dependency data.
            #       This means the above code should split instead of cut, and all the below code would be repeated per
            #       "chunk" of the example, not just once.
            #       Actually, Supar supports super long sentences. Just don't truncate at all and you're good.

            # Make the 1-based heads (with 0 the root) actually correspond to indices. This dummy has no head and no relation itself.
            subword_ids_per_word.insert(0, [self.tokenizer.unk_token_id])
            heads.insert(0, -100)
            deprels.insert(0, -100)

            # Due to truncation, some of the incoming arrows for the remaining words have no starting point (some heads have disappeared).
            # Set these to -100.
            #   TODO: An alternative strategy is to add something like an EOS/UNK and let the model learn "the head is not in the sequence", SQuAD-v2 unanswerable style.
            maximal_head = len(subword_ids_per_word) - 1
            heads = [(head if head <= maximal_head else -100) for head in heads]

            # The tokeniser expects to add special tokens to a flat list of token IDs. We have a nested list, but that doesn't matter!
            with_special_tokens_added = self.tokenizer.build_inputs_with_special_tokens(subword_ids_per_word)

            # Find how much these specials have offset each word.
            mask    = self.tokenizer.get_special_tokens_mask(with_special_tokens_added, already_has_special_tokens=True)  # I trust the implementation for already_has_special_tokens=False more, but you're not allowed to use it with a fast tokenizer apparently.
            offsets = [sum(mask[:i]) for i in range(len(mask)) if mask[i] != 1]  # Indexable on word indices, so the new head index is head + offsets[head].

            # Insert -100 as the labels of the special tokens, and turn [special, [tokens], special] into [[special], [tokens], [special]].
            actual_words_seen = 0
            word_head_labels   = []
            word_deprel_labels = []
            for i in range(len(with_special_tokens_added)):
                e = with_special_tokens_added[i]
                if not isinstance(e, list):
                    with_special_tokens_added[i] = [e]
                    word_head_labels  .append(-100)
                    word_deprel_labels.append(-100)
                else:
                    head   = heads[actual_words_seen]
                    deprel = deprels[actual_words_seen]

                    word_head_labels  .append(head + offsets[head] if head != -100 else -100)
                    word_deprel_labels.append(deprel)
                    actual_words_seen += 1

            enc = dict()
            enc["words"]       = with_special_tokens_added  # List of lists, because this is the format expected by ArchIt's BaseModelExtended class.
            enc["labels_arcs"] = word_head_labels
            enc["labels_rels"] = word_deprel_labels
            enc["attention_mask"] = [1]*len(with_special_tokens_added)  # word-level attention mask for the head
            return enc

        dataset = dataset.map(preprocess, batched=False)
        dataset = dataset.remove_columns(["tokens", "idx", "text", "lemmas", "upos", "xpos", "feats", "head", "deprel", "deps", "misc"])
        return dataset

    def getCollator(self) -> DataCollator:
        return DataCollatorForDependencyParsing(self.tokenizer)

    def getPredictionsAndReferences(self, eval: EvalPrediction) -> Tuple[Any,Any]:
        return [], []
