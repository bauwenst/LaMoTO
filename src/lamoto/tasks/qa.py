from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from datasets import load_dataset, DatasetDict

import torch
import warnings
from transformers.data.data_collator import DataCollatorMixin
from transformers import DataCollator, EvalPrediction, PreTrainedTokenizerBase

from tktkt.preparation.splitters import Pretokeniser, SplitNextToWhitespace, PunctuationPretokeniser, HyphenMode, PretokeniserSequence
from archit.instantiation.tasks import ForExtractiveAQA, ForExtractiveQA
from archit.instantiation.heads import ExtractiveQAHeadConfig, ExtractiveAQAHeadConfig

from ._core import Task, MetricSetup
from ..util.datasets import imputeTestSplit, DictOfLists, \
    replaceDatasetColumns_OneExampleToOneExample, replaceDatasetColumns_ManyExamplesToManyExamples, ImplicitLabel, \
    TextField, ForeignField, CharacterIndex
from ..measuring import AQA


DefaultSquadPretokeniser = PretokeniserSequence([
    SplitNextToWhitespace(before_not_after=True),
    PunctuationPretokeniser(HyphenMode.EXCLUDED, protect_apostrophes_without_spaces=True)
])


@dataclass
class DataCollatorForQA(DataCollatorMixin):

    tokenizer: PreTrainedTokenizerBase
    max_length: int
    padding: str
    return_tensors: str = "pt"

    def torch_call(self, examples: List[Dict[str,Any]]):
        # Take labels out of the examples beforehand.
        label_names = [name for name in examples[0].keys() if "label" in name]
        labels = {name: [example.pop(name) for example in examples] for name in label_names}

        # Pad the rest (this method is surprisingly lenient about the format of the input)
        batch = self.tokenizer.pad(examples, max_length=self.max_length, padding=self.padding, return_tensors="pt")

        # Format labels as either a tensor or a tuple of tensors.
        if len(labels) == 1:
            _, labels = labels.popitem()
            batch["labels"] = torch.tensor(labels)
        else:  # The fact that this is a tuple is reflected in the signature for SQuAD.sneakyLogitTransform, and in ArchIt's ForExtractiveAQA.computeLoss and matches the output of ExtractiveAQAHead.forward.
            batch["labels"] = tuple(torch.tensor(labels[name]) for name in sorted(labels.keys(), key=lambda name: bool("ua" in name)))  # ua comes after qa.
        return batch


class _SquadTask(Task):

    def __init__(self, with_answerability: bool,
                 context_stride: float=0.5, concatenable_pretoken_generator: Pretokeniser=DefaultSquadPretokeniser):
        """
        :param concatenable_pretoken_generator: Although SQuAD doesn't come pre-tokenised like some other tasks (e.g. PoS), you need
                                   to convert the character indices of start/end in the raw text to token indices in the
                                   split preprocessed text, which could have more characters.
                                   The approximation we make is to first map the character-level labels to pretoken-level
                                   labels, assuming that an answer never starts in the middle of a pretoken. Then, we
                                   tokenise each pretoken and just label all tokens belonging to all pretokens between
                                   the labelled ones as the answer span.
                                   The approximation made is that any characters in the same pretoken as the start/end
                                   characters are also included in the answer span.
                                   Make sure that this splitter does not destroy any characters!
        :param context_stride: Each example is constructed by always taking the full question, and then striding the
                               context (see https://github.com/google-research/bert/issues/27#issuecomment-435265194).
                               That is: you fill the rest of the model's input length (call it R) with context tokens,
                               and if there is more context left, you construct another example that again has the full
                               question but with the context slid over by a certain fraction of R. Repeat until there are
                               no more context tokens.
        """
        super().__init__(
            task_name="SQuAD2" if with_answerability else "SQuAD1",
            text_fields=["question", "context"],
            label_field=[CharacterIndex(ImplicitLabel(lambda ex: ex["answers"]["answer_start"][0] if ex["answers"]["answer_start"] else -1), "context"),  # Why we need this: CharacterIndex to tell us "this indexes into another field" and ImplicitLabel to tell us "this is where it is". That's all you need when e.g. applying typos to the referent.
                         ForeignField(ImplicitLabel(lambda ex: ex["answers"]["text"][0] if ex["answers"]["text"] else ""), TextField("context"))]
                        + with_answerability*[ImplicitLabel(lambda ex: int(len(ex["answers"]["answer_start"]) > 0))],
            metric_config=MetricSetup(
                to_compute=["aqa"],
                to_track={"aqa": {name: name.replace("_", "-") for name in AQA.keys()}}
            ) if with_answerability else MetricSetup(
                to_compute=["qa"],
                to_track={"qa": {"EM": "EM", "F1": "$F_1$"}}
            ),
            archit_class=ForExtractiveAQA if with_answerability else ForExtractiveQA,
            automodel_class=None
        )
        self._v2 = with_answerability
        self._get_pretokens = concatenable_pretoken_generator
        self._stride_fraction = context_stride

    def _loadDataset(self) -> DatasetDict:  # TODO: NewsQA?
        return imputeTestSplit(load_dataset("rajpurkar/squad" + "_v2"*self._v2), column_for_stratification=None, seed=self.hyperparameters.seed)

    def _prepareDataset(self, dataset: DatasetDict) -> DatasetDict:
        def squadTokenConversion(example: Dict[str, Any]) -> Dict[str, Any]:
            """
            About 20x slower per example than the preprocessor for UD PoS tagging because here you have about 400 pretokens
            per example rather than 20. You can offset this with example-level parallelism (since the dataset is not streamed)
            with a 6x speedup, so you end up being 6/20 = 30% ~ 1/3 as fast (5% without parallelism).
            Also, SQuAD v2 is 10x the size of UD (130k vs ~13k) so in total you're looking at 30x the time to preprocess
            the full dataset. Yikes.
            """
            answer = example["answers"]
            is_answerable = len(answer["answer_start"]) > 0

            # Tokenise context and question the same way. Splitting into pretokens is only really needed for the context,
            # but we do it to the question too lest the tokeniser act differently otherwise.
            context = example["context"]
            context_pretokens = self._get_pretokens.split(context)
            assert "".join(context_pretokens) == context
            context_tokens = [self.tokenizer(pretoken, add_special_tokens=False)["input_ids"]
                              for pretoken in context_pretokens]

            question = example["question"].strip()  # The question doesn't need to conserve the amount of characters, and SQuAD has one problematic 25000-token question. https://github.com/huggingface/transformers/issues/12880#issuecomment-888292959
            question_pretokens = self._get_pretokens.split(question)
            assert "".join(question_pretokens) == question
            question_tokens = [self.tokenizer(pretoken, add_special_tokens=False)["input_ids"]
                               for pretoken in question_pretokens]

            if is_answerable:
                # Step 1: Convert from character labels to pretoken labels.
                char_idx_answer_start = answer["answer_start"][0]
                char_idx_answer_end   = answer["answer_start"][0] + len(answer["text"][0]) - 1  # This is an INCLUSIVE index.

                pretoken_idx_answer_start = 0
                char_cursor = 0
                while not(char_cursor > char_idx_answer_start):  # Assuming the character is anywhere in the middle of a pretoken, and the cursor has NOT been visited yet, we want the pretoken that made the cursor go past the character.
                    char_cursor += len(context_pretokens[pretoken_idx_answer_start])
                    pretoken_idx_answer_start += 1
                pretoken_idx_answer_start -= 1

                pretoken_idx_answer_end = 0
                char_cursor = 0
                while not(char_cursor > char_idx_answer_end):
                    char_cursor += len(context_pretokens[pretoken_idx_answer_end])
                    pretoken_idx_answer_end += 1
                pretoken_idx_answer_end -= 1

                # Step 2: Convert from pretoken labels to token labels.
                token_idx_answer_start = 0
                pretoken_cursor = 0
                while not(pretoken_cursor == pretoken_idx_answer_start):  # Example: if answer starts in pretoken 2, and you have tokens [[0,1], [2,3,4,5,7], [8,9,10]], you want to find token 8.
                    token_idx_answer_start += len(context_tokens[pretoken_cursor])
                    pretoken_cursor += 1

                token_idx_answer_end = 0
                pretoken_cursor = 0
                while not(pretoken_cursor > pretoken_idx_answer_end):
                    token_idx_answer_end += len(context_tokens[pretoken_cursor])
                    pretoken_cursor += 1
                token_idx_answer_end -= 1
            else:
                token_idx_answer_start = 0
                token_idx_answer_end   = 0

            return {
                "question_ids": sum(question_tokens, []),
                "context_ids":  sum(context_tokens,  []),
                "answer_start_in_context": token_idx_answer_start,
                "answer_end_in_context":   token_idx_answer_end,
                "answerable": is_answerable
            }

        def squadStriding(key_to_values: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
            """
            Has to manage two things:
              1. Given a certain layout for the special tokens, compute how many examples you will need to represent
                 the (question,context) pair given a stride, i.e. an amount of tokens you drop from the context with
                 each new example.
              2. Update the start_idx and end_idx as follows:
                      1. The true index equals len(specials before context) + len(question) + index.
                      2. If either or both start_idx and end_idx are not inside the context, re-flag this question as unanswerable
                         for this iteration of the stride.
            """
            model_window = self._getMaxInputLength()
            output_examples = DictOfLists(
                list(key_to_values.keys()) + ["input_ids", "attention_mask", "context_mask", "labels_qa"] + ["labels_ua"]*self._v2,
                append_none_for_missing_keys=True
            )

            # Batch loop
            for question_ids, context_ids, raw_start_idx, raw_end_idx, answer_exists in zip(
                    key_to_values["question_ids"], key_to_values["context_ids"],
                    key_to_values["answer_start_in_context"], key_to_values["answer_end_in_context"], key_to_values["answerable"]
            ):
                n_specials = self.tokenizer.num_special_tokens_to_add(pair=True)
                n_question = len(question_ids)
                n_context  = len(context_ids)
                n_context_per_window = model_window - (n_specials + n_question)
                stride = int(self._stride_fraction * n_context_per_window)

                if n_context_per_window <= 0:
                    warnings.warn(f"Question is so long ({n_question} tokens, {n_question + n_specials} with specials) that the model's context length ({model_window}) cannot hold any context tokens! Skipping...")
                    print(self.tokenizer.decode(question_ids))
                    continue

                # Striding loop
                n_pre_context = n_question + n_specials - 1  # FIXME: This is wrong in general, but it works for RoBERTa.

                shift = 0
                while n_context > 0:
                    # Build IDs
                    window_ids = context_ids[shift:shift+n_context_per_window]
                    input_ids = self.tokenizer.build_inputs_with_special_tokens(question_ids, window_ids)

                    # Update answer indices
                    start_idx = n_pre_context + raw_start_idx - shift  # [CLS] n_question [SEP] [SEP] context...
                    end_idx   = n_pre_context + raw_end_idx - shift

                    # Is it still answerable?
                    stride_is_answerable = answer_exists and start_idx >= n_pre_context and end_idx < model_window
                    if not stride_is_answerable:
                        start_idx = 0
                        end_idx   = 0

                    # Finish building example
                    output_examples.append({
                        "input_ids": input_ids,
                        "attention_mask": [1]*len(input_ids),
                        "context_mask": [0]*n_pre_context + [1]*len(window_ids) + [0]*(len(input_ids) - n_pre_context - len(window_ids)),
                        "labels_qa": [start_idx, end_idx]
                    } | ({
                        "labels_ua": int(stride_is_answerable),
                    } if self._v2 else dict()))
                    n_context -= n_context_per_window if shift == 0 else stride
                    shift     += stride

            return output_examples.toDict()

        dataset = replaceDatasetColumns_OneExampleToOneExample(dataset, squadTokenConversion)  # 50/s for 1 thread, 200/s for 4, 300/s for 8 (same for 6). Edit: ooo, if you keep spaces attached to the pretokens rather than isolating them, you can get 500/s at 6. That's more like it!
        dataset = replaceDatasetColumns_ManyExamplesToManyExamples(dataset, squadStriding)
        return dataset

    def sneakyLogitTransform(self, logits: Tuple[torch.Tensor,torch.Tensor], labels: Tuple[torch.Tensor,torch.Tensor]) -> torch.Tensor:
        # Note: this is not hooked up to the metric because unlike in DP, the metric is autonomous since it requires access to input_ids and context_mask.
        return torch.tensor([[1]], device=logits[0].device)

    def adjustHyperparameters(self, hp):  # AQA has its labels pre-set to 2.
        pass

    def getCollator(self) -> DataCollator:
        return DataCollatorForQA(self.tokenizer, padding="longest", max_length=self._getMaxInputLength())

    def getPredictionsAndReferences(self, eval: EvalPrediction) -> Tuple[Any,Any]:
        return [], []


class SQuADv1(_SquadTask):
    """
    Extractive question answering.
    https://aclanthology.org/D16-1264.pdf
    """
    def __init__(self, context_stride: float=0.5, concatenable_pretoken_generator: Pretokeniser=DefaultSquadPretokeniser):
        super().__init__(with_answerability=False,
                         context_stride=context_stride, concatenable_pretoken_generator=concatenable_pretoken_generator)


class SQuADv2(_SquadTask):
    """
    Extractive answerability and question answering.
    https://aclanthology.org/P18-2124.pdf
    """
    def __init__(self, context_stride: float=0.5, concatenable_pretoken_generator: Pretokeniser=DefaultSquadPretokeniser):
        super().__init__(with_answerability=True,
                         context_stride=context_stride, concatenable_pretoken_generator=concatenable_pretoken_generator)
