# Types
from dataclasses import dataclass
from typing import Iterable, Tuple, Any

# ML libs
import datasets
from datasets import DatasetDict, IterableDatasetDict
from transformers import \
    DataCollatorForLanguageModeling, \
    AutoModelForCausalLM, \
    PreTrainedTokenizerBase, DataCollator, EvalPrediction

# Relative
from ._core import *
from ..measuring.ppl import PPL_Parameters
from ..trainer.hyperparameters import Intervals


@dataclass
class ClmHyperparameters(TaskHyperparameters):
    PPL: PPL_Parameters  # Which fraction of the model's context length we stride in the perplexity function. The complement of this is the amount of context the first token of the second chunk of an example sees. 1/contextlength is slowest but gives actual perplexity, whilst 1.0 is fastest but means that long examples act like multiple independent examples.


SUGGESTED_HYPERPARAMETERS_CLM = ClmHyperparameters(
    SAVE_AS=None,
    WANDB_PROJECT=None,

    EXAMPLES_PER_EFFECTIVE_BATCH = 512,   # From the OpenAI GPT-2 paper.
    EXAMPLES_PER_DEVICEBATCH = 64,
    EFFECTIVE_BATCHES_WARMUP=0.1,
    HARD_STOPPING_CONDITION=AfterNTokens(10_000_000_000, tokens_per_packed_example=1024, effective_batch_size=512),  # From GEITje.

    EXAMPLES_PER_EVALUATION = 2**14,

    TRACK_BEST_MODEL=False,
    EVALS_OF_PATIENCE=None,
    EVAL_VS_SAVE_INTERVALS=Intervals(
        evaluation=EveryNDescents(descents=128),
        checkpointing=EveryNMinutes(minutes=30)
    ),

    ADD_SPECIAL_TOKENS=False,
    INIT_WEIGHTS=False,
    CHECKPOINT_OR_CONFIG="openai-community/gpt2",
    TOKENISER_CHECKPOINT="openai-community/gpt2",

    LEARNING_RATE = 2e-5,
    L2_REGULARISATION = 0.01,

    PPL = PPL_Parameters(stride_fraction=1/8)
)

# Timeout configuration (I tested this and it works, but it sure is a weird use of Python imports... https://github.com/huggingface/datasets/issues/6172#issuecomment-1794876229)
datasets.config.STREAMING_READ_RETRY_INTERVAL = 60   # Seconds between retries; ideally this would work with exponential backoff, but it doesn't, because... HuggingFace engineers.
datasets.config.STREAMING_READ_MAX_RETRIES    = 120  # Retry for up to 2 hours.


def packedDatasetGenerator(dataset: Iterable, tokenizer: PreTrainedTokenizerBase, context_length: int, key='text'):
    """Concatenate ("pack") samples from a dataset into tokenized chunks of `context_length`.

    Used for efficient training of causal models without padding. No special measures are taken
    to disallow a sequence attending to a previous sequence. The model is left to learn the
    unrelatedness of sequences from the presence of the start- and end-of-sequence-tokens
    between the samples, following a similar convention from GPT-3 and T5.
    See https://github.com/huggingface/transformers/issues/17726 for a feature request for
    HuggingFace Transformers.

    The incomplete final chunk at the end of the dataset is discarded.

    Source: Edwin Rijgersberg's script to train GEITje (https://github.com/Rijgersberg/GEITje)

    :param dataset: Dataset of samples (iterable of dict-like, e.g. Hugging Face dataset)
    :param tokenizer: Callable that tokenizes the samples (e.g. Hugging Face tokenizer)
    :param context_length: number of tokens in packed sequences
    :param key: key of the text field in the sample. Defaults to 'text'
    :yield: dicts of packed input_ids, attention_masks and (self-supervised) labels
    """
    cache = []
    for row in dataset:  # TODO: <--- This can be affected by a network error. If HF doesn't fix their retries, wrap this. https://github.com/huggingface/datasets/pull/6844
        # print("Loaded row from dataset")

        # Add extra IDs to cache
        new_ids = tokenizer(row[key], max_length=1_000_000, truncation=True)['input_ids']  # max_length=None will give a warning because it assumes tokeniser output is passed to the model without further processing.
        if not new_ids[-1] == tokenizer.eos_token_id:  # You need an EOS between examples.
            new_ids.append(tokenizer.eos_token_id)
        cache.extend(new_ids)

        # If the cache is now bigger than the fixed length we want to output: empty it in chunks of that size until
        # the cache needs more data.
        while len(cache) >= context_length:
            chunk = cache[:context_length]
            yield {"input_ids": chunk,
                   "labels": chunk,  # For CLM, the labels ARE the inputs.
                   "attention_mask": [1] * context_length}
            cache = cache[context_length:]


class CLM(Task):

    def __init__(self):
        super().__init__(
            task_name="CLM",
            metric_config=MetricSetup(
                to_compute=["ppl"],
                to_track={
                    "ppl": {"ppl": "PPL", "nll": "NLL"}
                }
            ),
            automodel_class=AutoModelForCausalLM
        )

    def prepareDataset(self, dataset: IterableDatasetDict) -> IterableDatasetDict:
        dataset["train"] = datasets.IterableDataset.from_generator(
            generator=packedDatasetGenerator,
            gen_kwargs={"dataset": dataset["train"],
                        "tokenizer": self.tokenizer,
                        "context_length": self._getMaxInputLength()})
        # dataset["validation"] = datasets.IterableDataset.from_generator(
        #     generator=packedDatasetGenerator,
        #     gen_kwargs={"dataset": dataset["validation"],
        #                 "tokenizer": tokenizer,
        #                 "context_length": model.config.max_position_embeddings})
        return dataset

    def getCollator(self) -> DataCollator:
        return DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

    def getPredictionsAndReferences(self, eval: EvalPrediction) -> Tuple[Any, Any]:
        return None, None


class PretrainingC4(CLM):

    def loadDataset(self) -> IterableDatasetDict:
        return datasets.load_dataset("allenai/c4", "en", streaming=True)
