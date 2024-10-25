# Types
from dataclasses import dataclass

# ML libs
import datasets
from datasets import IterableDatasetDict
from transformers import \
    DataCollatorForLanguageModeling, \
    AutoModelForCausalLM

from archit.instantiation.heads import CausalLMHeadConfig
from archit.instantiation.tasks import ForCausalLM
from archit.instantiation.basemodels import GPT2BaseModel

# Relative
from ._core import *
from ..measuring.ppl import PPL_Parameters
from ..training.auxiliary.hyperparameters import *
from ..util.datasets import PackedDataset


@dataclass
class ClmHyperparameters(TaskHyperparameters[CausalLMHeadConfig]):
    PPL: PPL_Parameters  # Which fraction of the model's context length we stride in the perplexity function. The complement of this is the amount of context the first token of the second chunk of an example sees. 1/contextlength is slowest but gives actual perplexity, whilst 1.0 is fastest but means that long examples act like multiple independent examples.


SUGGESTED_HYPERPARAMETERS_CLM = ClmHyperparameters(
    SAVE_AS=None,
    WANDB_PROJECT=None,
    traceless=False,
    store_in_hf_cache=False,

    EXAMPLES_PER_EFFECTIVE_BATCH = 512,   # From the OpenAI GPT-2 paper.
    EXAMPLES_PER_DEVICEBATCH = 64,  # Used to fit on an A100, but recently got an error saying 80 GiB got filled
    EFFECTIVE_BATCHES_WARMUP=0.1,
    HARD_STOPPING_CONDITION=AfterNPackedTokens(10_000_000_000, tokens_per_packed_example=1024),  # From GEITje.

    EXAMPLES_PER_EVALUATION = 2**14,

    TRACK_BEST_MODEL=False,
    EVALS_OF_PATIENCE=None,
    EVAL_VS_SAVE_INTERVALS=Intervals(
        evaluation=EveryNDescents(descents=128),
        checkpointing=EveryNMinutes(minutes=30)
    ),

    SEED=69420,
    init_weights=False,
    MODEL_CONFIG_OR_CHECKPOINT="openai-community/gpt2",
    archit_basemodel_class=GPT2BaseModel,
    archit_head_config=CausalLMHeadConfig(),
    load_hf_automodel_if_hf_checkpoint_and_matches_task=True,
    custom_hf_class=None,

    TOKENISER="openai-community/gpt2",
    ADD_SPECIAL_TOKENS=False,

    learning_rate=2e-5,
    adamw_decay_rate=0.01,

    PPL = PPL_Parameters(stride_fraction=1/8)
)


class CLM(Task[CausalLMHeadConfig]):

    def __init__(self):
        super().__init__(
            task_name="CLM",
            metric_config=MetricSetup(
                to_compute=["ppl"],
                to_track={
                    "ppl": {"ppl": "PPL", "nll": "NLL"}
                },
                to_rank=RankingMetricSpec(metric_name="ppl", result_name="nll", higher_is_better=False)
            ),
            archit_class=ForCausalLM,
            automodel_class=AutoModelForCausalLM
        )

    def _prepareDataset(self, dataset: IterableDatasetDict) -> IterableDatasetDict:
        dataset["train"] = PackedDataset(dataset["train"], self.tokenizer, context_length=self._getMaxInputLength())
        # Note: the validation dataset in CLM is left untokenised. That's because (1) it is an IterableDataset and
        # hence can't be preprocessed anyway -- every evaluation it is re-tokenised -- and (2) the dataset is only
        # used by the PPL metric, which does tokenisation itself.
        # dataset["validation"] = PackedDataset(dataset["validation"], self.tokenizer, self._getMaxInputLength())
        return dataset

    def adjustHyperparameters(self, hp: TaskHyperparameters[CausalLMHeadConfig]):
        pass

    def getCollator(self) -> DataCollator:
        return DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

    def getPredictionsAndReferences(self, eval: EvalPrediction) -> Tuple[Any, Any]:
        return None, None


class PretrainingC4(CLM):

    def _loadDataset(self) -> IterableDatasetDict:
        return datasets.load_dataset("allenai/c4", "en", streaming=True)
