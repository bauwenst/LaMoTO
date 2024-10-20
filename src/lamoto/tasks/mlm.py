from dataclasses import dataclass

import datasets
from datasets import IterableDatasetDict, IterableDataset
from transformers import AutoModelForMaskedLM, DataCollatorForLanguageModeling
import torch

from archit.instantiation.basemodels import RobertaBaseModel
from archit.instantiation.heads import MaskedLMHeadConfig
from archit.instantiation.tasks import ForMaskedLM

from ._core import *
from ..measuring.pppl import PPPL_Parameters
from ..util.datasets import PackedDataset
from ..training.auxiliary.hyperparameters import *


@dataclass
class MlmHyperparameters(TaskHyperparameters[MaskedLMHeadConfig]):
    MLM_PROBABILITY: float
    PPPL_PARAMETERS: PPPL_Parameters


SUGGESTED_HYPERPARAMETERS_MLM = MlmHyperparameters(  # Attempt to mimic RoBERTa's hyperparameters.
    SAVE_AS=None,
    WANDB_PROJECT=None,
    traceless=False,

    EXAMPLES_PER_EFFECTIVE_BATCH=8192,
    EXAMPLES_PER_DEVICEBATCH=64,  # Should definitely fit on an A100.
    EFFECTIVE_BATCHES_WARMUP=0.05,
    HARD_STOPPING_CONDITION=AfterNDescents(500_000),
    EXAMPLES_PER_EVALUATION=2**14,  # C4 has a 365k validation split, so 16k isn't that bad. Two times the amount of data processed for one descent.

    TRACK_BEST_MODEL=False,
    EVALS_OF_PATIENCE=None,
    EVAL_VS_SAVE_INTERVALS=Intervals(
        evaluation=EveryNDescents(descents=128),  # 128 batches doesn't seem like a lot, but with that massive batch size it means you only evaluate once every 128 ba * 8192 ex/ba * 512 tk/ex = 0.5 billion tokens seen.
        checkpointing=EveryNMinutes(minutes=30)
    ),

    SEED=69420,
    init_weights=False,
    MODEL_CONFIG_OR_CHECKPOINT="roberta-base",
    archit_basemodel_class=RobertaBaseModel,
    archit_head_config=MaskedLMHeadConfig(),
    load_hf_automodel_if_hf_checkpoint_and_matches_task=True,
    custom_hf_class=None,

    TOKENISER="roberta-base",
    ADD_SPECIAL_TOKENS=True,

    learning_rate=6e-4,
    adamw_decay_rate=0.01,

    MLM_PROBABILITY=0.15,
    PPPL_PARAMETERS=PPPL_Parameters(right_fraction=0.5)
)


class MLM(Task[MaskedLMHeadConfig]):

    def __init__(self, packing: bool=False, use_pppl: bool=False):
        super().__init__(
            task_name="MLM",
            metric_config=MetricSetup(  # This is quite computation-heavy.
                to_compute=["pppl"],
                to_track={
                    "pppl": {"pppl": "PPPL", "nll": "NLL"}
                },
                to_rank=RankingMetricSpec(metric_name="pppl", result_name="nll", higher_is_better=False)
            ) if use_pppl else MetricSetup(  # Uses a cruder estimator for likelihood.
                to_compute=[],
                to_track=dict()
            ),
            archit_class=ForMaskedLM,
            automodel_class=AutoModelForMaskedLM
        )
        self.hyperparameters: MlmHyperparameters = None
        self._use_packing = packing
        self._use_pppl = use_pppl

    def prepareDataset(self, dataset: IterableDatasetDict) -> IterableDatasetDict:
        def preprocess(example):
            return self.tokenizer(example["text"], is_split_into_words=False, add_special_tokens=self.hyperparameters.ADD_SPECIAL_TOKENS, truncation=True, max_length=self._getMaxInputLength())

        if self._use_packing:  # Train split is not tokenised here but in the packer.
            dataset["train"] = PackedDataset(dataset["train"], self.tokenizer, context_length=self._getMaxInputLength())
            if not self._use_pppl:  # Without PPPL, you need to preprocess the validation set yourself for HuggingFace's logit calculation. As is customary, this involves truncation, which is not the case when packing.
                validation_set: IterableDataset = dataset["validation"]
                validation_set = validation_set.map(preprocess, batched=False)
                validation_set = validation_set.remove_columns(["text"])
                dataset["validation"] = validation_set
        else:  # You can just tokenise the whole corpus. Does have truncation to the context length.
            dataset = dataset.map(preprocess, batched=False)
            dataset = dataset.remove_columns(["text"])
        return dataset

    def adjustHyperparameters(self, hp: TaskHyperparameters[MaskedLMHeadConfig]):
        pass

    def getCollator(self) -> DataCollator:
        return DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=self.hyperparameters.MLM_PROBABILITY)

    def sneakyLogitTransform(self, logits, labels):
        return torch.tensor([[1]], device=logits[0].device)

    def getPredictionsAndReferences(self, eval: EvalPrediction) -> Tuple[Any, Any]:
        return None, None

    def train(self, hyperparameters: MlmHyperparameters=SUGGESTED_HYPERPARAMETERS_MLM, model_augmentation: ModelAugmentation=None, resume_from_folder: Path=None):
        return super().train(
            hyperparameters=hyperparameters,
            model_augmentation=model_augmentation,
            resume_from_folder=resume_from_folder
        )


class MLM_C4(MLM):
    def loadDataset(self) -> IterableDatasetDict:
        dataset: datasets.IterableDatasetDict = datasets.load_dataset("allenai/c4", "en", streaming=True)
        return dataset.remove_columns(["timestamp", "url"])


class MLM_SlimPajama(MLM):
    def loadDataset(self) -> IterableDatasetDict:
        dataset: datasets.IterableDatasetDict = datasets.load_dataset("cerebras/SlimPajama-627B", streaming=True, trust_remote_code=True)
        return dataset.remove_columns(["meta"])
