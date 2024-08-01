from dataclasses import dataclass

import datasets
from datasets import IterableDatasetDict
from transformers import AutoModelForMaskedLM, DataCollatorForLanguageModeling

from ._core import *


@dataclass
class MlmHyperparameters(TaskHyperparameters):
    MLM_PROBABILITY: float


SUGGESTED_HYPERPARAMETERS_MLM = MlmHyperparameters(  # Attempt to mimic RoBERTa's hyperparameters.
    SAVE_AS=None,
    WANDB_PROJECT=None,

    EXAMPLES_PER_EFFECTIVE_BATCH=8192,
    EXAMPLES_PER_DEVICEBATCH=64,  # Should definitely fit on an A100.
    EFFECTIVE_BATCHES_WARMUP=0.05,
    HARD_STOPPING_CONDITION=AfterNDescents(500_000),
    EXAMPLES_PER_EVALUATION=2**14,  # C4 has a 365k validation split, so 16k isn't that bad. Two times the amount of data processed for one descent.

    TRACK_BEST_MODEL=False,
    EVALS_OF_PATIENCE=None,
    EVAL_VS_SAVE_INTERVALS=Intervals(
        evaluation=EveryNDescents(descents=128),
        checkpointing=EveryNMinutes(minutes=30)
    ),

    ADD_SPECIAL_TOKENS=True,
    INIT_WEIGHTS=False,
    CHECKPOINT_OR_CONFIG="roberta-base",
    TOKENISER_CHECKPOINT="roberta-base",

    LEARNING_RATE = 6e-4,
    L2_REGULARISATION = 0.01,

    MLM_PROBABILITY=0.15
)


class MLM(Task):  # TODO: Should you use packing for MLM?

    def __init__(self):
        super().__init__(
            task_name="MLM",
            metric_config=MetricSetup(
                to_compute=[],
                to_track=dict()
            ),
            automodel_class=AutoModelForMaskedLM
        )
        self.hyperparameters: MlmHyperparameters = None

    def prepareDataset(self, dataset: IterableDatasetDict) -> IterableDatasetDict:
        def preprocess(example):
            return self.tokenizer(example["text"], is_split_into_words=False, add_special_tokens=self.hyperparameters.ADD_SPECIAL_TOKENS, truncation=True, max_length=self._getMaxInputLength())

        dataset = dataset.map(preprocess, batched=False)
        dataset = dataset.remove_columns(["text"])
        return dataset

    def getCollator(self) -> DataCollator:
        return DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=self.hyperparameters.MLM_PROBABILITY)

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
