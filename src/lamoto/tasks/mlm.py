from abc import abstractmethod
from dataclasses import dataclass
from datasets import IterableDatasetDict, IterableDataset

import torch
import datasets
from transformers import AutoModelForMaskedLM, DataCollatorForLanguageModeling

from archit.instantiation.basemodels import RobertaBaseModel
from archit.instantiation.heads import MaskedLMHeadConfig
from archit.instantiation.tasks import ForMaskedLM

from ._core import *
from ..measuring.pppl import PPPL_Parameters
from ..util.datasets import PackedDataset
from ..training.auxiliary.hyperparameters import *


@dataclass
class MlmHyperparameters(TaskHyperparameters[MaskedLMHeadConfig]):
    mlm_probability: float
    pppl: PPPL_Parameters


SUGGESTED_HYPERPARAMETERS_MLM = MlmHyperparameters(  # Attempt to mimic RoBERTa's hyperparameters.
    save_as=None,
    wandb_project=None,
    traceless=False,
    store_in_hf_cache=False,

    examples_per_effective_batch=8192,
    examples_per_device_batch=64,  # Should definitely fit on an A100.
    effective_batches_warmup=0.05,
    hard_stopping_condition=AfterNDescents(500_000),
    examples_per_evaluation=2 ** 14,  # C4 has a 365k validation split, so 16k isn't that bad. Two times the amount of data processed for one descent.

    track_best_checkpoint=False,
    rank_checkpoints_using_loss=False,
    evals_of_patience=None,
    eval_vs_save_intervals=Intervals(
        evaluation=EveryNDescents(descents=128),  # 128 batches doesn't seem like a lot, but with that massive batch size it means you only evaluate once every 128 ba * 8192 ex/ba * 512 tk/ex = 0.5 billion tokens seen.
        checkpointing=EveryNMinutes(minutes=30)
    ),

    seed=69420,
    init_weights=False,
    model_config_or_checkpoint="roberta-base",
    archit_basemodel_class=RobertaBaseModel,
    archit_head_config=MaskedLMHeadConfig(),
    load_hf_automodel_if_hf_checkpoint_and_matches_task=True,
    custom_hf_class=None,

    tokeniser="roberta-base",
    add_special_tokens=True,

    learning_rate=6e-4,
    adamw_decay_rate=0.01,

    mlm_probability=0.15,
    pppl=PPPL_Parameters(right_fraction=0.5)
)


class MLM(Task[MaskedLMHeadConfig]):

    def __init__(self, packing: bool=False, drop_train_examples: int=0, use_pppl: bool=False):
        """
        :param packing: Whether to concatenate tokens from multiple dataset examples to fill up the model's context length.
        :param use_pppl: Whether to evaluate with PPPL. Note that this takes way more time than the usual evaluation, which is
                         just computing NLL on masked evaluation examples.
        :param drop_train_examples: How many training examples to advance by before starting training.
                                    Note: a *training example* is not the same as a *dataset example*. Training examples
                                    are given to models, and are the unit we measure batch size in. Dataset examples can
                                    be much larger (e.g. for datasets of long articles) or much smaller (e.g. tweets)
                                    than training examples. Since you only know how many batches have been trained for,
                                    rather than how many examples have been used from the underlying dataset, we skip
                                    in units of train examples.
        """
        super().__init__(
            task_name="MLM",
            text_fields=["text"],
            label_field=[],
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
        self._drop_train = max(0,drop_train_examples)
        self._use_packing = packing
        self._use_pppl = use_pppl

    @abstractmethod
    def _loadIterableDataset(self) -> IterableDatasetDict:
        pass

    def _loadDataset(self) -> IterableDatasetDict:
        return self._loadIterableDataset()

    def _prepareDataset(self, dataset: IterableDatasetDict) -> IterableDatasetDict:
        def preprocess(example):
            return self.tokenizer(example["text"], is_split_into_words=False, add_special_tokens=self.hyperparameters.add_special_tokens, truncation=True, max_length=self._getMaxInputLength())

        if self._use_packing:  # Train split is not tokenised here but in the packer.
            dataset["train"] = PackedDataset(dataset["train"], self.tokenizer, context_length=self._getMaxInputLength())
            if not self._use_pppl:  # Without PPPL, you need to tokenise the validation set yourself for HuggingFace's logit calculation. As is customary, this involves truncation (i.e. data is lost for examples that are too long), which is not the case when packing.
                validation_set: IterableDataset = dataset["validation"]
                validation_set = validation_set.map(preprocess, batched=False)
                validation_set = validation_set.remove_columns(["text"])
                dataset["validation"] = validation_set
        else:  # You can just tokenise the whole corpus. Does have truncation to the context length, as per above.
            dataset = dataset.map(preprocess, batched=False)
            dataset = dataset.remove_columns(["text"])

        if self._drop_train:  # Drop AFTER packing.
            dataset["train"] = dataset["train"].skip(self._drop_train)

        return dataset

    def adjustHyperparameters(self, hp: TaskHyperparameters[MaskedLMHeadConfig]):
        pass

    def getCollator(self) -> DataCollator:
        return DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=self.hyperparameters.mlm_probability)

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
    def _loadIterableDataset(self) -> IterableDatasetDict:
        dataset: IterableDatasetDict = datasets.load_dataset("allenai/c4", "en", streaming=True)
        return dataset.remove_columns(["timestamp", "url"])


class MLM_SlimPajama(MLM):
    def _loadIterableDataset(self) -> IterableDatasetDict:
        dataset: IterableDatasetDict = datasets.load_dataset("cerebras/SlimPajama-627B", streaming=True, trust_remote_code=True)
        return dataset.remove_columns(["meta"])
