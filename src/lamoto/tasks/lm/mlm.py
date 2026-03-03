from dataclasses import dataclass

from transformers import AutoModelForMaskedLM
from archit.instantiation.heads import MaskedLMHeadConfig
from archit.instantiation.tasks import ForMaskedLM

from ._general import *
from ...measuring.pppl import PPPL_Parameters


@dataclass
class MlmHyperparameters(TaskHyperparameters[MaskedLMHeadConfig]):
    mlm_probability: float
    pppl: PPPL_Parameters


class MLM(TokenPredictionTask[MaskedLMHeadConfig]):

    def _isAutoregressive(self) -> bool:
        return False

    @classmethod
    def _metadata(cls, use_perplexity: bool) -> TaskMetadata:
        return TaskMetadata(
            task_name="MLM",
            text_fields=["text"],
            label_field=[],
            metric_config=MetricSetup(  # This is quite computation-heavy.
                to_compute=["pppl"],
                to_track={
                    "pppl": {"pppl": "PPPL", "nll": "NLL"}
                },
                to_rank=RankingMetricSpec(metric_name="pppl", result_name="nll", higher_is_better=False)
            ) if use_perplexity else MetricSetup(  # Uses a cruder estimator for likelihood.
                to_compute=[],
                to_track=dict()
            ),
            archit_class=ForMaskedLM,
            automodel_class=AutoModelForMaskedLM,
            automodel_kwargs=dict()
        )

    @classmethod
    def getDefaultHyperparameters(cls) -> MlmHyperparameters:
        from ...training.auxiliary.hyperparameters import AfterNDescents, Intervals, EveryNDescents, EveryNMinutes
        from archit.instantiation.basemodels import RobertaBaseModel
        return MlmHyperparameters(  # Attempt to mimic RoBERTa's hyperparameters.
            save_as=None,
            wandb_project=None,
            discard_artifacts=False,
            discard_results=False,
            store_in_hf_cache=False,

            examples_per_effective_batch=8192,
            examples_per_device_batch=64,  # Should definitely fit on an A100.
            effective_batches_warmup=1024,  # This greatly depends on how much time you have for your runs... RoBERTa had 1 million descents of which 10k (1%) warmup, but they had 1024 V100s.
            hard_stopping_condition=AfterNDescents(500_000),
            examples_per_evaluation=2**14,  # C4 has a 365k validation split, so 16k isn't that bad. Two times the amount of data processed for one descent.

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
            gradient_clipping_norm=None,
            gradient_checkpointing_if_possible=True,

            mlm_probability=0.15,
            pppl=PPPL_Parameters(right_fraction=0.5)
        )
