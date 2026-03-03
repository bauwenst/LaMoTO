from dataclasses import dataclass

from transformers import AutoModelForCausalLM
from archit.instantiation.heads import CausalLMHeadConfig
from archit.instantiation.tasks import ForCausalLM

from ._general import *
from ...measuring.ppl import PPL_Parameters


@dataclass
class ClmHyperparameters(TaskHyperparameters[CausalLMHeadConfig]):
    ppl: PPL_Parameters  # Which fraction of the model's context length we stride in the perplexity function. The complement of this is the amount of context the first token of the second chunk of an example sees. 1/contextlength is slowest but gives actual perplexity, whilst 1.0 is fastest but means that long examples act like multiple independent examples.


class CLM(TokenPredictionTask[CausalLMHeadConfig]):

    def _isAutoregressive(self) -> bool:
        return True

    @classmethod
    def _metadata(cls, use_perplexity: bool) -> TaskMetadata:
        return TaskMetadata(
            task_name="CLM",
            text_fields=["text"],
            label_field=[],
            metric_config=MetricSetup(
                to_compute=["ppl"],
                to_track={
                    "ppl": {"ppl": "PPL", "nll": "NLL"}
                },
                to_rank=RankingMetricSpec(metric_name="ppl", result_name="nll", higher_is_better=False)
            ),
            archit_class=ForCausalLM,
            automodel_class=AutoModelForCausalLM,
            automodel_kwargs=dict()
        )

    @classmethod
    def getDefaultHyperparameters(cls) -> ClmHyperparameters:
        from ...training.auxiliary.hyperparameters import AfterNPackedTokens, Intervals, EveryNDescents, EveryNMinutes
        from archit.instantiation.basemodels import GPT2BaseModel
        return ClmHyperparameters(
            save_as=None,
            wandb_project=None,
            discard_artifacts=False,
            discard_results=False,
            store_in_hf_cache=False,

            examples_per_effective_batch= 512,   # From the OpenAI GPT-2 paper.
            examples_per_device_batch= 64,  # Used to fit on an A100, but recently got an error saying 80 GiB got filled
            effective_batches_warmup=0.1,
            hard_stopping_condition=AfterNPackedTokens(total_tokens=10_000_000_000, max_context_length=1024),  # From GEITje.

            examples_per_evaluation=2 ** 14,

            track_best_checkpoint=False,
            rank_checkpoints_using_loss=False,
            evals_of_patience=None,
            eval_vs_save_intervals=Intervals(
                evaluation=EveryNDescents(descents=128),
                checkpointing=EveryNMinutes(minutes=30)
            ),

            seed=69420,
            init_weights=False,
            model_config_or_checkpoint="openai-community/gpt2",
            archit_basemodel_class=GPT2BaseModel,
            archit_head_config=CausalLMHeadConfig(),
            load_hf_automodel_if_hf_checkpoint_and_matches_task=True,
            custom_hf_class=None,

            tokeniser="openai-community/gpt2",
            add_special_tokens=False,

            learning_rate=2e-5,
            adamw_decay_rate=0.01,
            gradient_clipping_norm=None,
            gradient_checkpointing_if_possible=True,

            ppl= PPL_Parameters(stride_fraction=1/8)
        )
