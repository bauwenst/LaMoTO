"""
Tuning framework in which many models are trained for the same task with various hyperparameter sets.
"""
from dataclasses import dataclass
from typing import Optional, List

import numpy.random as npr
from tktkt.util.printing import dprint, pluralise

from .auxiliary.hyperparameters import TaskHyperparameters, AfterNExamples, EveryNExamplesOrOncePerEpoch
from .core import log
from ..tasks._core import Task, RankingMetricSpec


@dataclass
class MetaHyperparameters:
    meta_seed: int
    n_grid_samples: int

    max_examples_phase_1: int
    max_evals_phase_1: int

    max_examples_phase_2: int
    max_evals_phase_2: int

    rank_by: RankingMetricSpec


class TaskTuner:
    """
    Samples values for certain hyperparameters in their given domains, and knows how to alter the training procedure
    when those supported hyperparameters change.

        warmups               = [50, 100, 500, 1000]
        effective_batch_sizes = [16, 32, 64, 128, 256, 512]
        learning_rates        = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
        decay_rates           = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    """

    def __init__(self, warmup_steps: Optional[List[int]],
                 effective_batch_sizes: Optional[List[int]],
                 learning_rates: Optional[List[float]],
                 adamw_decay_rates: Optional[List[float]]):
        self._warmup_steps_grid         = warmup_steps or []
        self._effective_batch_size_grid = effective_batch_sizes or []
        self._learning_rates            = learning_rates
        self._decay_rates               = adamw_decay_rates

    @dataclass
    class _HyperparameterGridSample:
        wu: int
        bs: int
        lr: int
        dr: int

    def _setSample(self, hp: TaskHyperparameters, sample: _HyperparameterGridSample):
        """
        Adapter between TaskHyperparameters fields and _HyperparameterGridSample fields.
        """
        hp.EFFECTIVE_BATCHES_WARMUP     = sample.wu
        hp.EXAMPLES_PER_EFFECTIVE_BATCH = sample.bs
        hp.learning_rate                = sample.lr
        hp.adamw_decay_rate             = sample.dr

    def tune(self, task: Task, hp: TaskHyperparameters, meta: MetaHyperparameters):
        hp = hp.copy()

        # Sanity checks and imputations
        assert meta.n_grid_samples >= 1, "At least one hyperparameter sample must be taken."
        assert isinstance(hp.MODEL_CONFIG_OR_CHECKPOINT, str), "Can only tune starting from a pre-trained checkpoint."
        hp.init_weights = True

        # Find best hp changes
        original_stopping_condition = hp.HARD_STOPPING_CONDITION
        original_traceless          = hp.traceless
        best_sample = self._phase1(task, hp, meta)

        # Apply best hp changes
        hp.HARD_STOPPING_CONDITION = original_stopping_condition
        hp.traceless               = original_traceless
        return self._phase2(task, hp, meta, best_sample)

    def _phase1(self, task: Task, hp: TaskHyperparameters, meta: MetaHyperparameters) -> _HyperparameterGridSample:
        """
        Try to find the optimal set of hyperparameters from n grid samples.
        """
        # Hyperparameter setup
        hp.traceless = True
        hp.TRACK_BEST_MODEL = True
        hp.HARD_STOPPING_CONDITION = AfterNExamples(meta.max_examples_phase_1)  # Independent of batch size.
        hp.EVAL_VS_SAVE_INTERVALS.evaluation = EveryNExamplesOrOncePerEpoch(meta.max_examples_phase_1 // meta.max_evals_phase_1)
        hp.EXAMPLES_PER_EVALUATION = None  # Inference should be fast enough to process anything in GLUE quickly enough.

        # Grid setup
        rng = npr.default_rng(hp.SEED + meta.meta_seed)
        samples = zip(
            rng.choice(self._warmup_steps_grid,         size=meta.n_grid_samples).tolist(),
            rng.choice(self._effective_batch_size_grid, size=meta.n_grid_samples).tolist(),
            rng.choice(self._learning_rates,            size=meta.n_grid_samples).tolist(),
            rng.choice(self._decay_rates,               size=meta.n_grid_samples).tolist()
        )

        # Grid search
        ranking_metric_name = "eval_" + meta.rank_by.fullName()
        best_ranking_value = -float("inf") if meta.rank_by.higher_is_better else float("inf")
        best_sample = None
        for n, (wu, bs, lr, dr) in enumerate(samples):
            grid_sample = TaskTuner._HyperparameterGridSample(wu=wu, bs=bs, lr=lr, dr=dr)
            self._setSample(hp, grid_sample)

            log(f"\nStarting short tuning for {n+1}th hyperparameter set:", grid_sample)
            results = task.train(hp)
            log(f"Finished short tuning for {n+1}th hyperparameter set:", grid_sample)
            print("Results:")
            dprint(results, indent=1)

            # Ranking
            if ranking_metric_name not in results:
                log(f"WARNING: Missing ranking metric {ranking_metric_name}. Cannot rank this hyperparameter set.")
            else:
                new_ranking_value = results[ranking_metric_name]
                if meta.rank_by.higher_is_better and new_ranking_value > best_ranking_value or \
                    not meta.rank_by.higher_is_better and new_ranking_value < best_ranking_value:
                    best_ranking_value = new_ranking_value
                    best_sample = grid_sample
            print("=" * 50)

        if best_sample is None:
            raise RuntimeError(f"No hyperparameter sets resulted in the ranking metric '{ranking_metric_name}'.")

        log(f"Best hyperparameters out of {pluralise(meta.n_grid_samples, 'sample')} as measured by {ranking_metric_name}:", best_sample)
        return best_sample

    def _phase2(self, task: Task, hp: TaskHyperparameters, meta: MetaHyperparameters, best_sample: _HyperparameterGridSample):
        """
        Use the best hyperparameters you found and run until you can't.
        """
        hp.HARD_STOPPING_CONDITION = AfterNExamples(meta.max_examples_phase_2)
        hp.EVAL_VS_SAVE_INTERVALS.evaluation = EveryNExamplesOrOncePerEpoch(meta.max_examples_phase_2 // meta.max_evals_phase_2)

        self._setSample(hp, best_sample)
        log("Starting long tuning for best hyperparameters:", best_sample)
        task.metric_config.to_rank = meta.rank_by
        results = task.train(hp)
        log("Finished long tuning for best hyperparameters:", best_sample)
        print("Results:")
        dprint(results, indent=1)
        return results
