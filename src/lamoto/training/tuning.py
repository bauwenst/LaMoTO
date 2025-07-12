"""
Tuning framework in which many models are trained for the same task with various hyperparameter sets.
"""
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple, Iterable, Union, Set
from pathlib import Path

import json
import itertools
from copy import deepcopy
from math import prod
import numpy.random as npr

from tktkt.util.printing import dprint, pluralise, ordinal
from tktkt.util.iterables import keepFirst, take

from ..tasks._core import Task, RankingMetricSpec, ModelAugmentation
from .auxiliary.hyperparameters import TaskHyperparameters, AfterNExamples, EveryNExamplesOrOncePerEpoch, getDefaultHyperparameters
from .training import log, TaskTrainer, LamotoPaths, TrainerCallback


@dataclass
class MetaHyperparameters:
    meta_seed: int
    n_grid_samples: int

    max_examples_phase_1: int
    minmax_evals_phase_1: int  # When the maximum amount of examples is reached, this is the minimum amount of evals that have been done. (It can be higher when epochs are smaller than this.)

    max_examples_phase_2: int
    minmax_evals_phase_2: int

    rank_by: Optional[RankingMetricSpec] = None  # If None, the task's default ranking metric will be used in tuning.

    def copy(self) -> "MetaHyperparameters":
        return deepcopy(self)


@dataclass
class HyperparameterGrid:
    """
    For the GRaMPa paper, I used:
        warmup_steps          = [50, 100, 500, 1000]
        effective_batch_sizes = [16, 32, 64, 128, 256, 512]
        learning_rates        = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
        adamw_decay_rates     = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]

    From having done many finetuning experiments on 6-layer models, the grid can probably be reduced to:
        warmup_steps          = [50, 100]
        effective_batch_sizes = [16, 32]
        learning_rates        = [5e-5, 1e-4, 5e-4]
        adamw_decay_rates     = [0.05]

    Where possibly you use even smaller warmup steps and effective batch sizes.
    """
    seeds: Optional[List[int]] = None,
    warmup_steps: Optional[List[int]] = None,
    effective_batch_sizes: Optional[List[int]] = None,
    learning_rates: Optional[List[float]] = None,
    adamw_decay_rates: Optional[List[float]] = None,

    @dataclass(frozen=True)  # Allows hashing
    class Sample:
        seed: int
        wu: int
        bs: int
        lr: float
        dr: float

        def modifyHyperparameters(self, hp: TaskHyperparameters):
            """
            Adapter between TaskHyperparameters fields and _HyperparameterGridSample fields.
            """
            hp.seed                         = self.seed
            hp.effective_batches_warmup     = self.wu
            hp.examples_per_effective_batch = self.bs
            hp.learning_rate                = self.lr
            hp.adamw_decay_rate             = self.dr

        @staticmethod
        def fromHyperparameters(hp: TaskHyperparameters) -> "HyperparameterGrid.Sample":
            return HyperparameterGrid.Sample(
                seed=hp.seed,
                wu=hp.effective_batches_warmup,
                bs=hp.examples_per_effective_batch,
                lr=hp.learning_rate,
                dr=hp.adamw_decay_rate
            )

    def getFullGrid(self, defaults: TaskHyperparameters) -> List[List[Union[int, float]]]:
        default_sample = HyperparameterGrid.Sample.fromHyperparameters(defaults)
        return [
            self.seeds                 or [default_sample.seed],
            self.warmup_steps          or [default_sample.wu],
            self.effective_batch_sizes or [default_sample.bs],
            self.learning_rates        or [default_sample.lr],
            self.adamw_decay_rates     or [default_sample.dr]
        ]

    def domainSize(self) -> int:
        return prod(map(len, self.getFullGrid(defaults=getDefaultHyperparameters())))

    def enumerate(self, defaults: TaskHyperparameters) -> Iterable["HyperparameterGrid.Sample"]:
        for seed, wu, bs, lr, dr in itertools.product(*self.getFullGrid(defaults)):
            yield HyperparameterGrid.Sample(seed=seed, wu=wu, bs=bs, lr=lr, dr=dr)

    def sample(self, sampling_seed: int, n_samples: int, defaults: TaskHyperparameters, excluded: Set["HyperparameterGrid.Sample"]=None) -> Iterable["HyperparameterGrid.Sample"]:
        """
        Randomly samples the grid.
        :param excluded: Samples to not generate nor count towards the total amount of samples. Note that in case
                |domain \ excluded| < n_samples
            this method will internally generate the entire domain, output domain \ excluded, and then just quit with a
            warning, rather than quitting only when n_samples unique samples have been found (which is impossible).
        """
        excluded = excluded or set()

        def generateUniqueSamplesNotExcluded():
            for seed, wu, bs, lr, dr in sampleGridWithoutReplacement(npr.default_rng(sampling_seed), None, *self.getFullGrid(defaults)):  # Generates the entire domain, then quits.
                sample = HyperparameterGrid.Sample(seed=seed, wu=wu, bs=bs, lr=lr, dr=dr)
                if sample in excluded:
                    continue
                yield sample

        yield from take(n_samples, generateUniqueSamplesNotExcluded(), exact=False)


class TaskTuner:
    """
    Implements the tuning procedure described in the GRaMPa paper.

    Samples values for certain hyperparameters in their given domains, and knows how to alter the training procedure
    when those supported hyperparameters change.
    """

    def __init__(
        self,
        sampling_grid: Optional[HyperparameterGrid]=None,
        guaranteed_grid: Optional[HyperparameterGrid]=None,
        ###
        model_augmentation: Optional[ModelAugmentation]=None, custom_callbacks: List[TrainerCallback]=None
    ):
        assert sampling_grid is not None or guaranteed_grid is not None

        self._guaranteed      = guaranteed_grid
        self._sampling_domain = sampling_grid
        self._trainer = TaskTrainer(model_augmentation, custom_callbacks)

    def withAugmentation(self, model_augmentation: Optional[ModelAugmentation]) -> "TaskTuner":
        return TaskTuner(
            sampling_grid=self._sampling_domain,
            guaranteed_grid=self._guaranteed,
            custom_callbacks=self._trainer._extra_callbacks,
            model_augmentation=model_augmentation
        )

    def tune(self, task: Task, hp: TaskHyperparameters, meta: MetaHyperparameters):
        hp   = hp.copy()
        meta = meta.copy()

        # Sanity checks and imputations
        assert meta.n_grid_samples >= 1, "At least one hyperparameter sample must be taken."
        assert hp.init_weights and isinstance(hp.model_config_or_checkpoint, (str, Path)), "Can only tune starting from a pre-trained checkpoint."

        # Find best hp changes
        original_stopping_condition = hp.hard_stopping_condition
        original_traceless          = hp.traceless
        best_sample = self._phase1(task, hp, meta)

        # Apply best hp changes
        hp.hard_stopping_condition = original_stopping_condition  # FIXME: We override this in phase 2 regardless... Maybe allow both custom stopping condition and automatic stopping condition?
        hp.traceless               = original_traceless
        return self._phase2(task, hp, meta, best_sample)

    def _phase1(self, task: Task, hp: TaskHyperparameters, meta: MetaHyperparameters) -> HyperparameterGrid.Sample:
        """
        Try to find the optimal set of hyperparameters from n grid samples.
        """
        # Hyperparameter setup
        hp.traceless = True
        hp.track_best_checkpoint = True
        hp.rank_checkpoints_using_loss = True  # Within one HP sample, you use loss to find the best weights. Across samples, you select based on the ranking metric.
        hp.hard_stopping_condition = AfterNExamples(meta.max_examples_phase_1)  # Independent of batch size.
        hp.eval_vs_save_intervals.evaluation = EveryNExamplesOrOncePerEpoch(meta.max_examples_phase_1 // meta.minmax_evals_phase_1)
        hp.examples_per_evaluation = None  # Inference should be fast enough to process anything in GLUE quickly enough.
        rank_samples_by = meta.rank_by or task.metric_config.to_rank  # If no ranking metric is given, we use the task's default.

        # Grid search
        ranking_metric_name = "eval_" + rank_samples_by.fullName()
        best_ranking_value = -float("inf") if rank_samples_by.higher_is_better else float("inf")
        best_sample = None

        samples_so_far = set()
        cached_hp = hp.copy()  # Save these defaults for the second grid's loop. Otherwise, if it the second grid has a None where the first did not, the last HP value of the first grid will be used rather than the given default.
        for stochastic in [False, True]:
            if not stochastic:
                if self._guaranteed is None:
                    continue
                samples    = self._guaranteed.enumerate(defaults=hp)
                n_possible = self._guaranteed.domainSize()
                n_target   = n_possible
            else:
                if self._sampling_domain is None:
                    continue
                hp = cached_hp
                samples    = self._sampling_domain.sample(sampling_seed=hp.seed + meta.meta_seed, n_samples=meta.n_grid_samples, defaults=hp, excluded=samples_so_far)
                n_possible = self._sampling_domain.domainSize()
                n_target   = meta.n_grid_samples

            for n, grid_sample in enumerate(samples, start=1):
                samples_so_far.add(grid_sample)
                grid_sample.modifyHyperparameters(hp)

                log(f"\nStarting short tuning for {ordinal(len(samples_so_far))} hyperparameter set ({n} of {n_target} {'sampled' if stochastic else 'guaranteed'} from grid of size {n_possible}):", grid_sample)
                _, results = self._trainer.train(task, hp)
                log(f"Finished short tuning for {ordinal(len(samples_so_far))} hyperparameter set:", grid_sample)
                print("Results:")
                dprint(results, indent=1)

                # Ranking
                if ranking_metric_name not in results:
                    log(f"WARNING: Missing ranking metric {ranking_metric_name}. Cannot rank this hyperparameter set.")
                else:
                    new_ranking_value = results[ranking_metric_name]
                    if rank_samples_by.higher_is_better and new_ranking_value > best_ranking_value or \
                        not rank_samples_by.higher_is_better and new_ranking_value < best_ranking_value:
                        best_ranking_value = new_ranking_value
                        best_sample = grid_sample
                print("=" * 50)

        if best_sample is None:
            raise RuntimeError(f"No hyperparameter sets resulted in the ranking metric '{ranking_metric_name}'.")

        log(f"Best hyperparameters out of {pluralise(meta.n_grid_samples, 'sample')} as measured by {ranking_metric_name}:", best_sample, f"with metric value {best_ranking_value}.")
        return best_sample

    def _phase2(self, task: Task, hp: TaskHyperparameters, meta: MetaHyperparameters, best_sample: HyperparameterGrid.Sample) -> Dict[str,float]:
        """
        Use the best hyperparameters you found and run until you can't.
        """
        hp.hard_stopping_condition = AfterNExamples(meta.max_examples_phase_2)
        hp.eval_vs_save_intervals.evaluation = EveryNExamplesOrOncePerEpoch(meta.max_examples_phase_2 // meta.minmax_evals_phase_2)
        hp.track_best_checkpoint       = True
        hp.rank_checkpoints_using_loss = False
        if meta.rank_by is not None:  # Override the task's usual to_rank metric with a custom one.
            task.metric_config.to_rank = meta.rank_by

        best_sample.modifyHyperparameters(hp)
        log("Starting long tuning for best hyperparameters:", best_sample)
        identifier, results = self._trainer.train(task, hp)
        log("Finished long tuning for best hyperparameters:", best_sample)
        print("Results:")
        dprint(results, indent=1)

        # Save meta-hyperparameters so there is no confusion about how finetuning was done later on.
        #   TODO: Ideally, these are also added to the W&B log, but that would essentially mean giving an arbitrary dictionary to the Trainer
        #         (perhaps in a method?) that is then .log()ed after wandb.init and before wandb.finish.
        with open(LamotoPaths.append(LamotoPaths.pathToEvaluations(), identifier) / "tuning-config.json", "w", encoding="utf-8") as handle:
            json.dump(asdict(meta), handle)
        return results


def sampleGridWithoutReplacement(rng: npr.Generator, n_samples: Optional[int], *domains: List[float]) -> Iterable[Tuple[float, ...]]:
    max_n_samples = prod(len(domain) for domain in domains)
    n_samples     = n_samples or max_n_samples
    if n_samples > max_n_samples:
        raise ValueError(f"Cannot take {n_samples} samples from a grid of " + " x ".join(map(str,map(len,domains))) + f" == {max_n_samples} tuples.")

    def denumpyify(x):
        try:
            return x.item()
        except:
            return x

    def generateSamples():
        while True:
            yield tuple(denumpyify(rng.choice(domain)) for domain in domains)

    yield from take(n_samples, keepFirst(generateSamples()))
