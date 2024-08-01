"""
Core fine-tuning script for any task.

TODO:
    - Should have a way to load custom models beyond using from_pretrained, e.g. NILF with partial init.
    - Should the optimisers be given as training parameters to allow the use of accelerate (and perhaps multi-GPU)?
    - I wonder if .train(resume_from_checkpoint) keeps instance-level architecture or acts like .from_pretrained() in that it resets the architecture.
"""
from typing import Any, Dict, Type, List, Tuple
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass

import traceback
import wandb
import torch
from datasets import DatasetDict, Dataset
from transformers import DataCollator, Trainer, TrainingArguments, AutoTokenizer, PreTrainedModel, \
    PreTrainedTokenizerBase, PretrainedConfig, AutoConfig, IntervalStrategy, EarlyStoppingCallback, EvalPrediction
import transformers.optimization
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from fiject.hooks.transformers import FijectCallback
from tktkt.files.paths import DataPaths
from tktkt.util.timing import datetimeDashed

from ..augmenting.augment_model import ModelAugmentation
from ..measuring._core import Metric, LamotoMetric
from ..measuring import METRICS
from ..trainer.callbacks import *
from ..trainer.hyperparameters import *
from ..trainer.trainers import TrainerWithoutEvaluationLoop
from ..util.datasets import shuffleAndTruncate, getDatasetSize, totalBatches
from ..util.strings import getSubstringAfterLastSlash


#################################################
SUGGESTED_HYPERPARAMETERS = TaskHyperparameters(
    SAVE_AS=None,
    WANDB_PROJECT=None,

    EXAMPLES_PER_EFFECTIVE_BATCH=32,
    EXAMPLES_PER_DEVICEBATCH=32,
    EFFECTIVE_BATCHES_WARMUP=100,  # The RoBERTa paper says, for GLUE tasks, they warm up for 6% of all batches across 10 epochs. That's in the ballpark of 100 batches.

    HARD_STOPPING_CONDITION=AfterNEpochs(epochs=10, effective_batch_size=32),
    EXAMPLES_PER_EVALUATION=None,
    EVALS_OF_PATIENCE=9,

    TRACK_BEST_MODEL=True,
    EVAL_VS_SAVE_INTERVALS=Intervals(
        evaluation=NEveryEpoch(per_epoch=1, effective_batch_size=32),
        checkpointing=None
    ),

    INIT_WEIGHTS=True,
    CHECKPOINT_OR_CONFIG="roberta-base",
    TOKENISER_CHECKPOINT="roberta-base",

    LEARNING_RATE=2e-5,
    L2_REGULARISATION=0.01,

    ADD_SPECIAL_TOKENS=True
)
#################################################


@dataclass
class MetricSetup:
    to_compute: List[str]               # Names of all the HuggingFace evaluate metrics to load and compute in the end.
    to_track: Dict[str, Dict[str,str]]  # metric name -> result name -> formatted name, used for graphing intermediate evaluations.


class Task(ABC):

    def __init__(self, task_name: str, metric_config: MetricSetup, automodel_class: Type[_BaseAutoModelClass], **automodel_args):
        self.task_name       = task_name
        self.metric_config   = metric_config
        self.automodel_class = automodel_class
        self.automodel_args  = automodel_args

        # Fields that can be used by method implementations, but are only instantiated once .train() is called, to
        # avoid loading heavy objects that would be duplicated by the super() call of a task wrapper.
        self.hyperparameters: TaskHyperparameters = None
        self.tokenizer: PreTrainedTokenizerBase = None
        self.model_config: PretrainedConfig = None
        self.metrics: Dict[str, Metric] = None

    @abstractmethod
    def loadDataset(self) -> DatasetDict:
        pass

    @abstractmethod
    def prepareDataset(self, dataset: DatasetDict) -> DatasetDict:
        pass

    @abstractmethod
    def getCollator(self) -> DataCollator:
        pass

    @abstractmethod
    def getPredictionsAndReferences(self, eval: EvalPrediction) -> Tuple[Any,Any]:
        pass

    def computeMetrics(self, eval: EvalPrediction) -> dict:
        predictions, references = self.getPredictionsAndReferences(eval)
        results = dict()
        for metric_name, metric in self.metrics.items():
            subresults = metric.compute(predictions=predictions, references=references)
            for key, value in subresults.items():
                results[metric_name + "_" + key] = value
        return results  # To this dictionary, the eval loss will be added post-hoc.

    def sneakyLogitTransform(self, logits, labels):
        return logits

    def _getMaxInputLength(self) -> int:
        """
        Helper function to find the amount of tokens the model can accept at most.
        """
        # First try the tokeniser itself. For CANINE, this is the only place where you find the correct number (2048).
        try:
            n = self.tokenizer.model_max_length
            if n < 1e12:  # Due to very persistent issue where the model config is right and the tokeniser is wrong: https://github.com/huggingface/transformers/issues/14561
                return n
        except:
            pass

        # Alternatively try the model config. This name was standardised late, so it is possible that you can't find it.
        try:
            n = self.model_config.max_position_embeddings
            if n:
                return n
        except:
            if "max_position_embeddings" in self.model_config.attribute_map:
                return self.model_config.__dict__[self.model_config.attribute_map["max_position_embeddings"]]
            else:
                raise RuntimeError("Couldn't find maximum input length in the tokeniser nor the model config.")

    def train(self, hyperparameters: TaskHyperparameters=SUGGESTED_HYPERPARAMETERS, model_augmentation: ModelAugmentation=None, resume_from_folder: Path=None):
        transformers.set_seed(seed=69420)

        # Metadata
        self.hyperparameters = hyperparameters
        self.model_config = AutoConfig.from_pretrained(hyperparameters.CHECKPOINT_OR_CONFIG) if isinstance(hyperparameters.CHECKPOINT_OR_CONFIG, str) else hyperparameters.CHECKPOINT_OR_CONFIG

        if hyperparameters.SAVE_AS:
            model_name = hyperparameters.SAVE_AS
        elif isinstance(hyperparameters.CHECKPOINT_OR_CONFIG, str):
            model_name = getSubstringAfterLastSlash(hyperparameters.CHECKPOINT_OR_CONFIG)
        else:  # We don't use the tokeniser name because it isn't directly related to the model.
            raise RuntimeError("Cannot deduce name to save model as from a config.")

        global_model_identifier = model_name \
                                + ("" if not model_augmentation else ("-" + model_augmentation.name)) \
                                + f"_{self.task_name}_{datetimeDashed()}"

        PATH_CHECKPOINTS = DataPaths.pathToCheckpoints() / global_model_identifier
        PATH_CHECKPOINTS.mkdir(exist_ok=True, parents=True)

        # Set up tokeniser
        if hyperparameters.TOKENISER_CHECKPOINT:
            tokeniser_checkpoint = hyperparameters.TOKENISER_CHECKPOINT
        elif isinstance(hyperparameters.CHECKPOINT_OR_CONFIG, str):
            tokeniser_checkpoint = hyperparameters.CHECKPOINT_OR_CONFIG
        else:
            raise RuntimeError("Cannot deduce tokeniser checkpoint from config.")

        self.tokenizer = AutoTokenizer.from_pretrained(tokeniser_checkpoint, add_prefix_space=True)

        # - Old models like GPT-2 have no pad token, but this doesn't really matter because it's actually the attention
        #   mask that determines if a token is processed, so you can replace it by any token you want. https://github.com/stanford-crfm/BioMedLM/issues/4
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model_config.pad_token_id = self.model_config.eos_token_id

        # Now that you have the tokeniser, tokenise the dataset.
        datasetdict = self.loadDataset()
        datasetdict["train"]      = shuffleAndTruncate(datasetdict["train"])
        datasetdict["validation"] = shuffleAndTruncate(datasetdict["validation"], truncate_to=hyperparameters.EXAMPLES_PER_EVALUATION)
        datasetdict = self.prepareDataset(datasetdict)

        # Get the batch generator, a.k.a. collator (https://huggingface.co/docs/transformers/main_classes/data_collator).
        collator = self.getCollator()

        # Get the model...
        self.automodel_args["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        if hyperparameters.INIT_WEIGHTS:
            if not isinstance(hyperparameters.CHECKPOINT_OR_CONFIG, str):
                raise ValueError("To initialise model weights, you should give a checkpoint path, not a config object.")
            model: PreTrainedModel = self.automodel_class.from_pretrained(hyperparameters.CHECKPOINT_OR_CONFIG, **self.automodel_args)
        else:
            model: PreTrainedModel = self.automodel_class.from_config(self.model_config, **self.automodel_args)
        model.config.pad_token_id = self.model_config.pad_token_id

        # ...and augment it (possibly with the tokeniser).
        if model_augmentation:
            model = model_augmentation.augment(model, self.tokenizer)
        model.to("cuda")

        # Now that we have a reference to the dataset and model, build the metrics.
        env = EvaluationEnvironment(
            model=model,
            tokeniser=self.tokenizer,
            validation_dataset=datasetdict["validation"],
            hyperparameters=self.hyperparameters
        )
        self.metrics: Dict[str, Metric] = {name: METRICS.load(name,env) for name in self.metric_config.to_compute}

        # Set up reporting too
        wandb.init(
            mode="disabled" if not hyperparameters.WANDB_PROJECT else "online",

            project=hyperparameters.WANDB_PROJECT,
            group=model_name,
            name=global_model_identifier,
            tags=[self.task_name] + ([model_augmentation.name] if model_augmentation else [])
        )

        # Training arguments
        # - Sizes
        stopping_condition = hyperparameters.HARD_STOPPING_CONDITION
        n_gradient_descents = stopping_condition.getSteps(datasetdict["train"]) if not isinstance(stopping_condition, (NeverStop, AfterNMinutes)) else None
        n_accumulations     = hyperparameters.EXAMPLES_PER_EFFECTIVE_BATCH // (torch.cuda.device_count() * hyperparameters.EXAMPLES_PER_DEVICEBATCH)  # The amount of times, to get to one effective batch, you have to push a device batch through all devices in parallel.
        wu = hyperparameters.EFFECTIVE_BATCHES_WARMUP  # Alias to shorten this long name.
        if isinstance(wu, int):
            if wu < 0:
                raise ValueError("The amount of warmup batches has to be a positive integer or a float in [0,1].")
            n_descents_of_warmup = wu
        else:  # Fractional warmup in [0,1]
            if wu < 0 or wu > 1:
                raise ValueError("The amount of warmup batches has to be a positive integer or a float in [0,1].")
            if not n_gradient_descents:
                raise ValueError(f"Amount of warmup batches was given as a fraction of the total amount of training batches, but we don't know what that is for stopping condition {hyperparameters.HARD_STOPPING_CONDITION.__class__.__name__}")
            n_descents_of_warmup = int(n_gradient_descents*wu)

        # - Intervals
        eval_interval = hyperparameters.EVAL_VS_SAVE_INTERVALS.evaluation
        if not hyperparameters.TRACK_BEST_MODEL:
            save_interval = hyperparameters.EVAL_VS_SAVE_INTERVALS.checkpointing
        else:  # Ignore it and sync with eval interval.
            if isinstance(eval_interval, NoStrategy):
                raise ValueError("You indicated that you want to track the best model, but specified no interval strategy!")
            save_interval = eval_interval

        batches_between_evals = eval_interval.getSteps(datasetdict["train"]) if isinstance(eval_interval, (EveryNDescents, NEveryEpoch)) else None
        batches_between_saves = save_interval.getSteps(datasetdict["train"]) if isinstance(save_interval, (EveryNDescents, NEveryEpoch)) else None

        # - Finally get args
        training_args = TrainingArguments(
            max_steps=n_gradient_descents,  # Can be None.

            # Optimisation (adding all of this in the TrainingArguments because apparently Trainer knows how to use HuggingFace `accelerate` whereas I only know the old optimisers)
            # optim=OptimizerNames.ADAMW_TORCH,
            # learning_rate=hyperparameters.LEARNING_RATE,
            # weight_decay=hyperparameters.L2_REGULARISATION,

            # lr_scheduler_type=SchedulerType.CONSTANT_WITH_WARMUP,
            # warmup_steps=n_descents_of_warmup,

            # Batches
            per_device_train_batch_size=hyperparameters.EXAMPLES_PER_DEVICEBATCH,
            gradient_accumulation_steps=n_accumulations,

            # Style of computations
            gradient_checkpointing=True,  # Only if you have the VRAM though. Good explanation with animations: https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9
            bf16=torch.cuda.is_bf16_supported(),

            # Evaluation
            evaluation_strategy=IntervalStrategy.STEPS if batches_between_evals else IntervalStrategy.NO,
            eval_steps=batches_between_evals,
            per_device_eval_batch_size=hyperparameters.EXAMPLES_PER_DEVICEBATCH,
            eval_accumulation_steps=n_accumulations,  # "Number of predictions steps to accumulate the output tensors for, before moving the results to the CPU."

            # Saving
            save_strategy=IntervalStrategy.STEPS if batches_between_saves else IntervalStrategy.NO,
            save_steps=batches_between_saves,

            output_dir=PATH_CHECKPOINTS.as_posix(),

            load_best_model_at_end=hyperparameters.TRACK_BEST_MODEL,  # Will take the best model out of its checkpoint directory and load it into self.model, which can then be saved. At the end of Trainer's loop, the following happens: "Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint"
            metric_for_best_model="eval_loss" if hyperparameters.TRACK_BEST_MODEL else None,  # TODO: Can become an issue if you don't want to select based on eval loss but e.g. downstream F1.
            save_total_limit=1,  # This will keep the last model stored plus the best model if those aren't the same, allowing you to have the best model and continue training from last if you need to. https://stackoverflow.com/a/67615225/9352077

            # Logging
            report_to=["wandb"],  # Can be turned off below.
            logging_strategy=IntervalStrategy.STEPS,
            logging_steps=1,  # Gradient descents between each push to the log.
            logging_first_step=True,
            include_num_input_tokens_seen=True,

            # hub_model_id=new_model_name,
            # hub_private_repo=True,
            # push_to_hub=True,
            # hub_strategy='all_checkpoints',
        )

        # - Build optimiser
        optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparameters.LEARNING_RATE, weight_decay=hyperparameters.L2_REGULARISATION, betas=(0.9, 0.999))  # Not using transformers.optimization because it gives a deprecation warning.
        scheduler = transformers.optimization.get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=n_descents_of_warmup)  # Not using a linear decay because that's the whole point of having Adam.

        # - Build callbacks
        callbacks = [EvaluateBeforeTrainingCallback(), CheckpointLastModel(), SaveTokeniserWithCheckpoints(self.tokenizer)]
        if hyperparameters.TRACK_BEST_MODEL and hyperparameters.EVALS_OF_PATIENCE is not None:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=hyperparameters.EVALS_OF_PATIENCE))  # Patience is the amount of eval calls you can tolerate worsening loss.

        if isinstance(stopping_condition, AfterNMinutes):
            callbacks.append(CallbackAtTimeInterval(minutes=stopping_condition.minutes, events=EventType.STOP))

        if isinstance(eval_interval, EveryNMinutes) and isinstance(save_interval, EveryNMinutes) and eval_interval.minutes == save_interval.minutes:  # They are completely tied. This means you need a fully synchronised callback to prevent race conditions.
            callbacks.append(CallbackAtTimeInterval(minutes=eval_interval.minutes, events={EventType.EVALUATE, EventType.CHECKPOINT}))
        else:  # Can be neither, one, or both but with disparate minutes. Either way, you'll need a separate callback per type.
            if isinstance(eval_interval, EveryNMinutes):
                callbacks.append(CallbackAtTimeInterval(minutes=eval_interval.minutes, events=EventType.EVALUATE))
            if isinstance(save_interval, EveryNMinutes):
                callbacks.append(CallbackAtTimeInterval(minutes=save_interval.minutes, events=EventType.CHECKPOINT))

        if not hyperparameters.WANDB_PROJECT:
            callbacks.append(FijectCallback(global_model_identifier + "_eval_loss", evals_between_commits=4))
            callbacks.append(
                FijectCallback(global_model_identifier + "_eval_task",
                               evals_between_commits=4,
                               metric_names_with_formatting={(metric_name + "_" + result_name): formatting
                                                             for metric_name, result_formats in
                                                             self.metric_config.to_track.items()
                                                             for result_name, formatting in result_formats.items()})
            )

        # At last, the Trainer object.
        no_traditional_metrics = all(isinstance(m, LamotoMetric) and m.isAutonomous() for m in self.metrics.values())
        TrainerClass = TrainerWithoutEvaluationLoop if no_traditional_metrics and not hyperparameters.TRACK_BEST_MODEL else Trainer
        trainer = TrainerClass(
            model=model,
            # tokenizer=self.tokenizer,  # Don't pass it if you don't want to save it and have other wacky shit extracted from it to influence training.

            # Args
            args=training_args,
            optimizers=(optimizer, scheduler),
            callbacks=callbacks,

            # Data
            train_dataset=datasetdict["train"],
            eval_dataset=[] if no_traditional_metrics and not hyperparameters.TRACK_BEST_MODEL else datasetdict["validation"],
            data_collator=collator,

            # Evaluation
            compute_metrics=self.computeMetrics,
            preprocess_logits_for_metrics=self.sneakyLogitTransform
        )

        # Lastly, do some prints.
        print("=== TRAINING SIZES ===")
        bs = hyperparameters.EXAMPLES_PER_EFFECTIVE_BATCH
        e = getDatasetSize(datasetdict["train"], "train")
        ev = getDatasetSize(datasetdict["validation"], "validation")
        batches_per_epoch = totalBatches(e, bs)
        batches_per_eval  = totalBatches(ev, bs)
        print("Batch size:", bs)
        print("Training set:")
        print("\t", e, "examples per epoch")
        print("\t", batches_per_epoch, "batches per epoch")
        if n_gradient_descents:
            print("\t", round(n_gradient_descents / batches_per_epoch, 1), "epochs")
            print("\t", n_gradient_descents, "batches in total")
        print("Evaluation set:")
        print("\t", ev, "examples per evaluation")
        print("\t", batches_per_eval, "batches per evaluation")
        if batches_between_evals:
            print("\t", batches_between_evals, "training batches between evals")
            print("\t", batches_per_epoch // batches_between_evals, "evals per training epoch")
        print("======================")

        # Train, and evaluate afterwards.
        try:
            print("Training:")
            trainer.train(resume_from_checkpoint=resume_from_folder.as_posix() if resume_from_folder else None)
            # trainer.save_model()  # 1. We already checkpoint the last model with a callback, 2. LM pretraining basically never gets to convergence, and 3. we don't have a metric configured because we're not doing traditional eval (although this is probably not a problem since compute_metrics might be where you get your metric anyway).
            # trainer.push_to_hub()
            print("Evaluation of " + ("best" if hyperparameters.TRACK_BEST_MODEL else "last") + " model:")
            print(trainer.evaluate())
        except Exception:  # Catches any error that happens during training, and triggers a checkpoint (+ a callback event afterwards, if that's needed by any callback).
            print("Caught exception while training:")
            print("="*32)
            print(traceback.format_exc())
            print("="*32)
            print("A final checkpoint will be saved.")

            trainer.control.should_save     = True
            trainer.control.should_evaluate = False
            trainer.control.should_log      = False
            trainer._maybe_log_save_evaluate(tr_loss=None, grad_norm=None, model=None, trial=None, epoch=None, ignore_keys_for_eval=None)  # These arguments are imputed anyway.


__all__ = ["Task", "MetricSetup", "TaskHyperparameters", "SUGGESTED_HYPERPARAMETERS",
           "Intervals", "NoStrategy", "EveryNDescents", "NEveryEpoch", "EveryNMinutes", "NeverStop", "AfterNDescents", "AfterNEpochs", "AfterNTokens", "AfterNMinutes",
           "DatasetDict", "DataCollator", "Any", "Tuple", "Path", "ModelAugmentation", "EvalPrediction"]