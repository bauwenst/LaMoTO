"""
Core fine-tuning script for any task.

TODO:
    - Should the optimisers be given as training parameters to allow the use of accelerate (and perhaps multi-GPU)?
    - I wonder if .train(resume_from_checkpoint) keeps instance-level architecture or acts like .from_pretrained() in that it resets the architecture.
"""
from typing import Any, Dict, Type, List, Tuple, Generic
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass

import traceback
import time
import wandb
import torch
from datasets import DatasetDict, Dataset
from transformers import DataCollator, Trainer, TrainingArguments, AutoTokenizer, PreTrainedModel, \
    PreTrainedTokenizerBase, PretrainedConfig, AutoConfig, IntervalStrategy, EarlyStoppingCallback, EvalPrediction
import transformers.optimization
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from fiject.hooks.transformers import FijectCallback
from tktkt.files.paths import PathManager
from tktkt.interfaces.huggingface import TktktToHuggingFace
from tktkt.interfaces.tokeniser import TokeniserWithFiniteTypeDomain
from tktkt.util.timing import datetimeDashed
from tktkt.util.printing import pluralise
from archit.instantiation.abstracts import ModelWithHead, CombinedConfig
from archit.util import torchPrint

from ..augmenting.augment_model import ModelAugmentation
from ..measuring._core import Metric, LamotoMetric
from ..measuring import METRICS
from ..trainer.callbacks import *
from ..trainer.hyperparameters import *
from ..trainer.trainers import LamotoTrainer, LamotoTrainerWithoutEvaluationLoop
from ..util.datasets import shuffleAndTruncate, getDatasetSize, totalBatches
from ..util.exceptions import tryExceptNone
from ..util.strings import getSubstringAfterLastSlash
from ..util.visuals import printLamotoWelcome, log

LamotoPaths = PathManager("lamoto")


@dataclass
class MetricSetup:
    to_compute: List[str]               # Names of all the HuggingFace evaluate metrics to load and compute in the end.
    to_track: Dict[str, Dict[str,str]]  # metric name -> result name -> formatted name, used for graphing intermediate evaluations.


class Task(ABC, Generic[HC]):

    def __init__(self, task_name: str, metric_config: MetricSetup,
                 archit_class: Type[ModelWithHead[PC,HC]], automodel_class: Type[_BaseAutoModelClass], **automodel_args):
        self.task_name       = task_name
        self.metric_config   = metric_config
        self.archit_class    = archit_class
        self.automodel_class = automodel_class
        self.automodel_args  = automodel_args

        # Fields that can be used by method implementations, but are only instantiated once .train() is called, to
        # avoid loading heavy objects that would be duplicated by the super() call of a task wrapper.
        self.hyperparameters: TaskHyperparameters[HC] = None
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
    def adjustHyperparameters(self, hp: TaskHyperparameters[HC]):
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
            if "max_position_embeddings" in self.model_config.attribute_map:  # All PretrainedConfig classes have an attribute map.
                return self.model_config.__dict__[self.model_config.attribute_map["max_position_embeddings"]]
            else:
                raise RuntimeError("Couldn't find maximum input length in the tokeniser nor the model config.")

    def _isHfCheckpointForThisTask(self, architecture_name: str):
        """
        HuggingFace architectures look like "[Base]For[Task]" while ArchIt architectures can in principle be anything,
        although they conventionally look like "For[Task]".
        ArchIt architectures define which HuggingFace [Task] string is equivalent for them. If an architecture hence
        contains that string (but isn't equal to it, because in that case it is not HuggingFace and hence must be ArchIt)
        it comes from HuggingFace and is tailored to this task.
        """
        return self.archit_class.head_class.hfEquivalentSuffix() in architecture_name and \
               self.archit_class.head_class.hfEquivalentSuffix() != architecture_name

    def train(self, hyperparameters: TaskHyperparameters[HC]=getDefaultHyperparameters(), model_augmentation: ModelAugmentation=None, resume_from_folder: Path=None):
        """
        Encapsulation of everything you need to do to get a Trainer running.
        """
        printLamotoWelcome()
        transformers.set_seed(seed=hyperparameters.SEED)

        # Sanity check(s)
        if isinstance(hyperparameters.MODEL_CONFIG_OR_CHECKPOINT, Path):
            hyperparameters.MODEL_CONFIG_OR_CHECKPOINT = hyperparameters.MODEL_CONFIG_OR_CHECKPOINT.as_posix()  # FIXME: Possibly have to make it a relative path due to HF restrictions.
        if hyperparameters.init_weights and not isinstance(hyperparameters.MODEL_CONFIG_OR_CHECKPOINT, str):
            raise ValueError("You said you wanted to initialise model weights from the checkpoint, but didn't give a checkpoint path!")
        if hyperparameters.archit_basemodel_class is None:
            raise ValueError("In order to parse model configs, the archit_basemodel_class hyperparameter cannot be None.")

        # Metadata
        self.adjustHyperparameters(hyperparameters)
        self.hyperparameters = hyperparameters

        config_or_str = hyperparameters.MODEL_CONFIG_OR_CHECKPOINT
        if not isinstance(config_or_str, str):  # It's an object.
            if isinstance(config_or_str, CombinedConfig):
                raise ValueError("When instantiating a new model from a config, it must only parameterise the base model, without a head.")
            self.model_config = CombinedConfig(base_model_config=config_or_str,
                                               head_config=hyperparameters.archit_head_config,
                                               base_model_config_class=hyperparameters.archit_basemodel_class.config_class)
        else:  # It's a checkpoint string. Can either be a checkpoint for the ModelWithHead we're about to load, or for anything else compatible. We'll figure that out.
            self.model_config = CombinedConfig.from_pretrained(config_or_str,
                                                               head_config=hyperparameters.archit_head_config,
                                                               base_model_config_class=hyperparameters.archit_basemodel_class.config_class)  # Note that there is no need for AutoConfig because we KNOW the config class (even if not registered in AutoConfig). Also means we don't have to store the "model type" in the config.

        if hyperparameters.SAVE_AS:
            model_name = hyperparameters.SAVE_AS
        elif isinstance(hyperparameters.MODEL_CONFIG_OR_CHECKPOINT, str):
            model_name = getSubstringAfterLastSlash(hyperparameters.MODEL_CONFIG_OR_CHECKPOINT)
        else:  # We don't use the tokeniser name because it isn't directly related to the model.
            raise RuntimeError("Cannot deduce name to save model as from a config.")

        global_model_identifier = model_name \
                                + ("" if not model_augmentation else ("-" + model_augmentation.name)) \
                                + f"_{self.task_name}_{datetimeDashed()}"

        folder_to_this_models_checkpoints = LamotoPaths.pathToCheckpoints() / global_model_identifier
        folder_to_this_models_checkpoints.mkdir(exist_ok=True, parents=True)

        # Set up tokeniser
        if hyperparameters.TOKENISER:
            if isinstance(hyperparameters.TOKENISER, str):
                self.tokenizer = AutoTokenizer.from_pretrained(hyperparameters.TOKENISER, add_prefix_space=True)
            elif isinstance(hyperparameters.TOKENISER, TokeniserWithFiniteTypeDomain):
                self.tokenizer = TktktToHuggingFace(hyperparameters.TOKENISER)
            else:
                self.tokenizer = hyperparameters.TOKENISER
        elif isinstance(hyperparameters.MODEL_CONFIG_OR_CHECKPOINT, str):
            self.tokenizer = AutoTokenizer.from_pretrained(hyperparameters.MODEL_CONFIG_OR_CHECKPOINT, add_prefix_space=True)
        else:
            raise RuntimeError("Cannot deduce tokeniser checkpoint from config.")

        # - Old models like GPT-2 have no pad token, but this doesn't really matter because it's actually the attention
        #   mask that determines if a token is processed, so you can replace it by any token you want. https://github.com/stanford-crfm/BioMedLM/issues/4
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token                         = self.tokenizer.eos_token
            self.model_config.base_model_config.pad_token_id = self.tokenizer.eos_token_id

        # Now that you have the tokeniser, tokenise the dataset.
        log("Loading dataset...")
        datasetdict = self.loadDataset()
        n_examples_validation = tryExceptNone(getDatasetSize(datasetdict["validation"], split="validation")) or 1_000_000_000_000  # Very very big number assumed when you can't find the dataset size.
        hyperparameters.EXAMPLES_PER_EVALUATION = n_examples_validation if not hyperparameters.EXAMPLES_PER_EVALUATION else min(n_examples_validation, hyperparameters.EXAMPLES_PER_EVALUATION)

        log("Preparing dataset...")
        datasetdict["train"]      = shuffleAndTruncate(datasetdict["train"], seed=hyperparameters.SEED)
        datasetdict["validation"] = shuffleAndTruncate(datasetdict["validation"], seed=hyperparameters.SEED, truncate_to=hyperparameters.EXAMPLES_PER_EVALUATION)
        datasetdict = self.prepareDataset(datasetdict)

        # Get the batch generator, a.k.a. collator (https://huggingface.co/docs/transformers/main_classes/data_collator).
        collator = self.getCollator()

        # Get the model...
        self.automodel_args["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

        hf_checkpoint_classname = self.model_config.architectures[0] if self.model_config.architectures is not None else ""  # Always present and correct for HuggingFace configs.
        is_exact_hf_checkpoint    = hyperparameters.init_weights and hyperparameters.load_hf_automodel_if_hf_checkpoint_and_matches_task and self._isHfCheckpointForThisTask(hf_checkpoint_classname)
        is_custom_hf_architecture = hyperparameters.custom_hf_class is not None
        if not is_exact_hf_checkpoint and not is_custom_hf_architecture:  # Use ArchIt. This is the usual case.
            log("Instantiating an ArchIt model.")
            torch.set_default_dtype(self.automodel_args["torch_dtype"])
            if hyperparameters.init_weights:
                # FIXME: This branch may be broken in case the checkpoint is an ArchIt checkpoint, see the FIXME under .from_pretrained().
                model: PreTrainedModel = self.archit_class.from_pretrained(hyperparameters.MODEL_CONFIG_OR_CHECKPOINT, hyperparameters.archit_basemodel_class, hyperparameters.archit_head_config)
            else:
                model: PreTrainedModel = self.archit_class.fromModelAndHeadConfig(hyperparameters.archit_basemodel_class.from_config(self.model_config), hyperparameters.archit_head_config)
        else:  # Edge cases.
            if is_custom_hf_architecture:
                log("Instantiating a custom HuggingFace class.")
                if hyperparameters.init_weights:  # model_config_or_checkpoint is a string
                    model: PreTrainedModel = hyperparameters.custom_hf_class.from_pretrained(hyperparameters.MODEL_CONFIG_OR_CHECKPOINT, **self.automodel_args)
                else:
                    model: PreTrainedModel = hyperparameters.custom_hf_class._from_config(self.model_config.base_model_config, **self.automodel_args)
            elif is_exact_hf_checkpoint:  # model_config_or_checkpoint is a string
                log(f"The given checkpoint seems to be a HuggingFace architecture ({hf_checkpoint_classname}) for this specific task ({self.archit_class.__name__}),\nwe will instantiate the model with AutoModel ({self.automodel_class.__name__}) instead of ArchIt.")
                model: PreTrainedModel = self.automodel_class.from_pretrained(hyperparameters.MODEL_CONFIG_OR_CHECKPOINT, **self.automodel_args)
            else:
                raise RuntimeError("Impossible.")
        model.config.pad_token_id = self.model_config.base_model_config.pad_token_id  # self.model_config might have been changed since AutoConfig.from_pretrained() was called, whereas model.config is the result of a fresh AutoConfig call.

        # ...and augment it in-place (possibly with the tokeniser). We assume the augmentation uses .base_model when it needs to.
        if model_augmentation:
            model_augmentation.augment(model, self.tokenizer)
        model.to("cuda")
        torchPrint(model)

        # Now that we have a reference to the dataset and model, build the metrics.
        env = EvaluationEnvironment(
            model=model,
            tokeniser=self.tokenizer,
            validation_dataset=datasetdict["validation"],
            hyperparameters=self.hyperparameters
        )
        self.metrics: Dict[str, Metric] = {name: METRICS.load(name,env) for name in self.metric_config.to_compute}

        # Set up reporting too
        folder_wandb = folder_to_this_models_checkpoints / "wandb"
        folder_wandb.mkdir(exist_ok=True)
        wandb.init(
            mode="disabled" if not hyperparameters.WANDB_PROJECT else "online",

            project=hyperparameters.WANDB_PROJECT,
            group=model_name,
            name=global_model_identifier,
            tags=[self.task_name] + ([model_augmentation.name] if model_augmentation else []),

            dir=folder_wandb.as_posix()
        )

        # Training arguments
        # - Sizes
        stopping_condition = hyperparameters.HARD_STOPPING_CONDITION
        n_gradient_descents = tryExceptNone(stopping_condition.getSteps(datasetdict["train"])) if not isinstance(stopping_condition, (NeverStop, AfterNMinutes)) else None
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
            if isinstance(eval_interval, NeverInterval):
                raise ValueError("You indicated that you want to track the best model, but specified no evaluation interval!")
            save_interval = eval_interval

        batches_between_evals = tryExceptNone(eval_interval.getSteps(datasetdict["train"])) if isinstance(eval_interval, (EveryNDescents, NEveryEpoch)) else None
        batches_between_saves = tryExceptNone(save_interval.getSteps(datasetdict["train"])) if isinstance(save_interval, (EveryNDescents, NEveryEpoch)) else None

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
            gradient_checkpointing=model.supports_gradient_checkpointing,  # Only if you have the VRAM though. Good explanation with animations: https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9
            bf16=torch.cuda.is_bf16_supported(),

            # Evaluation
            evaluation_strategy=IntervalStrategy.STEPS if batches_between_evals else IntervalStrategy.NO,
            eval_steps=batches_between_evals,
            per_device_eval_batch_size=hyperparameters.EXAMPLES_PER_DEVICEBATCH,
            eval_accumulation_steps=n_accumulations,  # "Number of predictions steps to accumulate the output tensors for, before moving the results to the CPU."

            # Saving
            save_strategy=IntervalStrategy.STEPS if batches_between_saves else IntervalStrategy.NO,
            save_steps=batches_between_saves,

            output_dir=folder_to_this_models_checkpoints.as_posix(),

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

            # Data
            remove_unused_columns=False,  # Otherwise, only those keys that match input arguments of the model are allowed to survive the preprocessor. Very weird system. They are already gone before the DataCollator gets to see anything. You'll get an "IndexError: is out of bounds for size 0" because the dataset looks like it has no columns.
            # dataloader_num_workers=4*IS_NOT_LINUX  # My wishful thinking was that this speeds up tokenisation by a factor of 4 for an IterableDataset.
        )

        # - Build optimiser
        optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparameters.LEARNING_RATE, weight_decay=hyperparameters.L2_REGULARISATION, betas=(0.9, 0.999))  # Not using transformers.optimization because it gives a deprecation warning.
        scheduler = transformers.optimization.get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=n_descents_of_warmup)  # Not using a linear decay because that's the whole point of having Adam.

        # - Build callbacks
        callbacks = [CheckpointLastModel(), SaveTokeniserWithCheckpoints(self.tokenizer)]
        if hyperparameters.TRACK_BEST_MODEL and hyperparameters.EVALS_OF_PATIENCE is not None:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=hyperparameters.EVALS_OF_PATIENCE))  # Patience is the amount of eval calls you can tolerate worsening loss.

        if not isinstance(eval_interval, NeverInterval):
            callbacks.append(EvaluateBeforeTrainingCallback())

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
            if hyperparameters.TRACK_BEST_MODEL:
                callbacks.append(FijectCallback(global_model_identifier + "_eval_loss", evals_between_commits=4))  # Automatically tracks the same metric as is used to decide best model.
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
        TrainerClass = LamotoTrainerWithoutEvaluationLoop if no_traditional_metrics and not hyperparameters.TRACK_BEST_MODEL else LamotoTrainer
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

        # Lastly, do some prints (not logs).
        print("="*17 + " TRAINING SIZES " + "="*17)
        bs = hyperparameters.EXAMPLES_PER_EFFECTIVE_BATCH
        e = getDatasetSize(datasetdict["train"], "train")
        ev = hyperparameters.EXAMPLES_PER_EVALUATION
        batches_per_epoch = totalBatches(e, bs)
        batches_per_eval  = totalBatches(ev, bs)
        print("Batch size:", bs)
        print("Training set:")
        print("\t", pluralise(e, "example"), "per epoch")
        print("\t", pluralise(batches_per_epoch, "batch", "es"), "per epoch")
        if n_gradient_descents:
            print("\t", round(n_gradient_descents / batches_per_epoch, 1), "epochs")
            print("\t", pluralise(n_gradient_descents, "batch", "es"), "in total")
        print("Evaluation set:")
        print("\t", pluralise(ev, "example"), "per evaluation")
        print("\t", pluralise(batches_per_eval, "batch", "es"), "per evaluation")
        if batches_between_evals:
            print("\t", pluralise(batches_between_evals, "training batch", "es"), "between evals")
            print("\t", pluralise(batches_per_epoch // batches_between_evals, "eval"), "per training epoch")
        print("="*50)

        # Train, and evaluate afterwards.
        try:
            log(f"Training {model.__class__.__name__} on {model.device}:")
            trainer.train(resume_from_checkpoint=resume_from_folder.as_posix() if resume_from_folder else None)
            # trainer.save_model()  # 1. We already checkpoint the last model with a callback, 2. LM pretraining basically never gets to convergence, and 3. we don't have a metric configured because we're not doing traditional eval (although this is probably not a problem since compute_metrics might be where you get your metric anyway).
            # trainer.push_to_hub()
            log("Evaluation of " + ("best" if hyperparameters.TRACK_BEST_MODEL else "last") + " model:")
            log(trainer.evaluate())
            wandb.finish()  # Finish because otherwise, running .train() in the same process after .init() has been called once already will raise an error.
        except Exception as e:  # Catches any error that happens during training, and triggers a checkpoint (+ a callback event afterwards, if that's needed by any callback).
            log("Caught exception while training. A checkpoint will be saved.\nAfterwards, we will raise the exception, so your run shows up as failed rather than completed.\n...")
            trainer.control.should_save     = True
            trainer.control.should_evaluate = False
            trainer.control.should_log      = False
            trainer._maybe_log_save_evaluate(tr_loss=None, grad_norm=None, model=None, trial=None, epoch=None, ignore_keys_for_eval=None)  # These arguments are imputed anyway.
            wandb.finish(exit_code=1)

            log("Save successful. Now raising the exception. Bye bye!")
            log("="*50)
            time.sleep(1)  # First let all the prints happen, so that the traceback doesn't race it to the output.
            # print(traceback.format_exc())
            raise e  # Automatically prints the traceback.


__all__ = ["Task", "MetricSetup", "TaskHyperparameters", "getDefaultHyperparameters",
           "Intervals", "NeverInterval", "EveryNDescents", "NEveryEpoch", "EveryNMinutes", "NeverStop", "AfterNDescents", "AfterNEpochs", "AfterNTokens", "AfterNMinutes",
           "DatasetDict", "DataCollator", "Any", "Tuple", "Path", "ModelAugmentation", "EvalPrediction"]