"""
Core fine-tuning script for any task.
"""
from typing import Any, Dict, Type, List, Tuple, Generic
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass

import traceback
import time
import json
import wandb
import torch
from datasets import DatasetDict, Dataset
import transformers
from transformers import DataCollator, AutoTokenizer, PreTrainedModel, \
    PreTrainedTokenizerBase, PretrainedConfig, AutoConfig, IntervalStrategy, EarlyStoppingCallback, EvalPrediction
from transformers.models.auto.auto_factory import _BaseAutoModelClass
from transformers.trainer import TrainingArguments
from transformers.trainer_utils import has_length
import transformers.optimization
from transformers.utils.logging import set_verbosity_error

from fiject.hooks.transformers import FijectCallback
from tktkt.files.paths import PathManager
from tktkt.interfaces.huggingface import TktktToHuggingFace
from tktkt.interfaces.tokeniser import TokeniserWithFiniteTypeDomain
from tktkt.util.timing import datetimeDashed
from tktkt.util.printing import pluralise, intsep
from archit.instantiation.abstracts import ModelWithHead, CombinedConfig
from archit.util import torchPrint, parameterCountBaseVsHead

from ..augmenting.augment_model import ModelAugmentation
from ..measuring._core import Metric, LamotoMetric
from ..measuring import METRICS
from ..trainer.callbacks import *
from ..trainer.hyperparameters import *
from ..trainer.trainers import LamotoTrainer, LamotoTrainerWithoutEvaluationLoop
from ..util.datasets import shuffleAndTruncate, getDatasetSize, totalBatches
from ..util.exceptions import tryExceptNone
from ..util.strings import getSubstringAfterLastSlash
from ..util.visuals import printLamotoWelcome, log, warn

LamotoPaths = PathManager("lamoto")

DO_WARNINGS_AND_PROGRESSBARS = True
def showWarningsAndProgress(enabled: bool):
    global DO_WARNINGS_AND_PROGRESSBARS
    DO_WARNINGS_AND_PROGRESSBARS = enabled


@dataclass
class RankingMetricSpec:
    """Specification of the metric used for determining the best model, if turned on in the hyperparameters."""
    metric_name: str
    result_name: str
    higher_is_better: bool

    def fullName(self) -> str:
        return self.metric_name + "_" + self.result_name if self.metric_name else self.result_name


@dataclass
class MetricSetup:
    to_compute: List[str]               # Names of all the HuggingFace evaluate metrics to load and compute in the end.
    to_track: Dict[str, Dict[str,str]]  # metric name -> result name -> formatted name, used for graphing intermediate evaluations.
    to_rank: RankingMetricSpec = None   # Which of these to measure for finding the best model with the validation set. Defaults to loss.


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

    def sneakyLogitTransform(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return logits

    ####################################################################################################################

    def _computeMetrics(self, eval: EvalPrediction) -> dict:
        predictions, references = self.getPredictionsAndReferences(eval)
        results = dict()
        for metric_name, metric in self.metrics.items():
            subresults = metric.compute(predictions=predictions, references=references)
            for key, value in subresults.items():
                results[metric_name + "_" + key] = value

        # Sanity checks
        if self.metric_config.to_rank.fullName() != "loss" and self.metric_config.to_rank.fullName() not in results:
            raise RuntimeError(f"The ranking metric '{self.metric_config.to_rank.metric_name}' did not compute the required result '{self.metric_config.to_rank.result_name}'. Results we did compute: {results}")
        for metric_name, result_names in self.metric_config.to_track.items():
            for result_name in result_names:
                if metric_name + "_" + result_name not in results:
                    warn(f"Metric '{metric_name}' did not compute the tracked result '{result_name}'.")

        return results  # To this dictionary, the eval loss will be added post-hoc, and all keys will be prefixed by "eval_".

    def _setHyperparameters(self, hp: TaskHyperparameters[HC]):
        self.hyperparameters = hp
    def _setModelConfig(self, mc: PretrainedConfig):
        self.model_config = mc
    def _setMetrics(self, m: Dict[str, Metric]):
        self.metrics = m
    def _setTokenizer(self, tk: PreTrainedTokenizerBase):
        self.tokenizer = tk

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

    def train(self, hyperparameters: TaskHyperparameters[HC]=getDefaultHyperparameters(), model_augmentation: ModelAugmentation=None, resume_from_folder: Path=None) -> Dict[str, float]:
        """
        Encapsulation of everything you need to do to get a Trainer running.
        """
        printLamotoWelcome()
        log("Running task:", self.__class__.__name__)
        transformers.set_seed(seed=hyperparameters.SEED)
        if not DO_WARNINGS_AND_PROGRESSBARS:
            set_verbosity_error()

        # Imputations and sanity checks
        if isinstance(hyperparameters.MODEL_CONFIG_OR_CHECKPOINT, Path):
            hyperparameters.MODEL_CONFIG_OR_CHECKPOINT = hyperparameters.MODEL_CONFIG_OR_CHECKPOINT.as_posix()  # FIXME: Possibly have to make it a relative path due to HF restrictions.
        if self.metric_config.to_rank is None:
            self.metric_config.to_rank = RankingMetricSpec(metric_name="", result_name="loss", higher_is_better=False)

        if hyperparameters.init_weights and not isinstance(hyperparameters.MODEL_CONFIG_OR_CHECKPOINT, str):
            raise ValueError("You said you wanted to initialise model weights from the checkpoint, but didn't give a checkpoint path!")
        if hyperparameters.archit_basemodel_class is None:
            raise ValueError("In order to parse model configs, the archit_basemodel_class hyperparameter cannot be None.")
        if hyperparameters.archit_head_config is None and not isinstance(hyperparameters.MODEL_CONFIG_OR_CHECKPOINT, str):  # Note: there is another failure case: when the checkpoint *is* a string, but *isn't* an ArchIt checkpoint. It errors below.
            raise ValueError("Without a checkpoint, a head config must be provided to instantiate a new head.")
        if hyperparameters.TRACK_BEST_MODEL and self.metric_config.to_rank.fullName() != "loss" and self.metric_config.to_rank.metric_name not in self.metric_config.to_compute:
            raise ValueError(f"Cannot rank models based on metric {self.metric_config.to_rank.metric_name} since it isn't computed.")
        for metric_name in self.metric_config.to_track.keys():
            if metric_name not in self.metric_config.to_compute:
                raise ValueError(f"Requested tracking results for metrics {sorted(self.metric_config.to_track)} yet you are only computing metrics {sorted(self.metric_config.to_compute)}.")

        # Metadata
        self.adjustHyperparameters(hyperparameters)
        self._setHyperparameters(hyperparameters)

        log("Loading model config...")
        config_or_str = hyperparameters.MODEL_CONFIG_OR_CHECKPOINT
        if not isinstance(config_or_str, str):  # It's an object.
            if isinstance(config_or_str, CombinedConfig):
                raise ValueError("When instantiating a new model from a config, it must only parameterise the base model. The head is parameterised in the hyperparameters.")
            model_config = CombinedConfig(base_model_config=config_or_str,
                                          head_config=hyperparameters.archit_head_config,
                                          base_model_config_class=hyperparameters.archit_basemodel_class.config_class)  # This call pretends to be CombinedConfig(**json).
        else:  # It's a checkpoint string. Can either be a checkpoint for the ModelWithHead we're about to load, or for anything else compatible. We'll figure that out.
            model_config = CombinedConfig.from_pretrained(config_or_str,
                                                          head_config=hyperparameters.archit_head_config,
                                                          base_model_config_class=hyperparameters.archit_basemodel_class.config_class)  # Note that there is no need for AutoConfig because we KNOW the config class (even if not registered in AutoConfig). Also means we don't have to store the "model type" in the config.
        self._setModelConfig(model_config)

        if hyperparameters.SAVE_AS:
            model_name = hyperparameters.SAVE_AS
        elif isinstance(hyperparameters.MODEL_CONFIG_OR_CHECKPOINT, str):
            model_name = getSubstringAfterLastSlash(hyperparameters.MODEL_CONFIG_OR_CHECKPOINT)
        else:  # We don't use the tokeniser name because it isn't directly related to the model.
            raise RuntimeError("Cannot deduce name to save model as from a config.")

        global_model_identifier = model_name \
                                + ("" if not model_augmentation else ("-" + model_augmentation.name)) \
                                + f"_{self.task_name}_{datetimeDashed()}"

        folder_to_this_models_checkpoints = LamotoPaths.append(LamotoPaths.pathToCheckpoints(), global_model_identifier)

        # Set up tokeniser
        log("Loading tokeniser...")
        if hyperparameters.TOKENISER:
            if isinstance(hyperparameters.TOKENISER, str):
                tokenizer = AutoTokenizer.from_pretrained(hyperparameters.TOKENISER, add_prefix_space=True)
            elif isinstance(hyperparameters.TOKENISER, TokeniserWithFiniteTypeDomain):
                tokenizer = TktktToHuggingFace(hyperparameters.TOKENISER)
            else:
                tokenizer = hyperparameters.TOKENISER
        elif isinstance(hyperparameters.MODEL_CONFIG_OR_CHECKPOINT, str):
            tokenizer = AutoTokenizer.from_pretrained(hyperparameters.MODEL_CONFIG_OR_CHECKPOINT, add_prefix_space=True)
        else:
            raise RuntimeError("Cannot deduce tokeniser checkpoint from config.")
        self._setTokenizer(tokenizer)

        # - Old models like GPT-2 have no pad token, but this doesn't really matter because it's actually the attention
        #   mask that determines if a token is processed, so you can replace it by any token you want. https://github.com/stanford-crfm/BioMedLM/issues/4
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token                         = self.tokenizer.eos_token
            self.model_config.base_model_config.pad_token_id = self.tokenizer.eos_token_id

        # Now that you have the tokeniser, tokenise the dataset.
        log("Loading dataset...")
        datasetdict = self.loadDataset()
        n_examples_validation = tryExceptNone(lambda: getDatasetSize(datasetdict["validation"], split="validation")) or 1_000_000_000_000  # Very very big number assumed when you can't find the dataset size.
        n_examples_validation = n_examples_validation if not hyperparameters.EXAMPLES_PER_EVALUATION else min(n_examples_validation, hyperparameters.EXAMPLES_PER_EVALUATION)

        log("Preparing dataset...")
        datasetdict["train"]      = shuffleAndTruncate(datasetdict["train"], seed=hyperparameters.SEED)
        datasetdict["validation"] = shuffleAndTruncate(datasetdict["validation"], seed=hyperparameters.SEED, truncate_to=n_examples_validation)
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
                assert hyperparameters.archit_head_config is not None, "You forgot to set the head config in the hyperparameters!"
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
                raise RuntimeError()
        model.config.pad_token_id = self.model_config.base_model_config.pad_token_id  # self.model_config might have been changed since AutoConfig.from_pretrained() was called, whereas model.config is the result of a fresh AutoConfig call.

        # ...and augment it in-place (possibly with the tokeniser). We assume the augmentation uses .base_model when it needs to.
        if model_augmentation:
            model_augmentation.augment(model, self.tokenizer)
        model.to("cuda")

        # Now that we have a reference to the dataset and model, build the metrics.
        env = EvaluationEnvironment(
            model=model,
            tokeniser=self.tokenizer,
            validation_dataset=datasetdict["validation"],
            hyperparameters=self.hyperparameters
        )
        self._setMetrics({name: METRICS.load(name,env) for name in self.metric_config.to_compute})

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
        n_gradient_descents = tryExceptNone(lambda: stopping_condition.getSteps(datasetdict["train"])) if not isinstance(stopping_condition, (NeverStop, AfterNMinutes)) else None
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

        batches_between_evals = tryExceptNone(lambda: eval_interval.getSteps(datasetdict["train"])) if isinstance(eval_interval, (EveryNDescents, EveryNDescentsOrOncePerEpoch, NEveryEpoch)) else None
        batches_between_saves = tryExceptNone(lambda: save_interval.getSteps(datasetdict["train"])) if isinstance(save_interval, (EveryNDescents, EveryNDescentsOrOncePerEpoch, NEveryEpoch)) else None

        # - Early stopping (only used if required)
        best_model_metric_handle = f"eval_{self.metric_config.to_rank.fullName()}" if hyperparameters.TRACK_BEST_MODEL else None

        # - Finally get args
        training_args = TrainingArguments(
            max_steps=(n_gradient_descents or -1) if n_gradient_descents or has_length(datasetdict["train"]) else 1_000_000_000_000,  # Handle a very specific illegal case according to HF. Only reason it exists is for learning rate schedules that decrease relative to the max amount of descents, but we don't use those schedules.

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
            eval_on_start=not isinstance(eval_interval, NeverInterval),  # Always do an evaluation at the start, unless you wanted to avoid all evaluations.
            evaluation_strategy=IntervalStrategy.STEPS if batches_between_evals else IntervalStrategy.NO,
            eval_steps=batches_between_evals,
            per_device_eval_batch_size=hyperparameters.EXAMPLES_PER_DEVICEBATCH,
            eval_accumulation_steps=n_accumulations,  # "Number of predictions steps to accumulate the output tensors for, before moving the results to the CPU."

            # Saving
            save_strategy=IntervalStrategy.STEPS if batches_between_saves else IntervalStrategy.NO,
            save_steps=batches_between_saves,

            output_dir=folder_to_this_models_checkpoints.as_posix(),

            load_best_model_at_end=hyperparameters.TRACK_BEST_MODEL,  # Will take the best model out of its checkpoint directory and load it into self.model, which can then be saved. At the end of Trainer's loop, the following happens: "Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint"
            metric_for_best_model=best_model_metric_handle,
            greater_is_better=self.metric_config.to_rank.higher_is_better,
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
        optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparameters.learning_rate, weight_decay=hyperparameters.adamw_decay_rate, betas=(0.9, 0.999))  # Not using transformers.optimization because it gives a deprecation warning.
        scheduler = transformers.optimization.get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=n_descents_of_warmup)  # Not using a linear decay because that's the whole point of having Adam.

        # - Build callbacks
        callbacks = [CheckpointLastModel(), SaveTokeniserWithCheckpoints(self.tokenizer)]
        if hyperparameters.TRACK_BEST_MODEL and hyperparameters.EVALS_OF_PATIENCE is not None:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=hyperparameters.EVALS_OF_PATIENCE))  # Patience is the amount of eval calls you can tolerate worsening loss.

        # if not isinstance(eval_interval, NeverInterval):  # Didn't work, but has since become an option that works. https://discuss.huggingface.co/t/how-to-evaluate-before-first-training-step/18838
        #     callbacks.append(EvaluateBeforeTrainingCallback())

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
                callbacks.append(FijectCallback(global_model_identifier + "_eval_goal", evals_between_commits=4))  # Automatically tracks the same metric as is used to decide best model.
            callbacks.append(
                FijectCallback(global_model_identifier + "_eval_tracked",
                               evals_between_commits=4,
                               metric_names_with_formatting={(metric_name + "_" + result_name): formatting
                                                             for metric_name, result_formats in
                                                             self.metric_config.to_track.items()
                                                             for result_name, formatting in result_formats.items()})
            )

        # At last, the Trainer object.
        if hyperparameters.TRACK_BEST_MODEL and best_model_metric_handle == "eval_loss":
            no_traditional_metrics = False  # A "traditional metric" is a metric that (1) uses prediction logits that (2) can come from naive iteration over the evaluation set (unlike strided PPL, for example, which iterates in a special manner).
        else:
            no_traditional_metrics = all(isinstance(m, LamotoMetric) and m.isAutonomous() for m in self.metrics.values())
        TrainerClass = LamotoTrainerWithoutEvaluationLoop if no_traditional_metrics else LamotoTrainer
        trainer = TrainerClass(
            model=model,
            # tokenizer=self.tokenizer,  # Don't pass it if you don't want to save it and have other wacky shit extracted from it to influence training.

            # Args
            args=training_args,
            optimizers=(optimizer, scheduler),
            callbacks=callbacks,

            # Data
            train_dataset=datasetdict["train"],
            eval_dataset=[] if no_traditional_metrics else datasetdict["validation"],
            data_collator=collator,

            # Evaluation
            compute_metrics=self._computeMetrics,
            preprocess_logits_for_metrics=self.sneakyLogitTransform
        )

        # Lastly, do some prints (not logs).
        # Print the loaded model and a breakdown of its parameter counts.
        log("="*17 + "ARCHITECTURE" + "="*17)
        torchPrint(model)
        (p_base_train, p_base_total), (p_head_train, p_head_total) = parameterCountBaseVsHead(model)
        print("Parameter counts:")
        print("|-- Base model:")
        print("|   |-- Trainable:", intsep(p_base_train))
        print("|   `-------- All:", intsep(p_base_total))
        print("|-- Head:")
        print("|   |-- Trainable:", intsep(p_head_train))
        print("|   `-------- All:", intsep(p_head_total))
        print("`-- Total:")
        print("    |-- Trainable:", intsep(p_base_train + p_head_train))
        print("    `-------- All:", intsep(p_base_total + p_head_total))
        print()

        print("="*17 + " TRAINING SIZES " + "="*17)
        batch_size = hyperparameters.EXAMPLES_PER_EFFECTIVE_BATCH
        n_examples_training = tryExceptNone(lambda: getDatasetSize(datasetdict["train"], "train"))
        print("Batch size:", pluralise(batch_size, "example"))
        print("Context length:", pluralise(self._getMaxInputLength(), "token"))

        print("Training set:")
        if n_examples_training:
            batches_per_epoch = totalBatches(n_examples_training, batch_size)
            print("\t", pluralise(n_examples_training, "example"), "per epoch")
            print("\t", pluralise(batches_per_epoch, "batch", "es"), "per epoch")
            if n_gradient_descents:
                print("\t", round(n_gradient_descents / batches_per_epoch, 1), "epochs")
                print("\t", pluralise(n_gradient_descents, "batch", "es"), "in total")
        else:
            batches_per_epoch = 0
            print("\t", "No sizes known.")

        print("Validation set:")
        if n_examples_validation:
            batches_per_eval = totalBatches(n_examples_validation, batch_size)
            print("\t", pluralise(n_examples_validation, "example"), "per evaluation")
            print("\t", pluralise(batches_per_eval, "batch", "es"), "per evaluation")
            if batches_between_evals:
                print("\t", pluralise(batches_between_evals, "training batch", "es"), "between evals")
            if batches_per_epoch and batches_between_evals:
                print("\t", pluralise(batches_per_epoch // batches_between_evals, "eval"), "per training epoch")
        else:
            print("\t", "No sizes known.")
        print("="*50)

        # Train, and evaluate afterwards.
        try:
            log(f"Training model {model.__class__.__name__} on task {self.__class__.__name__} on device {model.device}...")
            trainer.train(resume_from_checkpoint=resume_from_folder.as_posix() if resume_from_folder else None)
            # trainer.save_model()  # 1. We already checkpoint the last model with a callback, 2. LM pretraining basically never gets to convergence, and 3. we don't have a metric configured because we're not doing traditional eval (although this is probably not a problem since compute_metrics might be where you get your metric anyway).
            # trainer.push_to_hub()
            log("Evaluation of " + ("best" if hyperparameters.TRACK_BEST_MODEL else "last") + " model on validation set...")
            validation_results = trainer.evaluate(datasetdict["validation"], metric_key_prefix="eval")
            print(validation_results)
            log("Evaluation of " + ("best" if hyperparameters.TRACK_BEST_MODEL else "last") + " model on test set...")
            test_results = trainer.evaluate(datasetdict["test"], metric_key_prefix="test")
            print(test_results)
            wandb.finish()  # Finish because otherwise, running .train() in the same process after .init() has been called once already will raise an error.
            log("*** SUCCESSFULLY FINISHED LaMoTO TRAINING ***")
            all_results = validation_results | test_results
            with open(LamotoPaths.append(LamotoPaths.pathToEvaluations(), global_model_identifier) / f"metrics-{trainer.state.global_step}.json", "w", encoding="utf-8") as handle:
                json.dump(all_results, handle, indent=4)
            return all_results

        except Exception as e1:  # Catches any error that happens during training, and triggers a checkpoint (+ a callback event afterwards, if that's needed by any callback).
            log("Caught exception while training. A checkpoint will be saved.\nAfterwards, we will raise the exception, so your run shows up as failed rather than completed.")
            trainer.control.should_save     = True
            trainer.control.should_evaluate = False
            trainer.control.should_log      = False
            try:
                trainer._maybe_log_save_evaluate(tr_loss=None, grad_norm=None, model=None, trial=None, epoch=None, ignore_keys_for_eval=None)  # These arguments are imputed anyway.
                log("Save successful. Now raising the exception. Bye bye!")
            except Exception as e2:
                log("Save FAILED. Something is broken. Raising all exceptions.")
                log("=" * 50)
                wandb.finish(exit_code=1)
                time.sleep(1)  # First let all the prints happen, so that the traceback doesn't race it to the output.
                raise e2  # Automatically prints the traceback.

            log("=" * 50)
            wandb.finish(exit_code=1)
            time.sleep(1)
            raise e1


class TaskWrapper(Task[HC]):
    """
    A task which, by default, steals all the implementations from an underlying task.
    """

    def __init__(self, task: Task[HC]):
        super().__init__(
            task_name=task.task_name,
            metric_config=task.metric_config,
            archit_class=task.archit_class,
            automodel_class=task.automodel_class,
            **task.automodel_args
        )
        self._method_implementations: Task[HC] = task

    def loadDataset(self) -> DatasetDict:
        return self._method_implementations.loadDataset()

    def prepareDataset(self, dataset: DatasetDict) -> DatasetDict:
        return self._method_implementations.prepareDataset(dataset)

    def getCollator(self) -> DataCollator:
        return self._method_implementations.getCollator()

    def adjustHyperparameters(self, hp: TaskHyperparameters[HC]):
        return self._method_implementations.adjustHyperparameters(hp)

    def getPredictionsAndReferences(self, eval: EvalPrediction) -> Tuple[Any,Any]:
        return self._method_implementations.getPredictionsAndReferences(eval)

    def sneakyLogitTransform(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self._method_implementations.sneakyLogitTransform(logits, labels)

    # Finally, four methods to communicate the runtime fields with the underlying task, so it can use them in its implementations:

    def _setHyperparameters(self, hp: TaskHyperparameters[HC]):
        super()._setHyperparameters(hp)
        self._method_implementations._setHyperparameters(hp)

    def _setMetrics(self, m: Dict[str, Metric]):
        super()._setMetrics(m)
        self._method_implementations._setMetrics(m)

    def _setModelConfig(self, mc: PretrainedConfig):
        super()._setModelConfig(mc)
        self._method_implementations._setModelConfig(mc)

    def _setTokenizer(self, tk: PreTrainedTokenizerBase):
        super()._setTokenizer(tk)
        self._method_implementations._setTokenizer(tk)


__all__ = ["Task", "MetricSetup", "RankingMetricSpec", "TaskHyperparameters", "getDefaultHyperparameters", "TaskWrapper",
           "Intervals", "NeverInterval", "EveryNDescents", "NEveryEpoch", "EveryNMinutes", "NeverStop", "AfterNDescents", "AfterNEpochs", "AfterNTokens", "AfterNMinutes",
           "DatasetDict", "DataCollator", "Any", "Tuple", "Path", "ModelAugmentation", "EvalPrediction"]