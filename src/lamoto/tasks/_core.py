"""
Core fine-tuning script for any task.

TODO:
    - The name of the model currently comes from the checkpoint name, but that doesn't exist yet when training from config.
      The tokeniser *always* comes from a pretrained checkpoint.
    - Stopping condition can be either never, a time (which we have a callback for!), a total amount of steps, a total amount of epochs,
      or a total amount of tokens. Tokens is mostly for CLM and only allows simple arithmetic if you use packing.
    - Should the optimisers be given as training parameters to allow the use of accelerate (and perhaps multi-GPU)?
    - Not using EXAMPLES_PER_EVALUATION right now. Is it .take() for both normal and streamed datasets? (If not, put the truncation in util.datasets)
    - I wonder if .train(resume_from_checkpoint) keeps instance-level architecture or acts like .from_pretrained() in that it resets the architecture.

# TODO: In the old code I was using for MBR, I was also saving the tokeniser at the end of training. This is still done
#       in the CLM script, but not here. The reason you want to save it is that you otherwise can't use checkpoints
#       without loading the tokeniser from the hub, so you need to pass in two checkpoints.
#       The reason why you want to be careful saving the tokeniser by giving it to Trainer is that Trainer then also uses
#       it for a variety of purposes that you might not intend (like padding etc.).
#       The call you basically just want to insert into Trainer somehow is tokenizer.save_pretrained(save_directory=current_checkpoint.as_posix()).
#       Maybe a callback will work, but you need the path of the latest checkpoint.
"""
from typing import Any, Dict, Type, List, Tuple
from pathlib import Path

import wandb
import torch
from datasets import DatasetDict, Dataset
from transformers import DataCollator, Trainer, TrainingArguments, AutoTokenizer, RobertaTokenizer, PreTrainedModel, \
    PreTrainedTokenizerBase, PretrainedConfig, AutoConfig, IntervalStrategy, EarlyStoppingCallback
import transformers.optimization
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from fiject.hooks.transformers import FijectCallback
from tktkt.files.paths import DataPaths
from tktkt.util.timing import datetimeDashed

from ..augmenting.augment_model import ModelAugmentation
from ..measuring._core import Metric, EvaluationEnvironment, LamotoMetric
from ..measuring import METRICS
from ..trainer.callbacks import EvaluateBeforeTrainingCallback, CheckpointLastModel, CallbackAtTimeInterval, EventType
from ..trainer.hyperparameters import *
from ..trainer.trainers import TrainerWithoutEvaluationLoop
from ..util.strings import getSubstringAfterLastSlash


#################################################
DEFAULT_HYPERPARAMETERS = TaskHyperparameters(
    MAX_TRAINING_EPOCHS=10,
    BATCH_SIZE=32,
    BATCHES_WARMUP=100,  # The RoBERTa paper says, for GLUE tasks, they warm up for 6% of all batches across 10 epochs. That's in the ballpark of 100 batches.
    EVALS_PER_EPOCH=9,
    EVALS_OF_PATIENCE=9,

    LEARNING_RATE=2e-5,
    L2_REGULARISATION=0.01,
    MAX_INPUT_LENGTH=514,

    CHECKPOINT="roberta-base",
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
        self.config: PretrainedConfig = None
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
    def getPredictionsAndReferences(self, eval: transformers.EvalPrediction) -> Tuple[Any,Any]:
        pass

    def computeMetrics(self, eval: transformers.EvalPrediction) -> dict:
        predictions, references = self.getPredictionsAndReferences(eval)
        results = dict()
        for metric_name, metric in self.metrics.items():
            subresults = metric.compute(predictions=predictions, references=references)
            for key, value in subresults.items():
                results[metric_name + "_" + key] = value
        return results  # To this dictionary, the eval loss will be added post-hoc.

    def sneakyLogitTransform(self, logits, labels):
        return logits

    def train(self, hyperparameters: TaskHyperparameters=DEFAULT_HYPERPARAMETERS, model_augmentation: ModelAugmentation=None, resume_from_folder: Path=None):
        model_name = getSubstringAfterLastSlash(hyperparameters.CHECKPOINT)
        global_model_identifier = model_name \
                                + ("" if not model_augmentation else ("-" + model_augmentation.name)) \
                                + f"_{self.task_name}_{datetimeDashed()}"

        # Set up paths for checkpointing
        PATH_CHECKPOINTS = DataPaths.pathToCheckpoints() / global_model_identifier
        PATH_CHECKPOINTS.mkdir(exist_ok=True, parents=True)

        # Set up the three most important missing fields for use in subsequent method calls.
        self.hyperparameters = hyperparameters
        self.tokenizer    = AutoTokenizer.from_pretrained(hyperparameters.CHECKPOINT, add_prefix_space=True)
        self.model_config = AutoConfig.from_pretrained(hyperparameters.CHECKPOINT_OR_CONFIG) if isinstance(hyperparameters.CHECKPOINT_OR_CONFIG, str) else hyperparameters.CHECKPOINT_OR_CONFIG

        # For the tokeniser: old models like GPT-2 have no pad token, but this doesn't really matter because it's
        # actually the attention mask that determines if a token is processed, so you can replace it by any token you want.
        #   https://github.com/stanford-crfm/BioMedLM/issues/4
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.config.pad_token_id = self.config.eos_token_id

        # Now that you have the tokeniser, tokenise the dataset.
        datasetdict = self.prepareDataset(self.loadDataset())

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
        model.config.pad_token_id = self.config.pad_token_id

        # ...and augment it (possibly with the tokeniser).
        if model_augmentation:
            model = model_augmentation.augment(model, self.tokenizer)
        model.to("cuda")

        # Now that the dataset and model are known, build the metrics.
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
        n_examples_to_observe = hyperparameters.TOTAL_TOKEN_BUDGET // model.config.max_position_embeddings
        n_gradient_descents = n_examples_to_observe // hyperparameters.EXAMPLES_PER_EFFECTIVE_BATCH
        n_accumulations     = hyperparameters.EXAMPLES_PER_EFFECTIVE_BATCH // (torch.cuda.device_count() * hyperparameters.EXAMPLES_PER_DEVICEBATCH)  # The amount of times, to get to one effective batch, you have to push a device batch through all devices in parallel.
        n_descents_of_warmup = int(n_gradient_descents * hyperparameters.EFFECTIVE_BATCHES_WARMUP) if isinstance(hyperparameters.EFFECTIVE_BATCHES_WARMUP, float) else hyperparameters.EFFECTIVE_BATCHES_WARMUP

        # - Intervals
        eval_interval = hyperparameters.EVAL_VS_SAVE_INTERVALS.evaluation
        if not hyperparameters.TRACK_BEST_MODEL:
            save_interval = hyperparameters.EVAL_VS_SAVE_INTERVALS.checkpointing
        else:  # Ignore it and sync with eval interval.
            if isinstance(eval_interval, NoStrategy):
                raise ValueError("You indicated that you want to track the best model, but specified no interval strategy!")
            save_interval = eval_interval

        eval_steps = eval_interval.getSteps(datasetdict["train"]) if isinstance(eval_interval, (EveryNDescents, NEveryEpoch)) else None
        save_steps = save_interval.getSteps(datasetdict["train"]) if isinstance(save_interval, (EveryNDescents, NEveryEpoch)) else None

        # - Finally get args
        training_args = TrainingArguments(
            num_train_epochs=hyperparameters.MAX_TRAINING_EPOCHS,
            max_steps=n_gradient_descents,  # Overrides `num_train_epochs`.

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
            evaluation_strategy=IntervalStrategy.STEPS if eval_steps else IntervalStrategy.NO,
            eval_steps=eval_steps,
            per_device_eval_batch_size=hyperparameters.EXAMPLES_PER_DEVICEBATCH,
            eval_accumulation_steps=n_accumulations,  # "Number of predictions steps to accumulate the output tensors for, before moving the results to the CPU."

            # Saving
            save_strategy=IntervalStrategy.STEPS if save_steps else IntervalStrategy.NO,
            save_steps=save_steps,

            output_dir=PATH_CHECKPOINTS.as_posix(),

            load_best_model_at_end=hyperparameters.TRACK_BEST_MODEL,  # Will take the best model out of its checkpoint directory and load it into self.model, which can then be saved. At the end of Trainer's loop, the following happens: "Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint"
            metric_for_best_model="eval_loss" if hyperparameters.TRACK_BEST_MODEL else None,  # TODO: Can become an issue if you don't have eval loss.
            save_total_limit=1,  # This will keep the last model stored plus the best model if those aren't the same. https://stackoverflow.com/a/67615225/9352077

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
        callbacks = [EvaluateBeforeTrainingCallback(), CheckpointLastModel()]
        if hyperparameters.TRACK_BEST_MODEL and hyperparameters.EVALS_OF_PATIENCE is not None:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=hyperparameters.EVALS_OF_PATIENCE))  # Patience is the amount of eval calls you can tolerate worsening loss.

        if isinstance(eval_interval, EveryNMinutes) and isinstance(save_interval, EveryNMinutes) and eval_interval.minutes == save_interval.minutes:
            # They are completely tied. This means you need a fully synchronised callback to prevent race conditions.
            callbacks.append(CallbackAtTimeInterval(minutes=eval_interval.minutes, events={EventType.EVALUATE, EventType.CHECKPOINT}))
        else:  # Can be neither, one, or both but with disparate minutes. Either way, you'll need a separate callback per type.
            if isinstance(eval_interval, EveryNMinutes):
                callbacks.append(CallbackAtTimeInterval(minutes=eval_interval.minutes, events={EventType.EVALUATE}))
            if isinstance(save_interval, EveryNMinutes):
                callbacks.append(CallbackAtTimeInterval(minutes=save_interval.minutes, events={EventType.CHECKPOINT}))

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
        TrainerClass = TrainerWithoutEvaluationLoop if no_traditional_metrics else Trainer
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
            compute_metrics=self.computeMetrics,
            preprocess_logits_for_metrics=self.sneakyLogitTransform
        )

        # Lastly, do some prints.
        print("=== TRAINING SIZES ===")
        bs = hyperparameters.EXAMPLES_PER_EFFECTIVE_BATCH
        e = getDatasetSize(datasetdict["train"], "train")
        ev = getDatasetSize(datasetdict["validation"], "validation")
        batches_per_epoch = 1 + (e - 1) // bs  # Final batch might be smaller than bs.
        batches_per_eval  = 1 + (ev - 1) // bs
        print("Batch size:", bs)
        print("Training set:")
        print("\t", e, "examples per epoch")
        print("\t", batches_per_epoch, "batches per epoch")
        print("\t", hyperparameters.MAX_TRAINING_EPOCHS, "epochs")
        print("\t", hyperparameters.MAX_TRAINING_EPOCHS*batches_per_epoch, "batches in total")
        print("Evaluation set:")
        print("\t", ev, "examples per evaluation")
        print("\t", batches_per_eval, "batches per evaluation")
        if eval_steps:
            print("\t", eval_steps, "training batches between evals")
            print("\t", batches_per_epoch // eval_steps, "evals per training epoch")
        print("======================")

        # Train, and evaluate afterwards.
        try:
            print("Training:")
            trainer.train(resume_from_checkpoint=resume_from_folder.as_posix() if resume_from_folder else None)
            # trainer.save_model()  # 1. We already checkpoint the last model with a callback, 2. LM pretraining basically never gets to convergence, and 3. we don't have a metric configured because we're not doing traditional eval (although this is probably not a problem since compute_metrics might be where you get your metric anyway).
            # trainer.push_to_hub()
            print("Evaluation of " + ("best" if hyperparameters.TRACK_BEST_MODEL else "last") + " model:")
            print(trainer.evaluate())
        except Exception as e:  # Catches any error that happens during training, and triggers a checkpoint (+ a callback event afterwards, if that's needed by any callback).
            print("Caught exception while training:")
            print("="*32)
            print(e)
            print("="*32)
            print("A final checkpoint will be saved.")

            trainer.control.should_save     = True
            trainer.control.should_evaluate = False
            trainer.control.should_log      = False
            trainer._maybe_log_save_evaluate(tr_loss=None, grad_norm=None, model=None, trial=None, epoch=None, ignore_keys_for_eval=None)  # These arguments are imputed anyway.
