from typing import Dict
from pathlib import Path

import json
import time
import torch
import wandb
import transformers
from transformers import PreTrainedModel, TrainingArguments, IntervalStrategy, EarlyStoppingCallback, AutoTokenizer
from transformers.trainer_utils import has_length
from transformers.utils.logging import set_verbosity_error
from huggingface_hub.constants import HF_HUB_CACHE

from archit.instantiation.abstracts import CombinedConfig
from archit.util import torchPrint, parameterCountBaseVsHead
from fiject.hooks.transformers import FijectCallback
from tktkt.interfaces.huggingface import TktktToHuggingFace
from tktkt.interfaces.tokeniser import TokeniserWithFiniteTypeDomain
from tktkt.util.printing import intsep, pluralise
from tktkt.util.timing import datetimeDashed
from tktkt.files.paths import PathManager

from ..tasks._core import Task, RankingMetricSpec
from ..augmenting.augment_model import ModelAugmentation
from ..measuring import METRICS
from ..measuring._core import LamotoMetric
from .auxiliary.callbacks import CallbackAtTimeInterval, SaveTokeniserWithCheckpoints, CheckpointLastModel, EventType
from .auxiliary.hyperparameters import *
from .auxiliary.backends import ModelTrainer, ModelTrainerWithoutEvaluationLoop
from ..util.datasets import shuffleAndTruncate, getDatasetSize, totalBatches
from ..util.exceptions import tryExceptNone, ImpossibleBranchError
from ..util.strings import getSubstringAfterLastSlash
from ..util.visuals import log, printLamotoWelcome


LamotoPaths = PathManager("lamoto")


DO_WARNINGS_AND_PROGRESSBARS = True
def showWarningsAndProgress(enabled: bool):
    global DO_WARNINGS_AND_PROGRESSBARS
    DO_WARNINGS_AND_PROGRESSBARS = enabled


class TaskTrainer:

    def train(
        self,
        task: Task,
        hyperparameters: TaskHyperparameters[HC]=getDefaultHyperparameters(),
        model_augmentation: ModelAugmentation=None,
        resume_from_folder: Path=None
    ) -> Dict[str, float]:
        """
        Encapsulation of everything you need to do to get a Trainer running.
        """
        printLamotoWelcome()
        log("Running task:", task.__class__.__name__)
        transformers.set_seed(seed=hyperparameters.SEED)
        if not DO_WARNINGS_AND_PROGRESSBARS:
            set_verbosity_error()

        # Imputations and sanity checks
        if isinstance(hyperparameters.MODEL_CONFIG_OR_CHECKPOINT, Path):
            hyperparameters.MODEL_CONFIG_OR_CHECKPOINT = hyperparameters.MODEL_CONFIG_OR_CHECKPOINT.as_posix()  # FIXME: Possibly have to make it a relative path due to HF restrictions.
        if task.metric_config.to_rank is None:
            task.metric_config.to_rank = RankingMetricSpec(metric_name="", result_name="loss", higher_is_better=False)

        if hyperparameters.init_weights and not isinstance(hyperparameters.MODEL_CONFIG_OR_CHECKPOINT, str):
            raise ValueError("You said you wanted to initialise model weights from the checkpoint, but didn't give a checkpoint path!")
        if hyperparameters.archit_basemodel_class is None:
            raise ValueError("In order to parse model configs, the archit_basemodel_class hyperparameter cannot be None.")
        if hyperparameters.archit_head_config is None and not isinstance(hyperparameters.MODEL_CONFIG_OR_CHECKPOINT, str):  # Note: there is another failure case: when the checkpoint *is* a string, but *isn't* an ArchIt checkpoint. It errors below.
            raise ValueError("Without a checkpoint, a head config must be provided to instantiate a new head.")
        if hyperparameters.TRACK_BEST_MODEL and task.metric_config.to_rank.fullName() != "loss" and task.metric_config.to_rank.metric_name not in task.metric_config.to_compute:
            raise ValueError(f"Cannot rank models based on metric {task.metric_config.to_rank.metric_name} since it isn't computed.")
        for metric_name in task.metric_config.to_track.keys():
            if metric_name not in task.metric_config.to_compute:
                raise ValueError(f"Requested tracking results for metrics {sorted(task.metric_config.to_track)} yet you are only computing metrics {sorted(task.metric_config.to_compute)}.")

        # Imputation of device batch size specifically
        n_devices = torch.cuda.device_count()
        if hyperparameters.EXAMPLES_PER_EFFECTIVE_BATCH % n_devices != 0:  # This is an unsolvable issue by setting a new device batch size.
            raise ValueError(f"Effective batch size ({hyperparameters.EXAMPLES_PER_EFFECTIVE_BATCH}) must be a multiple of the amount of devices ({n_devices}).")

        n_examples_per_pass_per_device = hyperparameters.EXAMPLES_PER_DEVICEBATCH
        n_examples_per_pass            = n_examples_per_pass_per_device * n_devices
        if n_examples_per_pass > hyperparameters.EXAMPLES_PER_EFFECTIVE_BATCH:  # One pass through your devices already exceeds one effective batch.
            n_examples_per_pass_per_device = hyperparameters.EXAMPLES_PER_EFFECTIVE_BATCH // n_devices  # Suggest a new device batch size.
            n_examples_per_pass            = hyperparameters.EXAMPLES_PER_EFFECTIVE_BATCH
        if hyperparameters.EXAMPLES_PER_EFFECTIVE_BATCH % n_examples_per_pass != 0:  # You can't push an integer amount of passes through your devices to get to the effective batch size.
            # Example: effective batch size 96, 4 devices, device batch size 7  =>  28 per pass whilst you need a total of 24.
            # Because we get to choose the device batch size but not the effective batch size or amount of device, we choose a new one as follows:
            #   - The base criterion is that effective batch size % (device size * devices) == 0, thus effective batch size == device size * devices * integer.
            #   - It has to be smaller than whatever device batch size we have currently.
            #   - It has to be a power of 2.
            #   - It has to be as high as possible given the above.
            # In the example: we want to get to 96//4 == 24 examples with each device, so push 24 examples for 1 pass, 12 for 2 passes, 8 for 3 passes, 6 for 4 passes, 4 for 6 passes, 3 for 8 passes, 2 for 12 passes or 1 for 24 passes.
            # The best of these is size 4 for 6 passes because it is smaller than the given 7 whilst being the highest possible power of 2.
            n_examples_per_device = hyperparameters.EXAMPLES_PER_EFFECTIVE_BATCH // n_devices
            for n_passes in range(1, n_examples_per_device+1):
                if n_examples_per_device % n_passes == 0:
                    n_examples_per_pass_per_device = n_examples_per_device // n_passes
                    if n_examples_per_pass_per_device & (n_examples_per_pass_per_device-1) == 0:  # The only numbers for which -1 flips all the bits are powers of 2.
                        break
            else:
                raise ImpossibleBranchError()
            n_examples_per_pass = n_devices * n_examples_per_pass_per_device
        n_passes = hyperparameters.EXAMPLES_PER_EFFECTIVE_BATCH // n_examples_per_pass

        # Metadata
        task.adjustHyperparameters(hyperparameters)
        task._setHyperparameters(hyperparameters)

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
        task._setModelConfig(model_config)

        if hyperparameters.SAVE_AS:
            model_name = hyperparameters.SAVE_AS
        elif isinstance(hyperparameters.MODEL_CONFIG_OR_CHECKPOINT, str):
            model_name = getSubstringAfterLastSlash(hyperparameters.MODEL_CONFIG_OR_CHECKPOINT)
        else:  # We don't use the tokeniser name because it isn't directly related to the model.
            raise RuntimeError("Cannot deduce name to save model as from a config.")

        global_model_identifier = model_name \
                                + ("" if not model_augmentation else ("-" + model_augmentation.name)) \
                                + f"_{task.task_name}_{datetimeDashed()}"

        if hyperparameters.store_in_hf_cache:
            folder_to_this_models_checkpoints = LamotoPaths.append(Path(HF_HUB_CACHE), global_model_identifier)  # If that first path doesn't exist yet, it will be created automatically.
        else:
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
        task._setTokenizer(tokenizer)

        # - Old models like GPT-2 have no pad token, but this doesn't really matter because it's actually the attention
        #   mask that determines if a token is processed, so you can replace it by any token you want. https://github.com/stanford-crfm/BioMedLM/issues/4
        if task.tokenizer.pad_token is None:
            task.tokenizer.pad_token                         = task.tokenizer.eos_token
            task.model_config.base_model_config.pad_token_id = task.tokenizer.eos_token_id

        # Now that you have the tokeniser, tokenise the dataset.
        log("Loading dataset...")
        datasetdict = task.loadDataset()
        n_examples_validation = tryExceptNone(lambda: getDatasetSize(datasetdict["validation"], split="validation")) or 1_000_000_000_000  # Very very big number assumed when you can't find the dataset size.
        n_examples_validation = n_examples_validation if not hyperparameters.EXAMPLES_PER_EVALUATION else min(n_examples_validation, hyperparameters.EXAMPLES_PER_EVALUATION)

        log("Preparing dataset...")
        datasetdict["train"]      = shuffleAndTruncate(datasetdict["train"], seed=hyperparameters.SEED)
        datasetdict["validation"] = shuffleAndTruncate(datasetdict["validation"], seed=hyperparameters.SEED, truncate_to=n_examples_validation)
        datasetdict = task.prepareDataset(datasetdict)

        # Get the batch generator, a.k.a. collator (https://huggingface.co/docs/transformers/main_classes/data_collator).
        collator = task.getCollator()

        # Get the model...  FIXME: Somehow, the DP head uses an operation that doesn't exist for bfloat16. Temporary fix below.
        task.automodel_args["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() and task.__class__.__name__ != "DP" else torch.float32

        hf_checkpoint_classname = task.model_config.architectures[0] if task.model_config.architectures is not None else ""  # Always present and correct for HuggingFace configs.
        is_exact_hf_checkpoint    = hyperparameters.init_weights and hyperparameters.load_hf_automodel_if_hf_checkpoint_and_matches_task and task._isHfCheckpointForThisTask(hf_checkpoint_classname)
        is_custom_hf_architecture = hyperparameters.custom_hf_class is not None
        if not is_exact_hf_checkpoint and not is_custom_hf_architecture:  # Use ArchIt. This is the usual case.
            log("Instantiating an ArchIt model.")
            torch.set_default_dtype(task.automodel_args["torch_dtype"])
            if hyperparameters.init_weights:
                # FIXME: This branch may be broken in case the checkpoint is an ArchIt checkpoint, see the FIXME under .from_pretrained().
                model: PreTrainedModel = task.archit_class.from_pretrained(hyperparameters.MODEL_CONFIG_OR_CHECKPOINT, hyperparameters.archit_basemodel_class, hyperparameters.archit_head_config)
            else:
                assert hyperparameters.archit_head_config is not None, "You forgot to set the head config in the hyperparameters!"
                model: PreTrainedModel = task.archit_class.fromModelAndHeadConfig(hyperparameters.archit_basemodel_class.from_config(task.model_config), hyperparameters.archit_head_config)
        else:  # Edge cases.
            if is_custom_hf_architecture:
                log("Instantiating a custom HuggingFace class.")
                if hyperparameters.init_weights:  # model_config_or_checkpoint is a string
                    model: PreTrainedModel = hyperparameters.custom_hf_class.from_pretrained(hyperparameters.MODEL_CONFIG_OR_CHECKPOINT, **task.automodel_args)
                else:
                    model: PreTrainedModel = hyperparameters.custom_hf_class._from_config(task.model_config.base_model_config, **task.automodel_args)
            elif is_exact_hf_checkpoint:  # model_config_or_checkpoint is a string
                log(f"The given checkpoint seems to be a HuggingFace architecture ({hf_checkpoint_classname}) for this specific task ({task.archit_class.__name__}),\nwe will instantiate the model with AutoModel ({task.automodel_class.__name__}) instead of ArchIt.")
                model: PreTrainedModel = task.automodel_class.from_pretrained(hyperparameters.MODEL_CONFIG_OR_CHECKPOINT, **task.automodel_args)
            else:
                raise ImpossibleBranchError()
        model.config.pad_token_id = task.model_config.base_model_config.pad_token_id  # task.model_config might have been changed since AutoConfig.from_pretrained() was called, whereas model.config is the result of a fresh AutoConfig call.

        # ...and augment it in-place (possibly with the tokeniser). We assume the augmentation uses .base_model when it needs to.
        if model_augmentation:
            model_augmentation.augment(model, task.tokenizer)
        model.to("cuda")

        # Now that we have a reference to the dataset and model, build the metrics.
        env = EvaluationEnvironment(
            model=model,
            tokeniser=task.tokenizer,
            validation_dataset=datasetdict["validation"],
            hyperparameters=task.hyperparameters
        )
        task._setMetrics({name: METRICS.load(name,env) for name in task.metric_config.to_compute})

        # Set up reporting too
        folder_wandb = folder_to_this_models_checkpoints / "wandb"
        folder_wandb.mkdir(exist_ok=True)
        wandb.init(
            mode="disabled" if hyperparameters.traceless or not hyperparameters.WANDB_PROJECT else "online",

            project=hyperparameters.WANDB_PROJECT,
            group=model_name,
            name=global_model_identifier,
            tags=[task.task_name] + ([model_augmentation.name] if model_augmentation else []),

            dir=folder_wandb.as_posix()
        )

        # Training arguments
        # - Sizes
        stopping_condition = hyperparameters.HARD_STOPPING_CONDITION
        n_gradient_descents = tryExceptNone(lambda: stopping_condition.getSteps(batch_size=hyperparameters.EXAMPLES_PER_EFFECTIVE_BATCH,
                                                                                dataset=datasetdict["train"], split_name="train"))
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
            if isinstance(eval_interval, Never):
                raise ValueError("You indicated that you want to track the best model, but specified no evaluation interval!")
            save_interval = eval_interval

        batches_between_evals = tryExceptNone(lambda: eval_interval.getSteps(batch_size=hyperparameters.EXAMPLES_PER_EFFECTIVE_BATCH,
                                                                             dataset=datasetdict["train"], split_name="train"))
        batches_between_saves = tryExceptNone(lambda: save_interval.getSteps(batch_size=hyperparameters.EXAMPLES_PER_EFFECTIVE_BATCH,
                                                                             dataset=datasetdict["train"], split_name="train"))

        # - Early stopping (only used if required)
        best_model_metric_handle = f"eval_{task.metric_config.to_rank.fullName()}" if hyperparameters.TRACK_BEST_MODEL else None

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
            per_device_train_batch_size=n_examples_per_pass_per_device,
            gradient_accumulation_steps=n_passes,

            # Style of computations
            gradient_checkpointing=model.supports_gradient_checkpointing,  # Only if you have the VRAM though. Good explanation with animations: https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9
            bf16=torch.cuda.is_bf16_supported(),

            # Evaluation
            eval_on_start=not isinstance(eval_interval, Never),  # Always do an evaluation at the start, unless you wanted to avoid all evaluations.
            evaluation_strategy=IntervalStrategy.STEPS if batches_between_evals else IntervalStrategy.NO,
            eval_steps=batches_between_evals,
            per_device_eval_batch_size=hyperparameters.EXAMPLES_PER_DEVICEBATCH,  # We know that the GPU can handle at least this much data during eval if it can during training, since training additionally requires the gradients and optimiser to be stored in VRAM as overhead.
            eval_accumulation_steps=1,  # "Number of predictions steps to accumulate the output tensors for, before moving the results to the CPU. If left unset, all predictions are accumulated on GPU before being moved to the CPU (faster but requires more GPU memory)." You always need more RAM than VRAM, of course.

            # Saving
            save_strategy=IntervalStrategy.STEPS if batches_between_saves else IntervalStrategy.NO,
            save_steps=batches_between_saves,

            output_dir=folder_to_this_models_checkpoints.as_posix(),

            load_best_model_at_end=hyperparameters.TRACK_BEST_MODEL,  # Will take the best model out of its checkpoint directory and load it into task.model, which can then be saved. At the end of Trainer's loop, the following happens: "Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint"
            metric_for_best_model=best_model_metric_handle,
            greater_is_better=task.metric_config.to_rank.higher_is_better,
            save_total_limit=1,  # This will keep the last model stored plus the best model if those aren't the same, allowing you to have the best model and continue training from last if you need to. https://stackoverflow.com/a/67615225/9352077

            # Logging
            report_to=["wandb"],  # Can be turned off above.
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
        callbacks = [CheckpointLastModel(), SaveTokeniserWithCheckpoints(task.tokenizer)]
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

        if not hyperparameters.traceless and not hyperparameters.WANDB_PROJECT:
            if hyperparameters.TRACK_BEST_MODEL:
                callbacks.append(FijectCallback(global_model_identifier + "_eval_goal", evals_between_commits=4))  # Automatically tracks the same metric as is used to decide best model.
            callbacks.append(
                FijectCallback(global_model_identifier + "_eval_tracked",
                               evals_between_commits=4,
                               metric_names_with_formatting={(metric_name + "_" + result_name): formatting
                                                             for metric_name, result_formats in
                                                             task.metric_config.to_track.items()
                                                             for result_name, formatting in result_formats.items()})
            )

        # At last, the Trainer object.
        if hyperparameters.TRACK_BEST_MODEL and best_model_metric_handle == "eval_loss":
            no_traditional_metrics = False  # A "traditional metric" is a metric that (1) uses prediction logits that (2) can come from naive iteration over the evaluation set (unlike strided PPL, for example, which iterates in a special manner).
        else:
            no_traditional_metrics = all(isinstance(m, LamotoMetric) and m.isAutonomous() for m in task.metrics.values())
        TrainerClass = ModelTrainerWithoutEvaluationLoop if no_traditional_metrics else ModelTrainer
        trainer = TrainerClass(
            model=model,
            # tokenizer=task.tokenizer,  # Don't pass it if you don't want to save it and have other wacky shit extracted from it to influence training.

            # Args
            args=training_args,
            optimizers=(optimizer, scheduler),
            callbacks=callbacks,

            # Data
            train_dataset=datasetdict["train"],
            eval_dataset=[] if no_traditional_metrics else datasetdict["validation"],
            data_collator=collator,

            # Evaluation
            compute_metrics=task._computeMetrics,
            preprocess_logits_for_metrics=task.sneakyLogitTransform
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
        print("Context length:", pluralise(task._getMaxInputLength(), "token"))

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
            log(f"Training model {model.__class__.__name__} on task {task.__class__.__name__} on device {model.device}...")
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

            # Save results
            all_results = validation_results | test_results
            results_path = LamotoPaths.append(LamotoPaths.pathToEvaluations(), global_model_identifier) / f"metrics-{trainer.state.global_step}.json"
            log(f"Saving results to {results_path.as_posix()} ...")
            with open(results_path, "w", encoding="utf-8") as handle:
                json.dump(all_results, handle, indent=4)

            # Delete all other artifacts if requested.
            if hyperparameters.traceless:
                log("Deleting models...")
                trainer.deleteCheckpointsInOrder(amount=2)

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