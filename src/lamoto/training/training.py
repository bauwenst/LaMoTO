from typing import Dict, Tuple, Any, List
from pathlib import Path

import time
import torch
import wandb
import transformers
from transformers import PreTrainedModel, TrainingArguments, IntervalStrategy, EarlyStoppingCallback, AutoTokenizer, \
    PreTrainedTokenizerBase, EvalPrediction, PretrainedConfig
from transformers.trainer_utils import has_length
from transformers.utils.logging import set_verbosity_error
from huggingface_hub.constants import HF_HUB_CACHE

from archit.instantiation.abstracts import CombinedConfig
from archit.util import torchPrint, parameterCountBaseVsHead
from fiject.applications.transformers import FijectCallback
from tktkt.interfaces.huggingface import TktktToHuggingFace
from tktkt.interfaces.tokenisers import TokeniserWithVocabulary
from tktkt.interfaces.factories import TokeniserFactory
from tktkt.util.printing import intsep, pluralise, dprint
from tktkt.util.timing import datetimeDashed, Timer
from tktkt.util.dicts import dictToJson
from tktkt.paths import PathManager

from ..tasks._core import Task, RankingMetricSpec
from ..augmenting.augment_model import ModelAugmentation
from ..measuring import METRICS
from ..measuring._core import LamotoMetric, EvaluationEnvironment
from ..util.datasets import shuffleAndTruncate, getDatasetSize, totalBatches
from ..util.exceptions import tryExceptNone, ImpossibleBranchError
from ..util.strings import getSubstringAfterLastSlash
from ..util.visuals import log, printLamotoWelcome
from .auxiliary.callbacks import CallbackAtTimeInterval, SaveTokeniserWithCheckpoints, CheckpointLastModel, EventType, \
    SaveModelOnLinearInterval, SaveModelOnTimeInterval, _SaveModelMixin, TrainerCallback
from .auxiliary.hyperparameters import *
from .auxiliary.backends import ModelTrainer, ModelTrainerWithoutEvaluationLoop

LamotoPaths = PathManager("lamoto")


DO_WARNINGS_AND_PROGRESSBARS = True
def showWarningsAndProgress(enabled: bool):
    global DO_WARNINGS_AND_PROGRESSBARS
    DO_WARNINGS_AND_PROGRESSBARS = enabled


class TaskTrainer:

    def __init__(self, model_augmentation: ModelAugmentation=None, custom_callbacks: List[TrainerCallback]=None):
        self._model_augmentation = model_augmentation
        self._extra_callbacks = custom_callbacks or []

    def train(
        self,
        task: Task,
        hyperparameters: TaskHyperparameters[HC]=getDefaultHyperparameters(),
        resume_from_folder: Path=None
    ) -> Tuple[str, Dict[str, float]]:
        """
        Encapsulation of everything you need to do to get a (modified) `transformers.Trainer` running.
        """
        printLamotoWelcome()
        timer = Timer()
        timer.start(echo=True)
        log("Running task:", task.task_name)
        transformers.set_seed(seed=hyperparameters.seed)
        if not DO_WARNINGS_AND_PROGRESSBARS:
            set_verbosity_error()
        task.resetTemporaryFields()

        # Imputations and sanity checks
        if isinstance(hyperparameters.model_config_or_checkpoint, Path):
            hyperparameters.model_config_or_checkpoint = hyperparameters.model_config_or_checkpoint.as_posix()  # FIXME: Possibly have to make it a relative path due to HF restrictions.
        if task.metric_config.to_rank is None or hyperparameters.rank_checkpoints_using_loss:
            metric_to_rank = RankingMetricSpec(metric_name="", result_name="loss", higher_is_better=False)
        else:
            metric_to_rank = task.metric_config.to_rank

        if hyperparameters.init_weights and not isinstance(hyperparameters.model_config_or_checkpoint, str):
            raise ValueError("You said you wanted to initialise model weights from the checkpoint, but didn't give a checkpoint path!")
        if hyperparameters.archit_basemodel_class is None:
            raise ValueError("In order to parse model configs, the archit_basemodel_class hyperparameter cannot be None.")
        if hyperparameters.archit_head_config is None and not isinstance(hyperparameters.model_config_or_checkpoint, str):  # Note: there is another failure case: when the checkpoint *is* a string, but *isn't* an ArchIt checkpoint. It errors below.
            raise ValueError("Without a checkpoint, a head config must be provided to instantiate a new head.")
        if not hyperparameters.track_best_checkpoint and hyperparameters.rank_checkpoints_using_loss:
            raise ValueError("Asked to rank models with loss, but also to not track the best model.")
        if hyperparameters.track_best_checkpoint and metric_to_rank.fullName() == "loss" and not hyperparameters.rank_checkpoints_using_loss:
            raise ValueError("Asked to NOT rank models with loss, but no ranking metric is configured for this task meaning models would be ranked by loss.")
        if hyperparameters.track_best_checkpoint and metric_to_rank.fullName() != "loss" and metric_to_rank.metric_name not in task.metric_config.to_compute:
            raise ValueError(f"Cannot rank models based on metric {metric_to_rank.metric_name} since it isn't computed.")
        for metric_name in task.metric_config.to_track.keys():
            if metric_name not in task.metric_config.to_compute:
                raise ValueError(f"Requested tracking results for metrics {sorted(task.metric_config.to_track)} yet you are only computing metrics {sorted(task.metric_config.to_compute)}.")

        # Imputation of device batch size specifically
        n_devices = torch.cuda.device_count()
        if hyperparameters.examples_per_effective_batch % n_devices != 0:  # This is an unsolvable issue by setting a new device batch size.
            raise ValueError(f"Effective batch size ({hyperparameters.examples_per_effective_batch}) must be a multiple of the amount of devices ({n_devices}).")

        n_examples_per_pass_per_device = hyperparameters.examples_per_device_batch
        n_examples_per_pass            = n_examples_per_pass_per_device * n_devices
        if n_examples_per_pass > hyperparameters.examples_per_effective_batch:  # One pass through your devices already exceeds one effective batch.
            n_examples_per_pass_per_device = hyperparameters.examples_per_effective_batch // n_devices  # Suggest a new device batch size.
            n_examples_per_pass            = hyperparameters.examples_per_effective_batch
        if hyperparameters.examples_per_effective_batch % n_examples_per_pass != 0:  # You can't push an integer amount of passes through your devices to get to the effective batch size.
            # Example: effective batch size 96, 4 devices, device batch size 7  =>  28 per pass whilst you need a total of 24.
            # Because we get to choose the device batch size but not the effective batch size or amount of device, we choose a new one as follows:
            #   - The base criterion is that effective batch size % (device size * devices) == 0, thus effective batch size == device size * devices * integer.
            #   - It has to be smaller than whatever device batch size we have currently.
            #   - It has to be a power of 2.
            #   - It has to be as high as possible given the above.
            # In the example: we want to get to 96//4 == 24 examples with each device, so push 24 examples for 1 pass, 12 for 2 passes, 8 for 3 passes, 6 for 4 passes, 4 for 6 passes, 3 for 8 passes, 2 for 12 passes or 1 for 24 passes.
            # The best of these is size 4 for 6 passes because it is smaller than the given 7 whilst being the highest possible power of 2.
            n_examples_per_device = hyperparameters.examples_per_effective_batch // n_devices
            for n_passes in range(1, n_examples_per_device+1):
                if n_examples_per_device % n_passes == 0:
                    n_examples_per_pass_per_device = n_examples_per_device // n_passes
                    if n_examples_per_pass_per_device & (n_examples_per_pass_per_device-1) == 0:  # The only numbers for which -1 flips all the bits are powers of 2.
                        break
            else:
                raise ImpossibleBranchError()
            n_examples_per_pass = n_devices * n_examples_per_pass_per_device
        n_passes = hyperparameters.examples_per_effective_batch // n_examples_per_pass

        # Metadata
        try:
            task.adjustHyperparameters(hyperparameters)
        except:
            log("Failed to adjust hyperparameters; an error is on its way.")
        task._setHyperparameters(hyperparameters)

        log("Loading model config...")
        config_or_str = hyperparameters.model_config_or_checkpoint
        if isinstance(config_or_str, str):  # It's a checkpoint string. Can either be a checkpoint for the ModelWithHead we're about to load, or for anything else compatible. We'll figure that out.
            model_config = CombinedConfig.from_pretrained(config_or_str,
                                                          head_config=hyperparameters.archit_head_config,
                                                          base_model_config_class=hyperparameters.archit_basemodel_class.config_class)  # Note that there is no need for AutoConfig because we KNOW the config class (even if not registered in AutoConfig). Also means we don't have to store the "model type" in the config.
        else:  # It's an object.
            assert isinstance(config_or_str, PretrainedConfig)
            if isinstance(config_or_str, CombinedConfig):
                raise ValueError("When instantiating a new model from a config, it must only parameterise the base model. The head has its own config.")
            model_config = CombinedConfig(base_model_config=config_or_str,
                                          head_config=hyperparameters.archit_head_config,
                                          base_model_config_class=hyperparameters.archit_basemodel_class.config_class)  # This call pretends to be CombinedConfig(**json).
        task._setModelConfig(model_config)

        model_identifier, global_model_identifier = self._getRunIdentifiers(task, hyperparameters)
        folder_of_this_models_checkpoints = self._getCheckpointPath(global_model_identifier, hyperparameters)

        # Set up tokeniser
        log("Loading tokeniser...")
        tokenizer = hyperparameters.tokeniser
        if tokenizer:
            if isinstance(tokenizer, str):
                tokenizer = AutoTokenizer.from_pretrained(tokenizer, add_prefix_space=True)
            elif isinstance(tokenizer, TokeniserWithVocabulary):
                tokenizer = TktktToHuggingFace(tokenizer)
            elif isinstance(tokenizer, TokeniserFactory):
                tokenizer = TktktToHuggingFace(tokenizer.buildTokeniser())
            elif not isinstance(tokenizer, PreTrainedTokenizerBase):
                raise RuntimeError(f"Cannot handle tokeniser of type '{type(hyperparameters.tokeniser)}'.")
        elif isinstance(hyperparameters.model_config_or_checkpoint, str):
            tokenizer = AutoTokenizer.from_pretrained(hyperparameters.model_config_or_checkpoint, add_prefix_space=True)
        else:
            raise RuntimeError("Cannot deduce tokeniser checkpoint from a model config.")
        task._setTokenizer(tokenizer)

        # - Old models like GPT-2 have no pad token, but this doesn't really matter because it's actually the attention
        #   mask that determines if a token is processed, so you can replace it by any token you want. https://github.com/stanford-crfm/BioMedLM/issues/4
        if task.tokenizer.pad_token is None:
            task.tokenizer.pad_token                         = task.tokenizer.eos_token
            task.model_config.base_model_config.pad_token_id = task.tokenizer.eos_token_id
        assert task.tokenizer.pad_token is not None

        # - When the model is instantiated from scratch and the config disagrees with the tokeniser vocabulary about the
        #   pad token, the tokeniser gets precedent and overwrites the config.
        if not hyperparameters.init_weights and task.tokenizer.pad_token_id != task.model_config.base_model_config.pad_token_id:
            print(f"Note: Found mismatch between tokeniser pad token ({task.tokenizer.pad_token_id}) and config pad token ({task.model_config.base_model_config.pad_token_id}). The latter will be overwritten.")
            task.model_config.base_model_config.pad_token_id = task.tokenizer.pad_token_id

        # Now that you have the tokeniser, tokenise the dataset.
        log("Loading dataset...")
        datasetdict = task.loadDataset()
        n_examples_validation = tryExceptNone(lambda: getDatasetSize(datasetdict["validation"], split="validation")) or 1_000_000_000_000  # Very very big number assumed when you can't find the dataset size.
        n_examples_validation = n_examples_validation if not hyperparameters.examples_per_evaluation else min(n_examples_validation, hyperparameters.examples_per_evaluation)
        print(datasetdict)

        log("Preparing dataset...")
        datasetdict["train"]      = shuffleAndTruncate(datasetdict["train"], seed=hyperparameters.seed)
        datasetdict["validation"] = shuffleAndTruncate(datasetdict["validation"], seed=hyperparameters.seed, truncate_to=n_examples_validation)
        datasetdict = task.prepareDataset(datasetdict)

        # Get the batch generator, a.k.a. collator (https://huggingface.co/docs/transformers/main_classes/data_collator).
        collator = task.getCollator()

        # Get the model...  FIXME: Somehow, the DP head uses an operation that doesn't exist for bfloat16. Temporary fix below.
        do_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)  # including_emulation keyword arg needed due to https://github.com/pytorch/pytorch/issues/124996
        task.automodel_args["torch_dtype"] = torch.bfloat16 if do_bf16 and not task.task_name.lower().startswith("dp") else torch.float32

        hf_checkpoint_classname = task.model_config.architectures[0] if task.model_config.architectures is not None else ""  # Always present and correct for HuggingFace configs.
        is_exact_hf_checkpoint    = hyperparameters.init_weights and hyperparameters.load_hf_automodel_if_hf_checkpoint_and_matches_task and task._isHfCheckpointForThisTask(hf_checkpoint_classname)
        is_custom_hf_architecture = hyperparameters.custom_hf_class is not None
        if not is_exact_hf_checkpoint and not is_custom_hf_architecture:  # Use ArchIt. This is the usual case.
            log("Instantiating an ArchIt model.")
            torch.set_default_dtype(task.automodel_args["torch_dtype"])
            if hyperparameters.init_weights:
                model: PreTrainedModel = task.archit_class.from_pretrained(hyperparameters.model_config_or_checkpoint, hyperparameters.archit_basemodel_class, hyperparameters.archit_head_config)
            else:
                assert hyperparameters.archit_head_config is not None, "You forgot to set the head config in the hyperparameters!"
                model: PreTrainedModel = task.archit_class.fromModelAndHeadConfig(hyperparameters.archit_basemodel_class.from_config(task.model_config), hyperparameters.archit_head_config)
        else:  # Edge cases.
            if is_custom_hf_architecture:
                log("Instantiating a custom HuggingFace class.")
                if hyperparameters.init_weights:  # model_config_or_checkpoint is a string
                    model: PreTrainedModel = hyperparameters.custom_hf_class.from_pretrained(hyperparameters.model_config_or_checkpoint, **task.automodel_args)
                else:
                    model: PreTrainedModel = hyperparameters.custom_hf_class._from_config(task.model_config.base_model_config, **task.automodel_args)
            elif is_exact_hf_checkpoint:  # model_config_or_checkpoint is a string
                log(f"The given checkpoint seems to be a HuggingFace architecture ({hf_checkpoint_classname}) for this specific task ({task.archit_class.__name__}),\nwe will instantiate the model with AutoModel ({task.automodel_class.__name__}) instead of ArchIt.")
                model: PreTrainedModel = task.automodel_class.from_pretrained(hyperparameters.model_config_or_checkpoint, **task.automodel_args)
            else:
                raise ImpossibleBranchError()

        # ...and augment it in-place (possibly with the tokeniser). We assume the augmentation uses .base_model when it needs to.
        if self._model_augmentation:
            if hyperparameters.init_weights:
                self._model_augmentation.augmentAndLoad(model, task.tokenizer, checkpoint=hyperparameters.model_config_or_checkpoint)
            else:
                self._model_augmentation.augment(model, task.tokenizer)
        model.to("cuda")

        # ...and verify that after all of this, the model, the config and the tokeniser all agree about the pad token.
        pad_tk = task.tokenizer.pad_token_id
        pad_model = model.config.pad_token_id
        pad_config = task.model_config.base_model_config.pad_token_id
        print("Verifying that padding indices match:")
        print("\t    Model:", pad_model)
        print("\t   Config:", pad_config)
        print("\tTokeniser:", pad_tk)
        print()
        if pad_model != pad_tk:
            raise RuntimeError("Mismatch between padding token in the model and the tokeniser!")
        if pad_model != pad_config:
            raise RuntimeError("Mismatch between padding token in the model and in the loaded model config!")
        if pad_config != pad_tk:
            raise RuntimeError("Mismatch between padding token in the loaded model config and the tokeniser!")

        # Now that we have a reference to the dataset and model, build the metrics.
        # (Some need access to the ModelTrainer for collation, but since we don't know which trainer we'll instantiate
        #  before knowing the properties of the metrics, the trainer is instantiated and linked to the env afterwards.)
        env = EvaluationEnvironment(
            model=model,
            tokeniser=task.tokenizer,
            validation_dataset=datasetdict["validation"],
            test_dataset=tryExceptNone(lambda: datasetdict["test"]),  # MLM doesn't have this, for example.
            hyperparameters=task.hyperparameters,
            trainer=None
        )
        task._setMetrics({name: METRICS.load(name,env) for name in task.metric_config.to_compute})

        # Set up reporting too
        folder_wandb = folder_of_this_models_checkpoints / "wandb"
        folder_wandb.mkdir(exist_ok=True)
        wandb.init(
            mode="disabled" if hyperparameters.discard_artifacts or not hyperparameters.wandb_project else "online",

            project=hyperparameters.wandb_project,
            group=model_identifier,
            name=global_model_identifier,
            tags=[task.task_name, torch.cuda.get_device_name()] + ([self._model_augmentation.name] if self._model_augmentation else []),

            dir=folder_wandb.as_posix()
        )

        # Training arguments
        # - Sizes
        stopping_condition = hyperparameters.hard_stopping_condition
        n_gradient_descents = tryExceptNone(lambda: stopping_condition.getSteps(batch_size=hyperparameters.examples_per_effective_batch,
                                                                                dataset=datasetdict["train"], split_name="train"))
        wu = hyperparameters.effective_batches_warmup  # Alias to shorten this long name.
        if isinstance(wu, int):
            if wu < 0:
                raise ValueError(f"The amount of warmup batches ({wu}) has to be a positive integer or a float in [0,1].")
            n_descents_of_warmup = wu
        else:  # Fractional warmup in [0,1]
            if wu < 0 or wu > 1:
                raise ValueError(f"The amount of warmup batches ({wu}) has to be a positive integer or a float in [0,1].")
            if n_gradient_descents is None:
                raise ValueError(f"Amount of warmup batches ({wu}) was given as a fraction of the total amount of training batches, but we don't know what that is for stopping condition {hyperparameters.hard_stopping_condition.__class__.__name__}")
            n_descents_of_warmup = int(n_gradient_descents*wu)

        # - Intervals
        eval_interval = hyperparameters.eval_vs_save_intervals.evaluation or Never()
        if not hyperparameters.track_best_checkpoint:
            save_interval = hyperparameters.eval_vs_save_intervals.checkpointing or Never()
        else:  # Ignore it and sync with eval interval.
            if isinstance(eval_interval, Never):
                raise ValueError("You indicated that you want to track the best model, but specified no evaluation interval!")
            save_interval = eval_interval
        backup_interval = hyperparameters.eval_vs_save_intervals.backups or Never()  # Not relevant to the TrainingArguments, but will come in later.

        batches_between_evals = tryExceptNone(lambda: eval_interval.getSteps(batch_size=hyperparameters.examples_per_effective_batch,
                                                                             dataset=datasetdict["train"], split_name="train"))
        batches_between_saves = tryExceptNone(lambda: save_interval.getSteps(batch_size=hyperparameters.examples_per_effective_batch,
                                                                             dataset=datasetdict["train"], split_name="train"))
        batches_between_backups = tryExceptNone(lambda: backup_interval.getSteps(batch_size=hyperparameters.examples_per_effective_batch,
                                                                                 dataset=datasetdict["train"], split_name="train"))

        # - Early stopping (only used if required)
        best_model_metric_handle = f"eval_{metric_to_rank.fullName()}" if hyperparameters.track_best_checkpoint else None

        # - Finally get args
        training_args = TrainingArguments(
            max_steps=(n_gradient_descents or -1) if n_gradient_descents is not None or has_length(datasetdict["train"]) else 1_000_000_000_000,  # Normally, -1 is all you need to indicate "no limit". However, HF has a check that you cannot ask for "no limit" when the dataset's actual limit is unknown (which is counter-intuitive...). The check exists for learning rate schedules that decrease relative to the max amount of descents, but we don't use those schedules.
            num_train_epochs=1_000_000_000_000 if n_gradient_descents != 0 else 0,  # This value is used when max_steps is -1 (its default value is 3 but clearly it should be "run forever").

            # Optimisation (adding all of this in the TrainingArguments because apparently Trainer knows how to use HuggingFace `accelerate` whereas I only know the old optimisers)
            # optim=OptimizerNames.ADAMW_TORCH,
            # learning_rate=hyperparameters.LEARNING_RATE,
            # weight_decay=hyperparameters.L2_REGULARISATION,

            # lr_scheduler_type=SchedulerType.CONSTANT_WITH_WARMUP,
            # warmup_steps=n_descents_of_warmup,
            max_grad_norm=hyperparameters.gradient_clipping_norm,

            # Batches
            per_device_train_batch_size=n_examples_per_pass_per_device,
            gradient_accumulation_steps=n_passes,

            # Style of computations
            gradient_checkpointing=model.supports_gradient_checkpointing,  # Only if you have the VRAM though. Good explanation with animations: https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9
            bf16=do_bf16,

            # Evaluation
            eval_on_start=not isinstance(eval_interval, Never),  # Always do an evaluation at the start, unless you wanted to avoid all evaluations.
            eval_strategy=IntervalStrategy.STEPS if batches_between_evals else IntervalStrategy.NO,
            eval_steps=batches_between_evals,
            per_device_eval_batch_size=hyperparameters.examples_per_device_batch,  # We know that the GPU can handle at least this much data during eval if it can during training, since training additionally requires the gradients and optimiser to be stored in VRAM as overhead.
            eval_accumulation_steps=1,  # "Number of predictions steps to accumulate the output tensors for, before moving the results to the CPU. If left unset, all predictions are accumulated on GPU before being moved to the CPU (faster but requires more GPU memory)." You always need more RAM than VRAM, of course.

            # Saving
            save_strategy=IntervalStrategy.STEPS if batches_between_saves else IntervalStrategy.NO,
            save_steps=batches_between_saves,

            output_dir=folder_of_this_models_checkpoints.as_posix(),

            load_best_model_at_end=hyperparameters.track_best_checkpoint,  # Will take the best model out of its checkpoint directory and load it into task.model, which can then be saved. At the end of Trainer's loop, the following happens: "Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint"
            metric_for_best_model=best_model_metric_handle,
            greater_is_better=metric_to_rank.higher_is_better,
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
        callbacks: List[TrainerCallback] = [CheckpointLastModel(), SaveTokeniserWithCheckpoints(task.tokenizer)]
        if hyperparameters.track_best_checkpoint and hyperparameters.evals_of_patience is not None:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=hyperparameters.evals_of_patience))  # Patience is the amount of eval calls you can tolerate worsening loss.

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

        if backup_interval is not None:  # => There is a backup strategy.
            if batches_between_backups is not None:  # => It can even be expressed using linear steps.
                callbacks.append(SaveModelOnLinearInterval(start=batches_between_backups, step=batches_between_backups))
            elif isinstance(backup_interval, EveryNMinutes):
                callbacks.append(SaveModelOnTimeInterval(minutes=backup_interval.minutes))
            elif not isinstance(backup_interval, Never):
                raise ValueError(f"Cannot handle backup interval: {backup_interval.__class__.__name__}")

        # Note that training is always done by computing a loss, so it's always implemented to compute evaluation loss.
        # There is only one situation in which you don't do any evaluation at all, namely when you don't track the best
        # model and you have no evaluation schedule.
        # Of the other situations, there are some where you will use the eval loss and some where you won't.
        at_least_one_traditional_metric = any(
            not isinstance(m, LamotoMetric) or (isinstance(m, LamotoMetric) and not m.isAutonomous())
            for m in task.metrics.values()
        )

        compute_eval_loss = not isinstance(eval_interval, Never) and (
            at_least_one_traditional_metric or (
                hyperparameters.track_best_checkpoint and best_model_metric_handle == "eval_loss"  # eval loss doesn't show up as a 'traditional metric' because it isn't one; other models for tracking do.
            ) or (
                    not hyperparameters.track_best_checkpoint and not task.metric_config.to_track  # => You asked to evaluate on an interval, yet you didn't ask for any metrics to evaluate, so it must be loss.
            )
        )

        if not hyperparameters.discard_artifacts and not hyperparameters.wandb_project:
            if hyperparameters.track_best_checkpoint:
                callbacks.append(FijectCallback(global_model_identifier + "_eval_goal", evals_between_commits=4))  # Automatically tracks the same metric as is used to decide best model.

            metrics_to_track = task.metric_config.to_track
            if not hyperparameters.track_best_checkpoint and compute_eval_loss:  # => eval loss will be included in the results even though it isn't captured by the above callback.
                metrics_to_track = metrics_to_track | {"eval": {"loss": "loss"}}

            if metrics_to_track:
                callbacks.append(
                    FijectCallback(global_model_identifier + "_eval_tracked",
                                   evals_between_commits=4,
                                   metric_names_with_formatting={(metric_name + "_" + result_name): formatting
                                                                 for metric_name, result_formats in task.metric_config.to_track.items()
                                                                 for result_name, formatting in result_formats.items()})
                )

        # At last, the Trainer object.
        TrainerClass = ModelTrainer if compute_eval_loss else ModelTrainerWithoutEvaluationLoop
        trainer = TrainerClass(
            model=model,
            # tokenizer=task.tokenizer,  # Don't pass it if you don't want to save it and have other wacky shit extracted from it to influence training.

            # Args
            args=training_args,
            optimizers=(optimizer, scheduler),
            callbacks=callbacks + self._extra_callbacks,

            # Data
            train_dataset=datasetdict["train"],
            eval_dataset=datasetdict["validation"] if compute_eval_loss else [],
            data_collator=collator,

            # Evaluation
            compute_metrics=task._computeMetrics,
            preprocess_logits_for_metrics=task.sneakyLogitTransform
        )

        # Bidirectional associations with the trainer
        env.trainer = trainer
        for cb in callbacks:
            if isinstance(cb, _SaveModelMixin):
                cb.setTrainer(trainer)

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
        batch_size = hyperparameters.examples_per_effective_batch
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
                if batches_per_epoch > batches_between_evals:
                    print("\t", pluralise(round(batches_per_epoch / batches_between_evals, 2), "eval"), "per training epoch")
                else:
                    print("\t", pluralise(round(batches_between_evals / batches_per_epoch, 2), "training epoch"), "per eval")
        else:
            print("\t", "No sizes known.")
        print("="*50)

        # Train, and evaluate afterwards.
        try:
            log(f"Training starting with:\n\tModel: {model.__class__.__name__}\n\tTask: {task.task_name}\n\tRanking metric: {best_model_metric_handle if hyperparameters.track_best_checkpoint else 'None'}\n\tDevice: {model.device}...")
            trainer.train(resume_from_checkpoint=resume_from_folder.as_posix() if resume_from_folder else None)
            # trainer.save_model()  # Not needed since we already checkpoint the last model with a callback.
            # trainer.push_to_hub()
            global_step = trainer.state.global_step
            all_results = {"train_global_step": global_step}

            log("Evaluation of " + ("best" if hyperparameters.track_best_checkpoint else "last") + " model on validation set...")
            env.use_test_not_validation = False
            validation_results = trainer.evaluate(datasetdict["validation"], metric_key_prefix="eval") if compute_eval_loss else self._prefixMetrics(task._computeMetrics(EvalPrediction(predictions=[], label_ids=[])), metric_key_prefix="eval")
            dprint(validation_results, indent=1)

            if "test" in datasetdict:
                log("Evaluation of " + ("best" if hyperparameters.track_best_checkpoint else "last") + " model on test set...")
                env.use_test_not_validation = True
                test_results = trainer.evaluate(datasetdict["test"], metric_key_prefix="test") if compute_eval_loss else self._prefixMetrics(task._computeMetrics(EvalPrediction(predictions=[], label_ids=[])), metric_key_prefix="test")
                all_results |= validation_results | test_results
                dprint(test_results, indent=1)
            else:
                all_results |= validation_results
            wandb.finish()  # Finish because otherwise, running .train() in the same process after .init() has been called once already will raise an error.

            # Save results
            if not hyperparameters.discard_results:
                results_path = self._getEvalPath(global_model_identifier) / f"metrics-{global_step}.json"
                log(f"Saving results to {results_path.as_posix()} ...")
                dictToJson(all_results, results_path, do_indent=True)

            # Delete all other artifacts if requested. (We have at most two checkpoints. Backups are not checkpoints, and are never deleted.)
            if hyperparameters.discard_artifacts:
                log("Deleting models...")
                trainer.deleteCheckpointsInOrder(amount=2)
                trainer.tryDeleteFolder(unless_contains_subfolders=[_SaveModelMixin.BACKUPS_FOLDER])

            timer.soFar(echo=True)
            return global_model_identifier, all_results

        except Exception as e1:  # Catches any error that happens during training, and triggers a checkpoint (+ a callback event afterwards, if that's needed by any callback).
            log("Caught exception while training. A checkpoint will be saved.\nAfterwards, we will raise the exception, so your run shows up as failed rather than completed.")
            timer.soFar(echo=True)
            try:
                trainer._save_checkpoint(model, trial=None)
                trainer.callback_handler.on_save(trainer.args, trainer.state, trainer.control)
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

    def _prefixMetrics(self, metrics: Dict[str,Any], metric_key_prefix: str) -> Dict[str,Any]:
        metrics = metrics.copy()
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
        return metrics

    def _getRunIdentifiers(self, task: Task, hp: TaskHyperparameters) -> Tuple[str,str]:
        if hp.save_as:
            model_name = hp.save_as
        elif isinstance(hp.model_config_or_checkpoint, str):
            model_name = getSubstringAfterLastSlash(hp.model_config_or_checkpoint)
        else:  # We don't use the tokeniser name because it isn't directly related to the model.
            raise RuntimeError("Configs do not allow deducing the name under which to save the model.")

        return (
            model_name,
            model_name + ("" if not self._model_augmentation else ("+" + self._model_augmentation.name)) \
                       + f"_{task.task_name}" \
                       + f"_{datetimeDashed()}"
        )

    def _getCheckpointPath(self, global_identifier: str, hp: TaskHyperparameters) -> Path:
        if hp.store_in_hf_cache:
            return LamotoPaths.append(Path(HF_HUB_CACHE), global_identifier)  # If that first path doesn't exist yet, it will be created automatically.
        else:
            return LamotoPaths.append(LamotoPaths.pathToCheckpoints(), global_identifier)

    def _getEvalPath(self, global_identifier: str, identifier_suffix: str=""):
        return LamotoPaths.append(LamotoPaths.pathToEvaluations(), global_identifier + identifier_suffix)
