"""
Extensions of the HuggingFace Trainer, which implement the training loop behind LaMoTO's train() methods.
"""
from typing import Optional, List, Dict, Tuple, Callable, Union, Any
from pathlib import Path

import torch
import shutil
import warnings
import os

from torch import Tensor
from torch.nn import Module
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer as Optimizer
from torch.utils.data import Dataset, IterableDataset
from transformers.trainer import Trainer, PreTrainedModel, TrainingArguments, DataCollator, PreTrainedTokenizerBase, EvalPrediction, DataLoader, EvalLoopOutput, denumpify_detensorize, deepspeed_init, logger, has_length, TrainerCallback
from transformers.trainer_utils import speed_metrics

from archit.instantiation.mixins import LoggingState, ReportDiagnosticsMixin
from tktkt.util.dicts import dictToJson

BACKUPS_FOLDER = "backups"


class ModelTrainer(Trainer):
    """
    Adds three features on top of Trainer:
        1. Modules that extend ArchIt's ReportDiagnosticsMixin can call .report() to log any runtime value, e.g. to make it
           show up in WandB aside from metrics.
        2. Adds a method to delete checkpoints.
        3. Adds more callbacks.
    """
    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, Module]] = None,
        args: Optional[TrainingArguments] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset, Any]] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset], Any]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Optional[Tuple[Optimizer, LambdaLR]] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[Tensor, Tensor], Tensor]] = None
    ):
        try:  # super() interface with processing_class rather than tokenizer, which has existed since about transformers 4.46. https://github.com/huggingface/transformers/pull/32385.
            super().__init__(
                model=model,
                args=args,
                data_collator=data_collator,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                processing_class=tokenizer,
                model_init=model_init,
                compute_metrics=compute_metrics,
                callbacks=callbacks,
                optimizers=optimizers,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics
            )
        except TypeError:
            super().__init__(
                model=model,
                args=args,
                data_collator=data_collator,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                model_init=model_init,
                compute_metrics=compute_metrics,
                callbacks=callbacks,
                optimizers=optimizers,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics
            )

        # We store a reference to the tokeniser so that we can un-redirect HF's access to self.tokenizer.
        self._tokenizer = tokenizer

        # Callbacks
        from .callbacks import LamotoCallbackHandler, LamotoTrainerControl  # To avoid circular import.
        self.callback_handler = LamotoCallbackHandler.fromExisting(self.callback_handler)
        self.callback_handler.setTrainer(self)
        self.extra_control = LamotoTrainerControl()

        # ArchIt mixins
        self._extra_log = LoggingState(self.args)
        if model is not None:
            self._activateExtraLog(model)

    def _activateExtraLog(self, model: Module):
        # Define function that registers the above object in every relevant module.
        def f(module: Module):
            if isinstance(module, ReportDiagnosticsMixin):
                module.registerLog(self._extra_log)

        # Apply it recursively.
        model.apply(f)

    def log(self, logs: Dict[str, float], start_time: Optional[float]=None):
        # In transformers, the log() method figures out that you're training based on whether start_time is None... no.
        counts = {"train": 0, "eval": 0, "test": 0}
        for key in logs:
            for prefix in counts:
                if key.startswith(prefix + "_"):
                    counts[prefix] += 1
                    break
            else:  # Unprefixed metrics are counted as train metrics.
                counts["train"] += 1
        split = max(counts, key=counts.get)

        # Compute extra logs from registrations.
        extra_logs = self._extra_log.compute(tensor_gathering_function=self._nested_gather, round_digits=None)

        # Lastly, get extra diagnostic logs, namely for speed and total floating point operations.
        #   - In transformers v4.49.0, the last version before .from_pretrained() is broken, the speed metrics are actually calculated in the super() call below, but they forget to add the results to the logs... yeah.
        #   - The floating point operations are only counted during training, but they are also only outputted at the end of training. There's no good reason for that.
        profiling_logs = {f"{split}_device": torch.cuda.get_device_name()}
        if split == "train":
            profiling_logs |= {"train_flos": self.state.total_flos}
            if start_time is not None:
                profiling_logs |= speed_metrics(split, start_time, num_tokens=self.state.num_input_tokens_seen)
        if (split == "train" and start_time is None) or not any(key.endswith("_runtime") for key in logs):  # Then speed_metrics was run outside of self.log(). Bad design by HF that speed_metrics is run in two different places.
            warnings.warn("Speed metrics could not be added to the logs.")

        # Now call into the super method to add epoch and num_input_tokens_seen.
        try:
            super().log(extra_logs | profiling_logs | logs, start_time)
        except TypeError:  # The start_time argument is not yet supported in transformers v4.46.3, the last version with stable DeBERTa.
            super().log(extra_logs | profiling_logs | logs)

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch: int, ignore_keys_for_eval, start_time: Optional[float]=None):
        # maybe_log_save_evaluate (which should actually be called maybe_log_evaluate_save) is run AFTER on_step_end,
        # which means that anything that requires the latest logs, latest evals or latest checkpoint, needs to wait until then.
        try:
            super()._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time)
        except TypeError:
            super()._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)

        if self.extra_control.should_backup:
            self.save_backup()
            self.control = self.callback_handler.on_backup(self.args, self.state, self.control)

    def save_backup(self):
        output_dir = Path(self.args.output_dir) / BACKUPS_FOLDER / f"{self.state.global_step}"  # The reason we don't use the 'checkpoint-' prefix is that I suspect that HuggingFace's rotation system hunts for it with .glob(). Not sure if it searches recursively, but better safe than sorry.
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save the latest train/eval log at that point, which contains the GPU time and FLOPs.
        logs = self.state.log_history
        for split in ["train", "eval"]:
            for log in reversed(logs):
                if any(key.startswith(split + "_") for key in log):  # Note: this assumes that there is no interference between "train" and "eval", whereas any(eval) should never count as train.
                    dictToJson(log, output_dir / f"latest_log_{split}.json")
                    break

        # Save model as the last thing you do, because that operation is most likely to error.
        self.save_model(output_dir.as_posix())

    def deleteCheckpointsInOrder(self, amount: int):
        assert amount > 0
        checkpoint_paths = self._sorted_checkpoints(use_mtime=False, output_dir=self._get_output_dir(None))  # Note: puts best model at the end of the list if applicable.
        for checkpoint in checkpoint_paths[0:amount]:
            shutil.rmtree(checkpoint, ignore_errors=True)

    def tryDeleteFolder(self, unless_contains_subfolders: List[str]=None):
        """
        Deletes the entire model folder filled by the current trainer instance. If any of the given folders are
        present, the folder is preserved along with those subfolders.
        """
        if unless_contains_subfolders is None:
            unless_contains_subfolders = []

        main_folder_path = Path(self._get_output_dir(None))
        _, folder_names, _ = next(os.walk(main_folder_path))
        delete_everything = True
        for f in folder_names:
            if f in unless_contains_subfolders:
                delete_everything = False
            else:
                shutil.rmtree(main_folder_path / f)

        if delete_everything:
            shutil.rmtree(main_folder_path)

    @property
    def tokenizer(self) -> Optional[PreTrainedTokenizerBase]:
        return self._tokenizer


class ModelTrainerWithoutEvaluationLoop(ModelTrainer):
    """
    Has a version of the evaluation loop where, rather than looping over a dataloader, we call compute_metrics on Nones.
    The reason for not altering evaluate() or get_eval_dataloader() is that we now still have the benefits of
    getting speed metrics and of logging.

    We basically cut out everything to do with logits/labels, while keeping the acceleration setup.

    TODO: Needs to be updated to a more recent version of transformers.
    """

    def evaluation_loop(self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        # Copy-pasted from Trainer.evaluation_loop (transformers 4.39.3)
        ############################################################################################################
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            if self.is_fsdp_enabled:
                self.model = model
            if model is not self.model:
                self.model_wrapped = model
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        if not self.is_in_train:
            if self.args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=self.args.device)
            elif self.args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=self.args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        ############################################################################################################

        metrics = self.compute_metrics(EvalPrediction(predictions=None, label_ids=None))

        # Copy-pasted from Trainer.evaluation_loop
        ############################################################################################################

        metrics = denumpify_detensorize(metrics)
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=0)
