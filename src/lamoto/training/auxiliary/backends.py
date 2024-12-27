"""
Backend classes that do the actual model training itself.
"""
from typing import Optional, List, Dict, Tuple, Callable, Union, Any
import torch
import shutil

from torch import Tensor
from torch.nn import Module
from transformers.data.data_collator import DataCollator
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer as Optimizer
from torch.utils.data import Dataset, IterableDataset
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import Trainer, DataLoader, EvalLoopOutput, denumpify_detensorize, deepspeed_init, logger, has_length
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import TrainingArguments

from archit.instantiation.mixins import LoggingState, ReportDiagnosticsMixin


class ModelTrainer(Trainer):
    """
    Adds two features on top of Trainer:
        1. Modules that extend ArchIt's ReportDiagnosticsMixin can call .report() to log any runtime value, e.g. to make it
           show up in WandB aside from metrics.
        2. Adds a method to delete checkpoints.
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
        extra_logs = self._extra_log.compute(tensor_gathering_function=self._nested_gather, round_digits=None)
        try:
            super().log(logs | extra_logs, start_time)
        except TypeError:  # The start_time argument is not yet supported in transformers v4.46.3, the last version with stable DeBERTa.
            super().log(logs | extra_logs)

    def deleteCheckpointsInOrder(self, amount: int):
        assert amount > 0
        checkpoint_paths = self._sorted_checkpoints(use_mtime=False, output_dir=self._get_output_dir(None))  # Note: puts best model at the end of the list if applicable.
        for checkpoint in checkpoint_paths[0:amount]:
            shutil.rmtree(checkpoint, ignore_errors=True)


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
