from typing import Optional, List
import torch
import shutil
from transformers.trainer import DataLoader, EvalLoopOutput, EvalPrediction, denumpify_detensorize, deepspeed_init, logger, has_length

from hf_mtask_trainer import HfMultiTaskTrainer


class LamotoTrainer(HfMultiTaskTrainer):
    """
    By using this as the parent class (a subclass of Trainer), a model is equipped with a self.report_metrics() method
    before training that is linked back to the Trainer. This allows it to collect extra metrics inside its modules. To
    make use of this, your architecture would contain something like
    ```
        if hasattr(self, "report_metrics"):
            self.report_metrics(key1=val1, key2=val2, ...)
    ```
    Also, it adds a method to delete checkpoints.
    """

    def deleteCheckpointsInOrder(self, amount: int):
        assert amount > 0
        checkpoint_paths = self._sorted_checkpoints(use_mtime=False, output_dir=self._get_output_dir(None))  # Note: puts best model at the end of the list if applicable.
        for checkpoint in checkpoint_paths[0:amount]:
            shutil.rmtree(checkpoint, ignore_errors=True)


class LamotoTrainerWithoutEvaluationLoop(LamotoTrainer):
    """
    Has a version of the evaluation loop where, rather than looping over a dataloader, we call compute_metrics on Nones.
    The reason for not altering evaluate() or get_eval_dataloader() is that we now still have the benefits of
    getting speed metrics and of logging.

    We basically cut out everything to do with logits/labels, while keeping the acceleration setup.
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
