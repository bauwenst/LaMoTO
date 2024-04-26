import time
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl


class EvaluateBeforeTrainingCallback(TrainerCallback):
    """
    Triggers evaluation before the first training batch, so that you can benchmark all metrics before any finetuning
    has been done (and then print it or let it be caught by another callback).
    https://discuss.huggingface.co/t/how-to-evaluate-before-first-training-step/18838/7

    It should be noted that although on_train_begin() exists, there is a bunch of stuff that comes after it. Hence,
    we use on_step_begin(), which is called after the dataloader's first batch is consumed but right before the model is
    referenced for the first time in the training loop.
    """
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step == 0:
            control.should_evaluate = True


class CheckpointAtTimeInterval(TrainerCallback):
    """
    Rather than saving based on the amount of STEPS trained, save best on the amount of TIME trained.
    """

    def __init__(self, minutes: float):
        self.seconds_between_saves = minutes*60
        self.last_save_was_at = 0

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.last_save_was_at = time.perf_counter()

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        seconds_since_last_save = time.perf_counter() - self.last_save_was_at
        if seconds_since_last_save >= self.seconds_between_saves:
            control.should_save = True

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.last_save_was_at = time.perf_counter()


class CheckpointLastModel(TrainerCallback):
    """
    In the context of checkpoints (which include model, Adam momenta, learning schedule state, ...), "last" is used to
    refer to the most recently stored version of the model, not to the most recent weights. Yet, it is these weights we want.
    In the Trainer training loop, the following order of calls happens.

    In the within-epoch loop:
        - on_step_end(): sets control.should_training_stop if global_step >= max_steps.
        - break if control.should_training_stop

    In the across-epoch loop:
        - on_epoch_end()
        - _maybe_log_save_evaluate(): stores a full checkpoint if control.should_save
        - break if control.should_training_stop

    In the teardown:
        - _load_best_model(): throws out the current weights and replaces them by the weights of the best checkpoint on disk
        - on_train_end()

    That means, if you want to save the very latest weights, you should set control.should_save in on_epoch_end() when control.should_training_stop == True.
    """

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if control.should_training_stop:  # Training is being exited as we speak.
            control.should_save = True
