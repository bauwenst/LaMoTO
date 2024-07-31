from enum import Enum

import time
from typing import Set, Union

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl


class EvaluateBeforeTrainingCallback(TrainerCallback):
    """
    Triggers evaluation before the first training batch, so that you can benchmark all metrics before any finetuning
    has been done (and then print it or let it be caught by another callback).
    https://discuss.huggingface.co/t/how-to-evaluate-before-first-training-step/18838/7

    It should be noted that although on_train_begin() exists, there is a bunch of stuff that comes after it. Hence,
    we use on_step_begin(), which is called after the dataloader's first batch is consumed but right before the model is
    referenced for the first time in the training loop.

    FIXME: This does still run 1 full gradient descent pass first... In CLM, it shows that half a million tokens (512 examples of 1024 tokens)
           are processed BEFORE entering the eval loop for the first time.
    """
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step == 0:
            control.should_evaluate = True


class EventType(Enum):
    EVALUATE   = 1
    CHECKPOINT = 2
    STOP       = 3


class CallbackAtTimeInterval(TrainerCallback):
    """
    Rather than saving/evaluating/stopping based on the amount of STEPS trained, do it based on the amount of TIME trained.
    """

    def __init__(self, minutes: float, events: Union[EventType, Set[EventType]]):
        self.seconds_between_events = minutes * 60
        self.last_event_was_at = 0
        self.event_types = events if isinstance(events, set) else {events}

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.last_event_was_at = time.perf_counter()

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        seconds_since_last_event = time.perf_counter() - self.last_event_was_at
        if seconds_since_last_event >= self.seconds_between_events:
            if EventType.CHECKPOINT in self.event_types:
                control.should_save = True
            if EventType.EVALUATE in self.event_types:
                control.should_evaluate = True
            if EventType.STOP in self.event_types:
                control.should_training_stop = True

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if EventType.CHECKPOINT in self.event_types:
            self.last_event_was_at = time.perf_counter()

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if EventType.EVALUATE in self.event_types:
            self.last_event_was_at = time.perf_counter()

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if EventType.STOP in self.event_types:
            self.last_event_was_at = time.perf_counter()


class CheckpointLastModel(TrainerCallback):
    """
    In the context of checkpoints (which include model, Adam momenta, learning schedule state, ...), "last" is used to
    refer to the most recently stored version of the model, not to the most recent weights. Yet, it is these weights we want.
    In the Trainer training loop, the following order of calls happens.

    In the within-epoch loop:
        - on_step_end(): sets control.should_training_stop if global_step >= max_steps.
        - break if control.should_training_stop  (the loop will also exit naturally at the end of the epoch)

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
