from typing import Set, Union
from abc import abstractmethod, ABC
from pathlib import Path
from enum import Enum

import time
import warnings

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl, PreTrainedTokenizerBase
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


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


class CombinedCallback(TrainerCallback, ABC):
    """
    Combined callback for evaluating, checkpointing and stopping.
    """

    def __init__(self, events: Union[EventType, Set[EventType]]):
        self.event_types = events if isinstance(events, set) else {events}

    @abstractmethod
    def on_event_happens(self):
        pass

    @abstractmethod
    def should_event_happen(self, global_step: int) -> bool:
        pass

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.on_event_happens()

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.should_event_happen(state.global_step):
            if EventType.CHECKPOINT in self.event_types:
                control.should_save = True
            if EventType.EVALUATE in self.event_types:
                control.should_evaluate = True  # An evaluation will be started, and when it finishes, the timer is reset.
            if EventType.STOP in self.event_types:
                control.should_training_stop = True

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if EventType.CHECKPOINT in self.event_types:
            self.on_event_happens()

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if EventType.EVALUATE in self.event_types:
            self.on_event_happens()

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if EventType.STOP in self.event_types:
            self.on_event_happens()


class CallbackAtTimeInterval(CombinedCallback):
    """
    Rather than saving/evaluating/stopping based on the amount of STEPS trained, do it based on the amount of TIME trained.
    """

    def __init__(self, minutes: float, events: Union[EventType, Set[EventType]]):
        super().__init__(events)
        self.seconds_between_events = minutes * 60
        self.last_event_was_at = 0

    def on_event_happens(self):
        self.last_event_was_at = time.perf_counter()

    def should_event_happen(self, global_step: int) -> bool:
        seconds_since_last_event = time.perf_counter() - self.last_event_was_at
        return seconds_since_last_event >= self.seconds_between_events


class CallbackAtExpInterval(CombinedCallback):
    """
    Produces triggers that are linearly spaced on a log axis. This is equivalent to using linearly spaced values as
    exponents for an exponential function. Examples:
        Start 1, spacing 1:     1, 10^1, 10^2, 10^3, ...
        Start 10, spacing 0.1: 10, 10^1.1, 10^1.2, 10^1.3, ...
    If you need a sequence that goes like 1, 2, 3, ..., 10, 20, 30, ... This is not the right class.
    """

    def __init__(self, start: float, spacing: float, events: Union[EventType, Set[EventType]]):
        super().__init__(events)
        self.start = start
        self.spacing = spacing
        self.i = 0

    def on_event_happens(self):
        pass

    def should_event_happen(self, global_step: int) -> bool:
        threshold = self.getNextThreshold()
        if global_step >= threshold:
            while threshold == self.getNextThreshold():
                self.i += 1
            return True
        else:
            return False

    def getNextThreshold(self) -> int:
        return int(self.start * 10**(self.i * self.spacing))


class CallbackAtRatchetingInterval(CombinedCallback):
    """
    Starts at a given amount and ratchets up the step size after every N increases. For example:
        start 10, 9 steps: 10, 20, 30, ..., 100, 200, 300, ..., 1000, 2000, 3000, ...
        start 20, 4 steps: 20, 40, 60, 80, 100, 200, 300, 400, 500, 1000, 1500, 2000, 2500, 5000, 7500, 1000, ...
    """

    def __init__(self, start: int, steps_between_ratchets: int, events: Union[EventType, Set[EventType]]):
        super().__init__(events)
        self.increments_between_ratchets = steps_between_ratchets
        self.increments_since_last_ratchet = 0

        self.next_threshold = start
        self.delta_threshold = start

    def on_event_happens(self):
        pass

    def should_event_happen(self, global_step: int) -> bool:
        if global_step >= self.next_threshold:
            self.next_threshold += self.delta_threshold
            self.increments_since_last_ratchet += 1
            if self.increments_since_last_ratchet >= self.increments_between_ratchets:
                self.delta_threshold = self.next_threshold
                self.increments_since_last_ratchet = 0

            return True
        else:
            return False


class SaveTokeniserWithCheckpoints(TrainerCallback):
    """
    Every time a model is saved, add the tokeniser in the same folder, so that you can load both from the
    same checkpoint even if the Trainer is unaware of the tokeniser.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        output_dir = Path(args.output_dir) / f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"  # NOTE: There is one case where this will use the wrong folder, which is when you run trainer.save_model manually since it will save to the parent folder without creating a folder and hence without using global_step.
        if not output_dir.is_dir():
            warnings.warn(f"Tokeniser wasn't saved; tried predicting folder path for the latest checkpoint, but {output_dir.as_posix()} apparently doesn't exist.")
            return

        self.tokenizer.save_pretrained(output_dir)


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


__all__ = ["EvaluateBeforeTrainingCallback", "EventType", "CallbackAtTimeInterval", "SaveTokeniserWithCheckpoints", "CheckpointLastModel"]