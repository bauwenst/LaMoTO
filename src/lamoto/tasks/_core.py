"""
Core fine-tuning script for any task.

# TODO: In the old code I was using for MBR, I was also saving the tokeniser at the end of training. This is still done
#       in the CLM script, but not here. The reason you want to save it is that you otherwise can't use checkpoints
#       without loading the tokeniser from the hub, so you need to pass in two checkpoints.
#       The reason why you want to be careful saving the tokeniser by giving it to Trainer is that Trainer then also uses
#       it for a variety of purposes that you might not intend (like padding etc.).
#       The call you basically just want to insert into Trainer somehow is tokenizer.save_pretrained(save_directory=current_checkpoint.as_posix()).
#       Maybe a callback will work, but you need the path of the latest checkpoint.
"""
from typing import Protocol, Any, Dict, Type, List, Tuple
from dataclasses import dataclass
from abc import abstractmethod, ABC

import time

from datasets import DatasetDict
from transformers import DataCollator, Trainer, TrainingArguments, AutoTokenizer, RobertaTokenizer, PreTrainedModel, PreTrainedTokenizerBase
import transformers.optimization  # TODO: Trainer handles this and you shouldn't do it manually.
import torch
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from fiject.hooks.transformers import FijectCallback
from tktkt.files.paths import DataPaths

from ..augmenting.model import ModelAugmentation
from ..measuring._core import Metric
from ..measuring import METRICS
from ..trainer.callbacks import EvaluateBeforeTrainingCallback


@dataclass
class TaskHyperparameters:
    MAX_TRAINING_EPOCHS: int
    BATCH_SIZE: int
    BATCHES_WARMUP: int  # The RoBERTa paper says, for GLUE tasks, they warm up for 6% of all batches across 10 epochs. That's in the ballpark of 100 batches.
    EVALS_PER_EPOCH: int
    EVALS_OF_PATIENCE: int

    LEARNING_RATE: float
    L2_REGULARISATION: float
    MAX_INPUT_LENGTH: int

    CHECKPOINT: str = "roberta-base"
    ADD_SPECIAL_TOKENS: bool = True


#################################################
DEFAULT_HYPERPARAMETERS = TaskHyperparameters(
    MAX_TRAINING_EPOCHS=10,
    BATCH_SIZE=32,
    BATCHES_WARMUP=100,  # The RoBERTa paper says, for GLUE tasks, they warm up for 6% of all batches across 10 epochs. That's in the ballpark of 100 batches.
    EVALS_PER_EPOCH=9,
    EVALS_OF_PATIENCE=9,

    LEARNING_RATE=2e-5,
    L2_REGULARISATION=0.01,
    MAX_INPUT_LENGTH=512,

    CHECKPOINT="roberta-base",
    ADD_SPECIAL_TOKENS=True
)
#################################################


@dataclass
class MetricSetup:
    to_compute: List[str]               # Names of all the HuggingFace evaluate metrics to load and compute in the end.
    to_track: Dict[str, Dict[str,str]]  # metric name -> result name -> formatted name, used for graphing intermediate evaluations.


class FinetuningTask(ABC):

    def __init__(self, task_name: str, metric_config: MetricSetup, automodel_class: Type[_BaseAutoModelClass], **automodel_args):
        self.task_name       = task_name
        self.metric_config   = metric_config
        self.automodel_class = automodel_class
        self.automodel_args  = automodel_args

        # Fields that can be used by method implementations, but are only instantiated once .train() is called, to
        # avoid loading heavy objects that would be duplicated by the super() call of a task wrapper.
        self.hyperparameters: TaskHyperparameters = None
        self.tokenizer: PreTrainedTokenizerBase = None
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

    def train(self, hyperparameters: TaskHyperparameters=DEFAULT_HYPERPARAMETERS, model_augmentation: ModelAugmentation=None):
        global_model_identifier = hyperparameters.CHECKPOINT[hyperparameters.CHECKPOINT.rfind("/")+1:] \
                                + ("" if not model_augmentation else ("-" + model_augmentation.name)) \
                                + f"_{self.task_name}_{time.strftime('%F_%X').replace(':', '-')}"

        # Set up paths for checkpointing
        PATH_CHECKPOINTS = DataPaths.pathToCheckpoints() / global_model_identifier
        PATH_CHECKPOINTS.mkdir(exist_ok=True, parents=True)

        # Set up missing fields for use in the other method calls.
        self.hyperparameters = hyperparameters
        self.tokenizer = AutoTokenizer.from_pretrained(hyperparameters.CHECKPOINT, add_prefix_space=True)
        self.metrics = {name: METRICS.load(name) for name in self.metric_config.to_compute}

        # Get dataset
        datasetdict = self.prepareDataset(self.loadDataset())

        # Get the batch generator, a.k.a. collator (https://huggingface.co/docs/transformers/main_classes/data_collator).
        collator = self.getCollator()

        # Get model
        model: PreTrainedModel = self.automodel_class.from_pretrained(hyperparameters.CHECKPOINT, **self.automodel_args)
        if model_augmentation:
            model = model_augmentation.augment(model, self.tokenizer)
        model.to("cuda")

        # Training arguments
        interval = (len(datasetdict["train"]) // hyperparameters.BATCH_SIZE) // hyperparameters.EVALS_PER_EPOCH
        training_args = TrainingArguments(
            output_dir=PATH_CHECKPOINTS.as_posix(),

            # Training
            num_train_epochs=hyperparameters.MAX_TRAINING_EPOCHS,
            per_device_train_batch_size=hyperparameters.BATCH_SIZE,
            per_device_eval_batch_size=hyperparameters.BATCH_SIZE,
            # Not sure whether you need these given the custom AdamW in the Trainer constructor.
            weight_decay=hyperparameters.L2_REGULARISATION,  # L2 regularisation constant
            learning_rate=hyperparameters.LEARNING_RATE,  # Not sure if this is still needed

            # Evaluating
            evaluation_strategy="steps",
            eval_steps=interval,

            # Artifacts
            report_to="none",   # Disables weights-and-biases login requirement
            logging_strategy="no",
            push_to_hub=False,

            load_best_model_at_end=True,  # Will take the best model out of its checkpoint directory and load it into self.model, which can then be saved. At the end of Trainer's loop, the following happens: "Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint"
            metric_for_best_model="eval_loss",
            save_strategy="steps",  # Because we want to load the best model at the end, we need to be able to go back to it. Hence, we need to allow saving each evaluation.
            save_steps=interval,    # ... and save on the same interval.
            save_total_limit=1,     # This will keep the last model stored plus the best model if those aren't the same. https://stackoverflow.com/a/67615225/9352077
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparameters.LEARNING_RATE, weight_decay=hyperparameters.L2_REGULARISATION)  # Not using transformers.optimization because it gives a deprecation warning.
        scheduler = transformers.optimization.get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=hyperparameters.BATCHES_WARMUP)  # Not using a linear decay because that's the whole point of having Adam.
        trainer = Trainer(
            model=model,
            args=training_args,

            data_collator=collator,
            train_dataset=datasetdict["train"],
            optimizers=(optimizer, scheduler),
            callbacks=[
                EvaluateBeforeTrainingCallback(),
                FijectCallback(global_model_identifier + "_eval_loss", evals_between_commits=hyperparameters.EVALS_PER_EPOCH),
                FijectCallback(global_model_identifier + "_eval_task", evals_between_commits=hyperparameters.EVALS_PER_EPOCH,
                                                                       metric_names_with_formatting={(metric_name + "_" + result_name): formatting
                                                                                                     for metric_name, result_formats in self.metric_config.to_track.items()
                                                                                                     for result_name, formatting in result_formats.items()}),
                transformers.trainer_callback.EarlyStoppingCallback(early_stopping_patience=hyperparameters.EVALS_OF_PATIENCE)  # Patience is the amount of eval calls you can tolerate worsening loss.
            ],

            eval_dataset=datasetdict["validation"],
            compute_metrics=self.computeMetrics,

            preprocess_logits_for_metrics=self.sneakyLogitTransform
        )

        print("=== TRAINING SIZES ===")
        print("Batch size:", hyperparameters.BATCH_SIZE)
        print("Training set:")
        print("\t", len(datasetdict["train"]), "examples per epoch")
        print("\t", len(datasetdict["train"]) // hyperparameters.BATCH_SIZE, "batches per epoch")
        print("\t", hyperparameters.MAX_TRAINING_EPOCHS, "epochs")
        print("\t", (len(datasetdict["train"]) // hyperparameters.BATCH_SIZE)*hyperparameters.MAX_TRAINING_EPOCHS, "batches in total")
        print("Evaluation set:")
        print("\t", len(datasetdict["validation"]), "examples per evaluation")
        print("\t", 1 + (len(datasetdict["validation"])-1) // hyperparameters.BATCH_SIZE, "batches per evaluation")
        print("\t", hyperparameters.EVALS_PER_EPOCH, "evals per training epoch")
        print("\t", (len(datasetdict["train"]) // hyperparameters.BATCH_SIZE) // hyperparameters.EVALS_PER_EPOCH, "training batches between evals")
        print("======================")

        trainer.train()
        trainer.save_model()
        print(trainer.evaluate())
