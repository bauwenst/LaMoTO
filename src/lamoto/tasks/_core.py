from typing import Protocol, Any, Dict, Type, List, Tuple
from dataclasses import dataclass
from abc import abstractmethod, ABC

import time

from datasets import DatasetDict
from transformers import DataCollator, Trainer, TrainingArguments, AutoTokenizer, RobertaTokenizer
import transformers.optimization  # TODO: Trainer handles this and you shouldn't do it manually.
import torch
import evaluate
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from fiject.hooks.transformers import FijectCallback, EvaluateBeforeTrainingCallback
from tktkt.files.paths import DataPaths

from ..model.factory import fromCheckpoint  # TODO: Oops


##################################  TODO: These should become task arguments, perhaps as a config dataclass.
MAX_TRAINING_EPOCHS = 10
BATCH_SIZE = 32
BATCHES_WARMUP = 100  # The RoBERTa paper says, for GLUE tasks, they warm up for 6% of all batches across 10 epochs. That's in the ballpark of 100 batches.
EVALS_PER_EPOCH = 9
EVALS_OF_PATIENCE = 9

LEARNING_RATE = 2e-5
L2_REGULARISATION = 0.01
MAX_INPUT_LENGTH = 512

CHECKPOINT = "roberta-base"
ADD_SPECIAL_TOKENS = True
##################################


class Metric(Protocol):
    def compute(self, predictions: Any, references: Any) -> Dict[str,Any]:
        pass


class MetricRegistry:

    def __init__(self):
        self.custom_metrics: Dict[str,Type[Metric]] = dict()

    def registerMetric(self, name: str, metric: Type[Metric]):
        if name in self.custom_metrics:
            raise ValueError(f"Cannot register custom metric {name} because it already exists.")

        self.custom_metrics[name] = metric

    def load(self, name: str) -> Metric:
        return self.custom_metrics[name]() if name in self.custom_metrics else evaluate.load(name)

METRICS = MetricRegistry()


@dataclass
class MetricSetup:
    to_compute: List[str]               # Names of all the HuggingFace evaluate metrics to load and compute in the end.
    to_track: Dict[str, Dict[str,str]]  # metric name -> result name -> formatted name, used for graphing intermediate evaluations.


class FinetuningTask(ABC):

    def __init__(self, task_name: str, metrics: MetricSetup, automodel_class: Type[_BaseAutoModelClass], **automodel_args):
        self.task_name = task_name

        self.auto = automodel_class
        self.auto_args = automodel_args

        self.metrics: Dict[str, Metric] = {name: METRICS.load(name) for name in metrics.to_compute}
        self.result_formatting = metrics.to_track

        self.tokenizer: RobertaTokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, add_prefix_space=True)

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

    def train(self, do_hel: bool=False):
        global_model_identifier = CHECKPOINT[CHECKPOINT.rfind("/")+1:] + "-HEL"*do_hel + f"_{self.task_name}_{time.strftime('%F_%X').replace(':', '-')}"

        # Set up paths for checkpointing
        PATH_CHECKPOINTS = DataPaths.pathToCheckpoints() / global_model_identifier
        PATH_CHECKPOINTS.mkdir(exist_ok=True, parents=True)

        # Get dataset
        datasetdict = self.prepareDataset(self.loadDataset())

        # Get the batch generator, a.k.a. collator (https://huggingface.co/docs/transformers/main_classes/data_collator).
        collator = self.getCollator()

        # Get model
        model, _, _ = fromCheckpoint(checkpoint=CHECKPOINT, do_hel=do_hel,
                                     automodel_class=self.auto, **self.auto_args)
        model.to("cuda")

        # Training arguments
        interval = (len(datasetdict["train"]) // BATCH_SIZE) // EVALS_PER_EPOCH
        training_args = TrainingArguments(
            output_dir=PATH_CHECKPOINTS.as_posix(),

            # Training
            num_train_epochs=MAX_TRAINING_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            # Not sure whether you need these given the custom AdamW in the Trainer constructor.
            weight_decay=L2_REGULARISATION,  # L2 regularisation constant
            learning_rate=LEARNING_RATE,  # Not sure if this is still needed

            # Evaluating
            evaluation_strategy="steps",
            eval_steps=interval,

            # Artifacts
            report_to="none",   # Disables weights-and-biases login requirement
            logging_strategy="no",
            push_to_hub=False,

            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            save_strategy="steps",  # Because we want to load the best model at the end, we need to be able to go back to it. Hence, we need to allow saving each evaluation.
            save_steps=interval,    # ... and save on the same interval.
            save_total_limit=1,     # This will keep the last model stored plus the best model if those aren't the same. https://stackoverflow.com/a/67615225/9352077
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REGULARISATION)  # Not using transformers.optimization because it gives a deprecation warning.
        scheduler = transformers.optimization.get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=BATCHES_WARMUP)  # Not using a linear decay because that's the whole point of having Adam.
        trainer = Trainer(
            model=model,
            args=training_args,

            data_collator=collator,
            train_dataset=datasetdict["train"],
            optimizers=(optimizer, scheduler),
            callbacks=[
                EvaluateBeforeTrainingCallback(),
                FijectCallback(global_model_identifier + "_eval_loss", evals_between_commits=EVALS_PER_EPOCH),
                FijectCallback(global_model_identifier + "_eval_task", evals_between_commits=EVALS_PER_EPOCH,
                                                                       metric_names_with_formatting={(metric_name + "_" + result_name): formatting
                                                                                                     for metric_name, result_formats in self.result_formatting.items()
                                                                                                     for result_name, formatting in result_formats.items()}),
                transformers.trainer_callback.EarlyStoppingCallback(early_stopping_patience=EVALS_OF_PATIENCE)  # Patience is the amount of eval calls you can tolerate worsening loss.
            ],

            eval_dataset=datasetdict["validation"],
            compute_metrics=self.computeMetrics,

            preprocess_logits_for_metrics=self.sneakyLogitTransform
        )

        print("=== TRAINING SIZES ===")
        print("Batch size:", BATCH_SIZE)
        print("Training set:")
        print("\t", len(datasetdict["train"]), "examples per epoch")
        print("\t", len(datasetdict["train"]) // BATCH_SIZE, "batches per epoch")
        print("\t", MAX_TRAINING_EPOCHS, "epochs")
        print("\t", (len(datasetdict["train"]) // BATCH_SIZE)*MAX_TRAINING_EPOCHS, "batches in total")
        print("Evaluation set:")
        print("\t", len(datasetdict["validation"]), "examples per evaluation")
        print("\t", 1 + (len(datasetdict["validation"])-1) // BATCH_SIZE, "batches per evaluation")
        print("\t", EVALS_PER_EPOCH, "evals per training epoch")
        print("\t", (len(datasetdict["train"]) // BATCH_SIZE) // EVALS_PER_EPOCH, "training batches between evals")
        print("======================")

        trainer.train()
        trainer.save_model()
        print(trainer.evaluate())
