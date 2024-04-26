"""
Inspired by Edwin Rijgersberg's script to train GEITje.
    https://github.com/Rijgersberg/GEITje
"""
# Types
from dataclasses import dataclass
from typing import Iterable
from abc import abstractmethod, ABC

# Basic libs
import time
from pathlib import Path

# ML libs
import wandb
import torch
import datasets
from datasets import Dataset
from transformers import \
    TrainingArguments, SchedulerType, \
    DataCollatorForLanguageModeling, \
    AutoConfig, AutoModelForCausalLM, AutoTokenizer, \
    PreTrainedTokenizerBase, PretrainedConfig
from transformers.training_args import OptimizerNames

# Custom libs
from tktkt.files.paths import DataPaths
from fiject.hooks.transformers import FijectCallback

# Relative
from ._core import ModelAugmentation
from ..measuring.ppl import ppl
from ..trainer.trainers import TrainerWithoutEvaluationLoop
from ..trainer.callbacks import CheckpointAtTimeInterval, CheckpointLastModel, EvaluateBeforeTrainingCallback


# An "effective batch" is all the examples used to compute the gradient of one gradient descent step.
# Classically, the loss function looks like sum_{i=1}^N loss(x_i, y_i). You compute that sum by splitting the effort
# across devices and, per device, splitting the work into several runs because of memory limitations.
EXAMPLES_PER_EFFECTIVE_BATCH = 512   # From the OpenAI GPT-2 paper.
EXAMPLES_PER_DEVICEBATCH = 64        # A devicebatch is just whatever fits on the GPU, not N.
TOTAL_TOKEN_BUDGET = 10_000_000_000  # Could've been batches like in the BPE-knockout paper, but for CLMs this metric is more popular. In any case, we're just going to train some CLMs until our wall time runs out.

MINUTES_BETWEEN_CHECKPOINTS = 30
EFFECTIVE_BATCHES_BETWEEN_EVALUATIONS = 128
EXAMPLES_PER_EVALUATION = 2**14
PPL_STRIDE = 1/8  # Which fraction of the model's context length we stride in the perplexity function. The complement of this is the amount of context the first token of the second chunk of an example sees. 1/contextlength is slowest but gives actual perplexity, whilst 1.0 is fastest but means that long examples act like multiple independent examples.

LEARNING_RATE = 2e-5
L2_REGULARISATION = 0.01

# Timeout configuration (I tested this and it works, but it sure is a weird use of Python imports... https://github.com/huggingface/datasets/issues/6172#issuecomment-1794876229)
datasets.config.STREAMING_READ_RETRY_INTERVAL = 60   # Seconds between retries; ideally this would work with exponential backoff, but it doesn't, because... HuggingFace engineers.
datasets.config.STREAMING_READ_MAX_RETRIES    = 120  # Retry for up to 2 hours.


def packedDatasetGenerator(dataset: Iterable, tokenizer: PreTrainedTokenizerBase, context_length: int, key='text'):
    """Concatenate ("pack") samples from a dataset into tokenized chunks of `context_length`.

    Used for efficient training of causal models without padding. No special measures are taken
    to disallow a sequence attending to a previous sequence. The model is left to learn the
    unrelatedness of sequences from the presence of the start- and end-of-sequence-tokens
    between the samples, following a similar convention from GPT-3 and T5.
    See https://github.com/huggingface/transformers/issues/17726 for a feature request for
    HuggingFace Transformers.

    The incomplete final chunk at the end of the dataset is discarded.

    :param dataset: Dataset of samples (iterable of dict-like, e.g. Hugging Face dataset)
    :param tokenizer: Callable that tokenizes the samples (e.g. Hugging Face tokenizer)
    :param context_length: number of tokens in packed sequences
    :param key: key of the text field in the sample. Defaults to 'text'
    :yield: dicts of packed input_ids, attention_masks and (self-supervised) labels
    """
    cache = []
    for row in dataset:  # TODO: <--- This can be affected by a network error. If HF doesn't fix their retries, wrap this. https://github.com/huggingface/datasets/pull/6844
        # Add extra IDs to cache
        new_ids = tokenizer(row[key], max_length=1_000_000, truncation=True)['input_ids']  # max_length=None will give a warning because it assumes tokeniser output is passed to the model without further processing.
        if not new_ids[-1] == tokenizer.eos_token_id:  # You need an EOS between examples.
            new_ids.append(tokenizer.eos_token_id)
        cache.extend(new_ids)

        # If the cache is now bigger than the fixed length we want to output: empty it in chunks of that size until
        # the cache needs more data.
        while len(cache) >= context_length:
            chunk = cache[:context_length]
            yield {"input_ids": chunk,
                   "labels": chunk,  # For CLM, the labels ARE the inputs.
                   "attention_mask": [1] * context_length}
            cache = cache[context_length:]


@dataclass
class Pretraining(ABC):

    hub_name: str
    initialise_from_hub: bool
    custom_context_length: int=None
    wandb_project: str=""

    @abstractmethod
    def loadDataset(self):
        pass

    def train(self, model_augmentation: ModelAugmentation=None, resume_from_folder: Path=None):
        """
        :param model_augmentation: Transformation to apply to the model architecture.
        :param resume_from_folder: Folder containing model weights, optimiser state (momenta etc...) and scheduler state
                                   to continue training from. In other words: a local checkpoint.
                                   Should support the model augmentation!
        """
        base_name = self.hub_name[self.hub_name.rfind("/") + 1:]
        global_model_identifier = base_name \
                                + ("" if not model_augmentation else ("-" + model_augmentation.name)) \
                                + f"_CLM_{time.strftime('%F_%X').replace(':', '-')}"

        # Set up paths for checkpointing
        PATH_CHECKPOINTS = DataPaths.pathToCheckpoints() / global_model_identifier
        PATH_CHECKPOINTS.mkdir(exist_ok=True, parents=True)

        # Get model
        if not self.initialise_from_hub:  # Only get the config, then get a randomised model from that config.
            config: PretrainedConfig = AutoConfig.from_pretrained(self.hub_name)
            if self.custom_context_length:
                # max_position_embeddings is the standardised name for context length.
                # Yet, for GPT-2, you will find it as n_positions in the config,
                # which is mapped to max_position embeddings by config.attribute_map.
                if "max_position_embeddings" in config.attribute_map:
                    config.__dict__[config.attribute_map["max_position_embeddings"]] = self.custom_context_length
                else:
                    config.max_position_embeddings = self.custom_context_length

            model = AutoModelForCausalLM.from_config(
                config,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
                # low_cpu_mem_usage=True,
                # use_flash_attention_2=True  # Not supported for GPT-2
            )
        else:  # Can't set a custom context length when the position embeddings have been trained already.
            model = AutoModelForCausalLM.from_pretrained(
                self.hub_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
                # low_cpu_mem_usage=True,
                # use_flash_attention_2=True  # Not supported for GPT-2
            )

        # We re-use the old tokeniser.
        tokenizer = AutoTokenizer.from_pretrained(self.hub_name, use_fast=False)
        if model_augmentation:
            model = model_augmentation.augment(model, tokenizer)
        model.to("cuda")

        # Dataset
        datasetdict   = self.loadDataset()
        train_dataset = datasetdict["train"]     .shuffle(seed=42, buffer_size=10_000)  # https://huggingface.co/docs/datasets/stream#shuffle
        valid_dataset = datasetdict["validation"].shuffle(seed=42, buffer_size=10_000).take(EXAMPLES_PER_EVALUATION)

        # For efficiency, instead of training on batches with any padding at all, we pack examples until the input is full.
        # - GPT-2 has no pad token, but this doesn't really matter because it's actually the attention mask that determines
        #   if a token is processed, so you can replace it by any token you want. https://github.com/stanford-crfm/BioMedLM/issues/4
        if tokenizer.pad_token is None:
            tokenizer.pad_token       = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        # - Wrapper around the dataset
        packed_train_dataset = datasets.IterableDataset.from_generator(
            generator=packedDatasetGenerator,
            gen_kwargs={"dataset": train_dataset,
                        "tokenizer": tokenizer,
                        "context_length": model.config.max_position_embeddings})

        # packed_validation_dataset = datasets.IterableDataset.from_generator(
        #     generator=packedDatasetGenerator,
        #     gen_kwargs={"dataset": valid_dataset,
        #                 "tokenizer": tokenizer,
        #                 "context_length": model.config.max_position_embeddings})

        # Statistics
        n_examples          = TOTAL_TOKEN_BUDGET // model.config.max_position_embeddings
        n_gradient_descents = n_examples // EXAMPLES_PER_EFFECTIVE_BATCH
        n_accumulations     = EXAMPLES_PER_EFFECTIVE_BATCH // (torch.cuda.device_count() * EXAMPLES_PER_DEVICEBATCH)  # The amount of times, to get to one effective batch, you have to push a device batch through all devices in parallel.

        # n_descents_between_saves = n_gradient_descents // TOTAL_CHECKPOINTS
        n_descents_between_evals = EFFECTIVE_BATCHES_BETWEEN_EVALUATIONS
        n_descents_of_warmup = int(n_gradient_descents * 0.1)

        # Training args
        training_args = TrainingArguments(
            # Optimisation (adding all of this in the TrainingArguments because apparently Trainer knows how to use HuggingFace `accelerate` whereas I only know the old optimisers)
            max_steps=n_gradient_descents,

            optim=OptimizerNames.ADAMW_TORCH,
            learning_rate=LEARNING_RATE,
            lr_scheduler_type=SchedulerType.CONSTANT_WITH_WARMUP,
            warmup_steps=n_descents_of_warmup,
            weight_decay=L2_REGULARISATION,

            # Batches
            per_device_train_batch_size=EXAMPLES_PER_DEVICEBATCH,
            gradient_accumulation_steps=n_accumulations,

            # Style of computations
            gradient_checkpointing=True,  # Good explanation with animations: https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9
            bf16=torch.cuda.is_bf16_supported(),

            # Evaluation
            evaluation_strategy="steps",
            eval_steps=n_descents_between_evals,
            per_device_eval_batch_size=EXAMPLES_PER_DEVICEBATCH,
            eval_accumulation_steps=n_accumulations,  # "Number of predictions steps to accumulate the output tensors for, before moving the results to the CPU."

            # Checkpointing via the `DefaultFlowCallback` will be disabled, but we will still save using a time-based callback.
            save_strategy="steps",
            save_steps=0,

            output_dir=PATH_CHECKPOINTS.as_posix(),
            save_total_limit=2,  # We don't want to keep all the checkpoints we make, but we do want to keep the last checkpoint even if it's not the best checkpoint.

            # metric_for_best_model=
            # load_best_model_at_end=

            # Logging
            report_to=["wandb"],
            logging_steps=1,  # Gradient descents between each push to the log.
            logging_first_step=True,
            include_num_input_tokens_seen=True,

            # hub_model_id=new_model_name,
            # hub_private_repo=True,
            # push_to_hub=True,
            # hub_strategy='all_checkpoints',
        )

        wandb.init(
            mode="disabled" if not self.wandb_project else "online",

            project=self.wandb_project,
            group=base_name,
            name=global_model_identifier,
            tags=[model_augmentation.name, "CLM"] if model_augmentation else ["CLM"]
        )

        # optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REGULARISATION)
        # scheduler = transformers.optimization.get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_steps)
        trainer = TrainerWithoutEvaluationLoop(
            args=training_args,
            # optimizers=(optimizer,scheduler),

            model=model,
            tokenizer=tokenizer,

            train_dataset=packed_train_dataset,
            eval_dataset=[],
            callbacks=[
                EvaluateBeforeTrainingCallback(),
                CheckpointAtTimeInterval(minutes=MINUTES_BETWEEN_CHECKPOINTS),
                CheckpointLastModel()
            ] + ([] if self.wandb_project else [
                FijectCallback(global_model_identifier + "_eval_loss", evals_between_commits=5,
                               metric_names_with_formatting={"NLL": "loss"}),
                FijectCallback(global_model_identifier + "_eval_PPL", evals_between_commits=5,
                               metric_names_with_formatting={"PPL": "PPL"}),
            ]),

            compute_metrics=lambda _: {k:v for k,v in zip(["NLL", "PPL"], ppl(model, tokenizer, valid_dataset, PPL_STRIDE, EXAMPLES_PER_EVALUATION))},
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        )
        try:
            trainer.train(resume_from_checkpoint=resume_from_folder.as_posix() if resume_from_folder else None)
            # trainer.save_model()  # 1. We already checkpoint the last model, 2. LM pretraining basically never gets to convergence, and 3. we don't have a metric configured because we're not doing traditional eval (although this is probably not a problem since compute_metrics might be where you get your metric anyway).
            # trainer.push_to_hub()
        except:  # Catches any error that happens during training, and triggers a checkpoint (+ a callback event afterwards, if that's needed by any callback).
            trainer.control.should_save     = True
            trainer.control.should_evaluate = False
            trainer.control.should_log      = False
            trainer._maybe_log_save_evaluate(tr_loss=None, grad_norm=None, model=None, trial=None, epoch=None, ignore_keys_for_eval=None)  # These arguments are imputed anyway.


class PretrainingC4(Pretraining):

    def loadDataset(self):
        return datasets.load_dataset("allenai/c4", "en", streaming=True)
