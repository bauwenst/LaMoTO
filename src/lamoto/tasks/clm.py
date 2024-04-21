"""
Inspired by Edwin Rijgersberg's script to train GEITje.
    https://github.com/Rijgersberg/GEITje

TODO: When you don't have WandB logging, you should definitely have Fiject callbacks!
"""
# Types
from typing import Iterable, Tuple, Optional, List
from abc import abstractmethod, ABC

# Basic libs
import time
import numpy as np

# ML libs
from tqdm.auto import tqdm
import wandb
import torch
import datasets
from datasets import Dataset
from transformers import \
    Trainer, TrainingArguments, SchedulerType, \
    DataCollatorForLanguageModeling, \
    AutoConfig, AutoModelForCausalLM, AutoTokenizer, \
    PreTrainedTokenizerBase, PretrainedConfig, PreTrainedModel
from transformers.training_args import OptimizerNames

# Custom libs
from tktkt.files.paths import DataPaths
from fiject.hooks.transformers import EvaluateBeforeTrainingCallback

# Relative
from ._core import ModelAugmentation
from ..measuring.ppl import ppl


# An "effective batch" is all the examples used to compute the gradient of one gradient descent step.
# Classically, the loss function looks like sum_{i=1}^N loss(x_i, y_i). You compute that sum by splitting the effort
# across devices and, per device, splitting the work into several runs because of memory limitations.
EXAMPLES_PER_EFFECTIVE_BATCH = 512   # From the OpenAI GPT-2 paper.
EXAMPLES_PER_DEVICEBATCH = 4        # A devicebatch is just whatever fits on the GPU, not N.
TOTAL_TOKEN_BUDGET = 10_000_000_000  # Could've been batches like in the BPE-knockout paper, but for CLMs this metric is more popular. In any case, we're just going to train some CLMs until our wall time runs out.

TOTAL_CHECKPOINTS = 10  # Could also specify batches_per_checkpoint, which means more checkpoints for longer training. I do it relatively because that makes more sense.
EFFECTIVE_BATCHES_BETWEEN_EVALUATIONS = 128
EXAMPLES_PER_EVALUATION = 2**14
PPL_STRIDE = 1/8  # Which fraction of the model's context length we stride in the perplexity function. The complement of this is the amount of context the first token of the second chunk of an example sees. 1/contextlength is slowest but gives actual perplexity, whilst 1.0 is fastest but means that long examples act like multiple independent examples.

LEARNING_RATE = 2e-5
L2_REGULARISATION = 0.01


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
    for row in dataset:
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


from transformers.trainer import DataLoader, EvalLoopOutput, EvalPrediction, denumpify_detensorize, deepspeed_init, logger, has_length
class TrainerWithoutEvaluationLoop(Trainer):

    def evaluation_loop(self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Version of the evaluation loop where, rather than looping over a dataloader, we call compute_metrics on Nones.
        The reason for not altering evaluate() or get_eval_dataloader() is that we now still have the benefits of
        getting speed metrics and of logging.

        We basically cut out everything to do with logits/labels, while keeping the acceleration setup.
        """
        # Copy-pasted from Trainer.evaluation_loop
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


class Pretraining(ABC):

    def __init__(self, checkpoint: str, from_scratch: bool, custom_context_length: int=None, wandb_project: str=""):
        self.checkpoint = checkpoint
        self.from_scratch = from_scratch
        self.custom_context_length = custom_context_length
        self.wandb_project = wandb_project

    @abstractmethod
    def loadDataset(self):
        pass

    def train(self, model_augmentation: ModelAugmentation=None):
        base_name = self.checkpoint[self.checkpoint.rfind("/")+1:]
        global_model_identifier = base_name \
                                + ("" if not model_augmentation else ("-" + model_augmentation.name)) \
                                + f"_CLM_{time.strftime('%F_%X').replace(':', '-')}"

        # Set up paths for checkpointing
        PATH_CHECKPOINTS = DataPaths.pathToCheckpoints() / global_model_identifier
        PATH_CHECKPOINTS.mkdir(exist_ok=True, parents=True)

        # Get model
        if self.from_scratch:
            config: PretrainedConfig = AutoConfig.from_pretrained(self.checkpoint)
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
                self.checkpoint,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
                # low_cpu_mem_usage=True,
                # use_flash_attention_2=True  # Not supported for GPT-2
            )

        # We re-use the old tokeniser.
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, use_fast=False)
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

        save_steps = n_gradient_descents // TOTAL_CHECKPOINTS + 1
        eval_steps = EFFECTIVE_BATCHES_BETWEEN_EVALUATIONS
        warmup_steps = int(n_gradient_descents * 0.1)

        # Training args
        training_args = TrainingArguments(
            # Optimisation (adding all of this in the TrainingArguments because apparently Trainer knows how to use HuggingFace `accelerate` whereas I only know the old optimisers)
            max_steps=n_gradient_descents,

            optim=OptimizerNames.ADAMW_TORCH,
            learning_rate=LEARNING_RATE,
            lr_scheduler_type=SchedulerType.CONSTANT_WITH_WARMUP,
            warmup_steps=warmup_steps,
            weight_decay=L2_REGULARISATION,

            # Batches
            per_device_train_batch_size=EXAMPLES_PER_DEVICEBATCH,
            gradient_accumulation_steps=n_accumulations,

            # Style of computations
            gradient_checkpointing=True,  # Good explanation with animations: https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9
            bf16=torch.cuda.is_bf16_supported(),

            # Evaluation
            evaluation_strategy="steps",
            eval_steps=eval_steps,
            per_device_eval_batch_size=EXAMPLES_PER_DEVICEBATCH,
            eval_accumulation_steps=n_accumulations,  # "Number of predictions steps to accumulate the output tensors for, before moving the results to the CPU."

            # Checkpointing
            save_strategy="steps",
            save_steps=save_steps,
            output_dir=PATH_CHECKPOINTS.as_posix(),

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
                # EvaluateBeforeTrainingCallback()
            ],

            compute_metrics=lambda _: {k:v for k,v in zip(["NLL", "PPL"], ppl(model, tokenizer, valid_dataset, PPL_STRIDE, EXAMPLES_PER_EVALUATION))},
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        )
        trainer.train()
        trainer.save_model()
        # trainer.push_to_hub()


class PretrainingC4(Pretraining):

    def loadDataset(self):
        return datasets.load_dataset("allenai/c4", "en", streaming=True)
