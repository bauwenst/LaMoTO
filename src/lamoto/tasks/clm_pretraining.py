"""
Inspired by Edwin Rijgersberg's script to train GEITje.
    https://github.com/Rijgersberg/GEITje

FIXME: This file's imports are not corrected yet since importing.
"""
# Types
from typing import Iterable, Tuple
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

# Relative
from ..model.factory import insertHierarchicalLookupGivenTokeniser

# An "effective batch" is all the examples used to compute the gradient of one gradient descent step.
# Classically, the loss function looks like sum_{i=1}^N loss(x_i, y_i). You compute that sum by splitting the effort
# across devices and, per device, splitting the work into several runs because of memory limitations.
EXAMPLES_PER_EFFECTIVE_BATCH = 512   # From the OpenAI GPT-2 paper.
EXAMPLES_PER_DEVICEBATCH = 32        # A devicebatch is just whatever fits on the GPU, not N.
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
        new_ids = tokenizer(row[key], max_length=1_000_000)['input_ids']  # max_length=None will give a warning because it assumes tokeniser output is passed to the model without further processing.
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


def ppl(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, validation_dataset: Dataset) -> Tuple[float,float]:
    """
    Causal perplexity has two boundary conditions:
        - One "document" (a coherent sequence of sentences) cannot be conditioned on another.
        - If you have a fixed context length and a very long example, what you cannot do is process the example as
          several examples in sequence, because that means you will suddenly lose all context in the middle of the example.
    To ensure the first, you cannot use packing. To ensure the second, I use the strided implementation of
        https://huggingface.co/docs/transformers/perplexity#example-calculating-perplexity-with-gpt-2-in--transformers
    which does use a kind of packing, so I changed that.

    The algorithm looks like this. Imagine you have a maximum input length of 9, i.e. the maximum amount of tokens you
    can send through the model as context + prediction targets is 9. For a long document, you could compute perplexity as
    if the document consists of many documents of length 9. With prediction represented by { }:
        {a a a b b b c c c}d d d e e e f f f g g g h h h i i i
         a a a b b b c c c{d d d e e e f f f}g g g h h h i i i
         a a a b b b c c c d d d e e e f f f{g g g h h h i i i}
    ...where within one prediction context, earlier tokens are used as known context for later tokens, but everything is
    predicted. Yet, this means the first token in the second piece has no context. You are, however, progressing very
    fast through your predictions, since every token in your budget is also a predicted token.
    The trade-off we now make is to instead only predict a small stride of tokens every run, and fill the rest of the
    budget with context from the previous piece. Let the prediction still be { }, and let the total context be [ ]:
        [{a  a  a  b  b  b  c  c  c}]d  d  d  e  e  e  f  f  f  g  g  g  h  h  h  i  i  i
          a  a  a [b  b  b  c  c  c {d  d  d}]e  e  e  f  f  f  g  g  g  h  h  h  i  i  i
          a  a  a  b  b  b [c  c  c  d  d  d {e  e  e}]f  f  f  g  g  g  h  h  h  i  i  i
          a  a  a  b  b  b  c  c  c [d  d  d  e  e  e {f  f  f}]g  g  g  h  h  h  i  i  i
          a  a  a  b  b  b  c  c  c  d  d  d [e  e  e  f  f  f {g  g  g}]h  h  h  i  i  i
          a  a  a  b  b  b  c  c  c  d  d  d  e  e  e [f  f  f  g  g  g {h  h  h}]i  i  i
          a  a  a  b  b  b  c  c  c  d  d  d  e  e  e  f  f  f [g  g  g  h  h  h {i  i  i}]

    Note that the last token of the predictions is actually never used because no label is known for it inside the
    window. Hence, in practice, the tokens that partake in the loss are actually:
        [{a  a  a  b  b  b  c  c} c] d  d  d  e  e  e  f  f  f  g  g  g  h  h  h  i  i  i
          a  a  a [b  b  b  c  c {c  d  d} d] e  e  e  f  f  f  g  g  g  h  h  h  i  i  i
          a  a  a  b  b  b [c  c  c  d  d {d  e  e} e] f  f  f  g  g  g  h  h  h  i  i  i
          a  a  a  b  b  b  c  c  c [d  d  d  e  e {e  f  f} f] g  g  g  h  h  h  i  i  i
          a  a  a  b  b  b  c  c  c  d  d  d [e  e  e  f  f {f  g  g} g] h  h  h  i  i  i
          a  a  a  b  b  b  c  c  c  d  d  d  e  e  e [f  f  f  g  g {g  h  h} h] i  i  i
          a  a  a  b  b  b  c  c  c  d  d  d  e  e  e  f  f  f [g  g  g  h  h {h  i  i} i]
    """
    window_size = model.config.max_position_embeddings
    stride = int(PPL_STRIDE * window_size)

    # Iterate over examples and keep non-averaged NLLs for each.
    nlls = []
    total_tokens = 0
    for example in validation_dataset:
        encodings = tokenizer(example["text"], return_tensors="pt")  # This is a 1 x n_tokens batch.
        n_tokens  = encodings.input_ids.size(1)

        next_token_to_predict = 0
        for window_start in tqdm(range(0, n_tokens, stride)):  # Notice how the start of the context is INSIDE the previous context.
            window_end = min(window_start + window_size, n_tokens)  # exclusive bound
            n_tokens_to_predict_in_window = window_end - next_token_to_predict  # usually equal to the stride

            input_ids  = encodings.input_ids[:, window_start:window_end].to(model.device)
            target_ids = input_ids.clone()
            target_ids[:, 0:-n_tokens_to_predict_in_window] = -100  # This makes the labels look like [-100, -100, -100, -100, ..., 1, 2, 3, 4, 5] where the -100 is context that has already been predicted before.

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)

                # Loss is calculated using CrossEntropyLoss, which is an average (over the tokens that aren't labelled -100).
                # N.B.: the model only calculates loss over n_tokens_to_predict_in_window-1 labels. The reason is that actually,
                #       the label for token i should not be token i (because you can already see it at the input) but
                #       token i+1. HuggingFace allows us to let target_ids = input_ids.clone() but shifts the labels left
                #       internally, meaning that the final token in the window has no more label.
                #
                #       Given this, I added the three -1's below. They weren't in the original code.
                nlls.append((n_tokens_to_predict_in_window-1)*outputs.loss)

            next_token_to_predict = window_end-1  # This -1 is not in the original, but you do need it since window_end-1 is the last token for which a logit is computed and it is that logit that is shifted out of the loss, so you need to re-predict it.
            if next_token_to_predict == n_tokens-1:  # Since there is no next token for the final token (which is presumably EOS), you actually have to stop there.
                break

        total_tokens += n_tokens-1

    averaged_nll = (torch.stack(nlls).sum() / total_tokens).item()
    return averaged_nll, np.exp(averaged_nll)


class Pretraining(ABC):

    def __init__(self, checkpoint: str, from_scratch: bool, custom_context_length: int=None):
        self.checkpoint = checkpoint
        self.from_scratch = from_scratch
        self.custom_context_length = custom_context_length

    @abstractmethod
    def loadDataset(self):
        pass

    def train(self, do_hel=False):
        # Names
        base_name = self.checkpoint[self.checkpoint.rfind("/")+1:]
        global_model_identifier = base_name + "-HEL"*do_hel + "_CLM" + f"_{time.strftime('%F_%X').replace(':', '-')}"
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

        if do_hel:
            insertHierarchicalLookupGivenTokeniser(model, tokenizer)

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
            project="HEL",
            group=base_name,
            name=global_model_identifier,
            tags=["HEL", "CLM"] if do_hel else []
        )

        # optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REGULARISATION)
        # scheduler = transformers.optimization.get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_steps)
        trainer = Trainer(
            args=training_args,
            # optimizers=(optimizer,scheduler),

            model=model,
            tokenizer=tokenizer,

            train_dataset=packed_train_dataset,
            eval_dataset=[],  # We explicitly do not want to run classic prediction through the model head; the computeMetrics function generates data from scratch.

            compute_metrics=lambda _: {k:v for k,v in zip(["NLL", "PPL"], ppl(model, tokenizer, valid_dataset))},
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        )
        trainer.train()
        trainer.save_model()
        # trainer.push_to_hub()


class PretrainingC4(Pretraining):

    def loadDataset(self):
        return datasets.load_dataset("allenai/c4", "en", streaming=True)
