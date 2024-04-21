import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from datasets import Dataset
from typing import Tuple


def ppl(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, validation_dataset: Dataset, stride_fraction: float,
        tqdm_dataset_size: int=None) -> Tuple[float, float]:
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
    stride_fraction = int(stride_fraction * window_size)

    # Iterate over examples and keep non-averaged NLLs for each.
    nlls = []
    total_tokens = 0
    for example in tqdm(validation_dataset, total=tqdm_dataset_size):
        encodings = tokenizer(example["text"], return_tensors="pt")  # This is a 1 x n_tokens batch.
        n_tokens  = encodings.input_ids.size(1)

        next_token_to_predict = 0
        for window_start in range(0, n_tokens, stride_fraction):  # Notice how the start of the context is INSIDE the previous context.
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
