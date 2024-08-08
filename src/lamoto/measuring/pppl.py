from typing import Tuple, Dict
from tqdm.auto import tqdm
from dataclasses import dataclass

from math import exp
import torch
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ._core import MetricHyperparameters, AutonomousMetric


@dataclass
class PPPL_Parameters(MetricHyperparameters):
    right_fraction: float


class PseudoPerplexity(AutonomousMetric):

    def computeFromEnvironment(self) -> Dict[str, float]:
        params = PPPL_Parameters.extractFromTask(self.environment.hyperparameters)
        p, n, t = pppl(
            model=self.environment.model,
            tokenizer=self.environment.tokeniser,
            validation_dataset=self.environment.validation_dataset,
            rightward_fraction=params.right_fraction
        )
        return {
            "pppl": p,
            "nll": n,
            "total_tokens": t
        }


def pppl(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, validation_dataset: Dataset, rightward_fraction: float=0.5,
         tqdm_dataset_size: int=None) -> Tuple[float, float, int]:
    """
    Masked perplexity (pseudo-perplexity). Same boundary conditions as causal perplexity apply, except the task is not to
    predict the next token but to predict the token under a mask. This is fundamentally different to causal PPL, because
    it means that the embedding you use to predict a token label -- in causal PPL, that label is the next token, in masked PPL
    it is the current token, had the mask not been there -- cannot even be used as context by tokens to the right of it.
    The result is that you are forced to recompute the window for each token, i.e. you cannot stride. So, you move your
    window by 1 token for each token you predict.

    A different decision still has to be made. Perplexity is supposed to use infinite context. We don't have infinite
    context, so we have to decide which context is used for a prediction.
    - In causal PPL, for long documents and a model context length L, every token past the Lth token should use the most
      context possible in order to simulate an infinite context length, which is just the L preceding tokens. (This is
      the case if you slide your window over by 1 on each prediction. This is the minimally lossy case but maximally computationally
      expensive; because you can do striding to reduce the amount of inference, there is also a maximally lossy case where
      you slide the window over by L tokens, so the (L+1)th token has 0 tokens of context again, whilst the document has L available in theory.)

    - For masked PPL, you need to simulate an infinite leftward AND rightward context, given L tokens of space. It's
      not obvious whether left or right is more informative, but it's also not obvious if context should come equally
      from both sides. The fraction that comes from the right is a hyperparameter.

    The implementation at https://stackoverflow.com/a/70482924 has two flaws:
        1. It hardcodes the location of special tokens.
        2. It does not support sentences longer than the model's context length.

    The implementation below supports any special token format, and any context length. The latter can be done naively
    by having equal context on the left and right, or parameterised by the proportion of left/right context. This is the
    most general version of the algorithm and I have implemented that below.
    
    FIXME: This function crashes for long examples, because the batch we send to the model for an example of N
          tokens with a context length of L tokens is N x min(N,L) which could e.g. be a batch of 1024 examples, sent
          straight to the device. Normally one device handles at most 64 examples in a batch...
    """
    # Iterate over examples and keep non-averaged NLLs for each.
    nlls = []
    total_tokens = 0
    for example in tqdm(validation_dataset, total=tqdm_dataset_size):
        if "input_ids" in example:
            encodings = torch.tensor(example["input_ids"])
        else:
            encodings = tokenizer(example["text"], return_tensors="pt").input_ids  # This is a 1 x n_tokens batch.
        tokens = encodings.squeeze()
        assert tokens.dim() == 1  # The .squeeze() will remove the batch dimension only if there is 1 example in the batch. PPPL makes a batch out of 1 example, so can't take batches itself.
        n = tokens.size(0)

        # Sizes (reset every iteration because we alter them for sentences shorter than the model context's length)
        L = model.config.max_position_embeddings
        L_middle = 1
        L_right = int((L - L_middle) * rightward_fraction)
        L_left = (L - L_middle) - L_right

        if n >= L_left + L_middle + L_right:  # Normal case; there can be a top, middle and bottom.
            pass
        elif n >= L_left + L_middle:  # Edge case: smaller right side.
            L_right = n - (L_left + L_middle)
        else:  # Edge case: no right side, no middle, and smaller left side.
            L_right  = 0
            L_middle = 0
            L_left   = n

        L = L_left + L_middle + L_right  # This is smaller than the original L when n was smaller than the original L.

        # Contexts
        top    = tokens[:L].repeat(L_left, 1)  # Copies the first available window in the context.
        middle = tokens.unfold(dimension=0, size=L, step=1) if L_middle else torch.zeros(size=(0,L), dtype=torch.int64)  # Window shifting with step 1 of the same size as that available window.
        bottom = tokens[n-L:].repeat(L_right, 1)  # Tolerates L_right == 0. Also, n-L can never be negative since L <= n by the above.

        # Masks
        mask_top    = torch.eye(L_left, L)
        mask_middle = torch.zeros(n-L_left-L_right, L)
        if L_middle:
            mask_middle[:, L_left] = 1  # L_left is the token that follows the first L_left tokens, i.e. the middle token.
        mask_bottom = torch.roll(torch.eye(L_right, L), shifts=-L_right, dims=1)

        # Concatenate these two sets of 3 matrices.
        full_ids  = torch.cat((top, middle, bottom))
        full_mask = torch.cat((mask_top, mask_middle, mask_bottom))

        # Filter out the rows belonging to special tokens.
        special_tokens_mask = torch.tensor(tokenizer.get_special_tokens_mask(tokens, already_has_special_tokens=True))
        full_ids  = full_ids[special_tokens_mask == 0, :]
        full_mask = full_mask[special_tokens_mask == 0, :]
        n -= special_tokens_mask.sum().item()

        # Finally, apply the constructed mask to the constructed IDs.
        input_ids = full_ids.masked_fill(full_mask == 1, tokenizer.mask_token_id)
        labels    = full_ids.masked_fill(full_mask != 1, -100)
        attention_mask = torch.ones_like(input_ids)

        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids.to(model.device),
                attention_mask=attention_mask.to(model.device),
                labels=labels.to(model.device)
            )
            loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss
            nlls.append(n*loss)  # n is adjusted at this point to the amount of predicted tokens.

        total_tokens += n

    averaged_nll = (torch.cat(nlls).sum() / total_tokens).item()
    return exp(averaged_nll), averaged_nll, total_tokens
