"""
Bits-per-character is defined as the average character cross-entropy of a text sequence. Under some assumptions about
token probability models, you can derive BPC from them too.
https://datascience.stackexchange.com/a/94467/141432

Let the alphabet be A and let the sequence of text be the characters x_1 ... x_T. Then the BPC is defined as

    BPC = \frac{1}{T} \sum_{t=1}^T H(P_t, \hat P_t) = \frac{1}{T} \sum_{t=1}^T \sum_{c \in A} -P_t(c) \log_2 \hat P_t(c)

where \hat P_t(c) = \hat P(c \mid x_1 \dots x_{t-1}) and P_t(c) = 1 if c == x_t, which is arguably a very strange definition of language.
This reduces the last sum to a single term:

    BPC = \frac{1}{T} \sum_{t=1}^T -\log_2 \hat P_t(x_t) = \frac{1}{\ln 2}\cdot \frac{1}{T} \sum_{t=1}^T -\ln \hat P_t(x_t)

Define a new time scale with aggregated characters representing tokens: y_1 = x_1 ... x_{t_1}, ..., y_N = x_{t_{N-1}+1} ... y_{t_N}.
Then

    BPC = \frac{1}{\ln 2}\cdot \frac{1}{T} \sum_{t=1}^T -\ln \hat P_t(x_t)
        = \frac{1}{\ln 2}\cdot \frac{1}{T} \sum_{n=1}^N \sum_{i=t_{n-1}+1}^{t_n} -\ln \hat P_i(x_i)
        = \frac{1}{\ln 2}\cdot \frac{1}{T} \sum_{n=1}^N -\ln\left(\prod_{i=t_{n-1}+1}^{t_n} \hat P_i(x_i)\right)

The assumption is now that

    \prod_{i=t_{n-1}+1}^{t_n} \hat P_i(x_i) = \prod_{i=t_{n-1}+1}^{t_n} \hat P(x_i \mid x_1 \dots x_{i-1})
                                            = \prod_{i=t_{n-1}+1}^{t_n} \hat P(x_i \mid x_1 \dots x_{i-1})
                                            = \prod_{i=t_{n-1}+1}^{t_n} \hat P(x_i \mid y_1, ..., y_{n-1}, x_{t_{n-1}+1}, ..., x_{i-1})
                                            = \hat P(y_n \mid y_1, ..., y_{n-1}).

Hence, we get

    BPC = \frac{1}{\ln 2}\cdot \frac{1}{T} \sum_{n=1}^N -\ln \hat P(y_n \mid y_1, ..., y_{n-1})  = \frac{1}{\ln 2}\cdot \frac{1}{T} \sum_{n=1}^N NLL_n

see also https://arxiv.org/pdf/2404.09937. Since perplexity is defined as the exponential of average token negative log likelihood, i.e.

    PPL = e^{\frac{1}{N} \sum_{n=1}^N NLL_n}

You can express NPC in terms of PPL as

    BPC = \frac{N}{T} \cdot \frac{\ln(PPL)}{\ln(2)}
"""
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from datasets import Dataset
from typing import Tuple, Dict

from math import log as ln

from ._core import AutonomousMetric
from .ppl import ppl, PPL_Parameters


class BitsPerCharacter(AutonomousMetric):

    def computeFromEnvironment(self) -> Dict[str, float]:
        params = PPL_Parameters.extractFromTask(self.environment.hyperparameters)
        b, c = bpc(self.environment.model, self.environment.tokeniser, self.environment.getDatasetWithoutCollator(),
                   stride_fraction=params.stride_fraction)
        return {
            "bpc": b,
            "total_chars": c
        }


def bpc(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, dataset: Dataset, stride_fraction: float,
        tqdm_dataset_size: int=None) -> Tuple[float, int]:
    PPL, _, N_tokens = ppl(model, tokenizer, dataset, stride_fraction, tqdm_dataset_size)

    # Compute N_characters (note: this requires that the dataset can be iterated with the same return values both times)
    N_chars = 0
    for example in dataset:
        text = example["text"]
        N_chars += len(text)

    return N_tokens/N_chars * ln(PPL)/ln(2), N_chars
