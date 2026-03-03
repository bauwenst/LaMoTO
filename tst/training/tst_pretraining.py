from tst.preamble import *

from lamoto.tasks import MLM, CLM, Fineweb


def tst_roberta():
    task = MLM(Fineweb("English"), use_perplexity=False)
    task.train()


def tst_gpt():
    task = CLM(Fineweb("English"), use_perplexity=True)
    task.train()


if __name__ == "__main__":
    tst_roberta()
