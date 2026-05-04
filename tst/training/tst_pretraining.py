from tst.preamble import *

from lamoto.tasks import MLM, CLM, FineWeb


def tst_roberta():
    task = MLM(FineWeb("English"), use_perplexity=False)
    task.train()


def tst_gpt():
    task = CLM(FineWeb("English"), use_perplexity=True)
    task.train()


if __name__ == "__main__":
    tst_roberta()
