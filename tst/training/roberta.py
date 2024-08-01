from tst.preamble import *


def test_mlm_training():
    from lamoto.tasks import MLM_C4, SUGGESTED_HYPERPARAMETERS_MLM
    task = MLM_C4()
    task.train(
        hyperparameters=SUGGESTED_HYPERPARAMETERS_MLM
    )


if __name__ == "__main__":
    test_mlm_training()
