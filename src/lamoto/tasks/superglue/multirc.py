from datasets import DatasetDict

from ._general import CompareSentencesSuperGLUETask
from ...util.datasets import replaceDatasetColumns_OneExampleToOneExample


class MultiRC(CompareSentencesSuperGLUETask):
    """
    Binary answer verification.

    The task consists of triplet inputs
      (paragraph, question, answer)
    where the goal is to classify whether the given answer answers the given question given the given context.

    You can format this as (paragraph + question, answer) because that's how test questions are often phrased. Then it
    is just as hard as BoolQ (except BoolQ is answering whilst MultiRC is answer verification).

    The SuperGLUE paper writes about this:
     "For MultiRC, because each question can have more than one correct answer, we feed each answer representation into
      a logistic regression classifier."
    Not sure what they mean, but anyway, it's just a binary classification problem.
    """

    def __init__(self):
        super().__init__(task_name="MultiRC", num_labels=2,
                         text_field1="paragraph+question", text_field2="answer")

    def _prepareDataset(self, dataset: DatasetDict) -> DatasetDict:
        def preprocess(example):
            return self.tokenizer(" ".join(example[key] for key in self._field1.split("+")), example[self._field2],
                                  add_special_tokens=self.hyperparameters.add_special_tokens, truncation=True, max_length=self._getMaxInputLength())

        return replaceDatasetColumns_OneExampleToOneExample(dataset, preprocess, but_keep={"label"})
