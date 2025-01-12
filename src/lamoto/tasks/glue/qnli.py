from ._general import CompareSentencesGLUETask

class QNLI(CompareSentencesGLUETask):
    def __init__(self):
        super().__init__("QNLI", num_labels=2, text_field1="question", text_field2="sentence")  # Uses accuracy rather than F1 because the dataset has a skew of exactly 50%.

    # TODO: Temporary fix until HuggingFace resolve this bug. https://github.com/huggingface/transformers/issues/33985
    def _getMaxInputLength(self) -> int:
        return super()._getMaxInputLength() - 1