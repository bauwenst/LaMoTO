from ._general import CompareSentencesGLUETask

class MRPC(CompareSentencesGLUETask):
    def __init__(self):
        super().__init__("MRPC", n_classes=2)  # Uses accuracy rather than F1 because the positives are overrepresented.
