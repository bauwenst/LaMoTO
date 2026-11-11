from ._general import CompareSentencesGLUETask

class RTE(CompareSentencesGLUETask):
    def __init__(self):
        super().__init__("RTE", n_classes=2)
