from ._general import CompareSentencesGLUETask

class RTE(CompareSentencesGLUETask):
    def __init__(self):
        super().__init__("rte", num_labels=2)
