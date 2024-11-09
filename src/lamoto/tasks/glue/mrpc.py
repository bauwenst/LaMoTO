from ._general import CompareSentencesGLUETask

class MRPC(CompareSentencesGLUETask):
    def __init__(self):
        super().__init__("MRPC", num_labels=2)
