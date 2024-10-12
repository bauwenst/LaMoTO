from ._general import CompareSentencesGLUETask

class QNLI(CompareSentencesGLUETask):
    def __init__(self):
        super().__init__("qlni", num_labels=2)
