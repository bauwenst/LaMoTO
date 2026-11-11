from ._general import CompareSentencesGLUETask

class WNLI(CompareSentencesGLUETask):
    def __init__(self):
        super().__init__("WNLI", n_classes=2)
