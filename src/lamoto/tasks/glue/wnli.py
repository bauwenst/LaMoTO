from ._general import CompareSentencesGLUETask

class WNLI(CompareSentencesGLUETask):
    def __init__(self):
        super().__init__("WNLI", num_labels=2)
