from ._general import CompareSentencesGLUETask

class MNLI(CompareSentencesGLUETask):  # TODO: What about the "MNLI-matched/unmatched" datasets?
    def __init__(self):
        super().__init__("mlni", num_labels=3)
