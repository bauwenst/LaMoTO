from ._general import CompareSentencesGLUETask

class QQP(CompareSentencesGLUETask):
    """Duplicate question detection, as is done on StackExchange."""
    def __init__(self):
        super().__init__("qqp", num_labels=2)
