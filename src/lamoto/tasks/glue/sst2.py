from ._general import ClassifySentenceGLUETask

class SST2(ClassifySentenceGLUETask):
    """Binary sentiment analysis."""
    def __init__(self):
        super().__init__("sst2")
