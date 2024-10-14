from ._general import ClassifySentenceGLUETask

class CoLA(ClassifySentenceGLUETask):
    """Detection of erroneously formed sentences."""
    def __init__(self):
        super().__init__("cola")