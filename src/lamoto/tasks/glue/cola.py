from ._general import ClassifySentenceGLUETask

class CoLA(ClassifySentenceGLUETask):
    """Detection of erroneously formed sentences."""
    def __init__(self):  # TODO: Apparently this also needs the matthews_correlation metric.
        super().__init__("cola")
