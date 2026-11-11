from ._general import CompareSentencesSuperGLUETask


class BoolQ(CompareSentencesSuperGLUETask):
    """
    Yes/no question answering.
    Input looks like a (question, passage) pair.
    """
    def __init__(self):
        super().__init__(task_name="BoolQ", n_classes=2, text_field1="question", text_field2="passage")
