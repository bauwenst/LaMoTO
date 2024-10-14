from ._general import CompareSentencesGLUETask

class QQP(CompareSentencesGLUETask):
    """Duplicate question detection, as is done on StackExchange."""
    def __init__(self):
        super().__init__("qqp", num_labels=2, text_field1="question1", text_field2="question2")
