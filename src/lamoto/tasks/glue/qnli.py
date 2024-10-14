from ._general import CompareSentencesGLUETask

class QNLI(CompareSentencesGLUETask):
    def __init__(self):
        super().__init__("qnli", num_labels=2, text_field1="question", text_field2="sentence")
