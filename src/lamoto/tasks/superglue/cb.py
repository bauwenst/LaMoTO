from ._general import CompareSentencesSuperGLUETask


class CB(CompareSentencesSuperGLUETask):
    """
    3-way NLI like MNLI.
    """
    def __init__(self):
        super().__init__("CB", num_labels=3, text_field1="premise", text_field2="hypothesis")
