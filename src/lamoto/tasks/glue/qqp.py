from ._general import CompareSentencesGLUETask, RankingMetricSpec

class QQP(CompareSentencesGLUETask):
    """Duplicate question detection, as is done on StackExchange."""
    def __init__(self):
        super().__init__("QQP", rank_by=RankingMetricSpec("f1", "f1", True),  # Uses F1 rather than accuracy because positives are underrepresented.
                         num_labels=2, text_field1="question1", text_field2="question2")
