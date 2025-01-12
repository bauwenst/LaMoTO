from ._general import ClassifySentenceGLUETask, RankingMetricSpec

class SST2(ClassifySentenceGLUETask):
    """Binary sentiment analysis."""
    def __init__(self):
        super().__init__("SST-2", rank_by=RankingMetricSpec("accuracy", "accuracy", True))
