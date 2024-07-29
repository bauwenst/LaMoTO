from supar.utils.metric import AttachmentMetric
from typing import Any, Dict

from ._core import LogitLabelMetric


class DependencyParsingMetrics(LogitLabelMetric):

    def __init__(self):
        super().__init__(None)
        self.content = AttachmentMetric()

    def add(self, other: AttachmentMetric):
        self.content += other

    def compute(self, predictions: Any, references: Any) -> Dict[str,Any]:
        """
        Ignore the input and output your own internal metrics.
        """
        summary = {
            "uas": self.content.uas,
            "las": self.content.las,
            "ucm": self.content.ucm,
            "lcm": self.content.lcm
        }
        self.content = AttachmentMetric()
        return summary
