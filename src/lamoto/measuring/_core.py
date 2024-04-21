from typing import Protocol, Any, Dict, Type
import evaluate


class Metric(Protocol):
    def compute(self, predictions: Any, references: Any) -> Dict[str,Any]:
        pass


class MetricRegistry:

    def __init__(self):
        self.custom_metrics: Dict[str,Type[Metric]] = dict()

    def registerMetric(self, name: str, metric: Type[Metric]):
        if name in self.custom_metrics:
            raise ValueError(f"Cannot register custom metric {name} because it already exists.")

        self.custom_metrics[name] = metric

    def load(self, name: str) -> Metric:
        return self.custom_metrics[name]() if name in self.custom_metrics else evaluate.load(name)
